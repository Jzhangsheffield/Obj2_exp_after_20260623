#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampler.py

公共采样工具，用于 map-style 数据集的训练 DataLoader。

设计目标
========
1) 将采样逻辑从具体 dataloader 文件中剥离出来，避免 RGB / MindRove dataloader
   各自维护一份几乎相同的 weighted sampler 代码。

2) 同时支持两类采样器：
   - WeightedRandomSampler：控制“单个样本 index 被抽到的概率”。
   - ClassBalancedBatchSampler：直接控制“一个 batch 由哪些 index 组成”。

3) 本文件只依赖 dataset 的轻量元信息：
   - dataset.records
   - dataset.label_map
   - dataset.cfg.tier_mode

   因此构建 sampler 时不会触发 dataset.__getitem__，不会读取 rgb.pt、depth.pt、mindrove.pt，
   也不会执行任何数据增强。这一点对训练启动速度和工程可维护性都很重要。

使用建议
========
- 如果只是想缓解 epoch 级别的类别不平衡：
    sampler_type = "weighted"

- 如果使用 SupCon / MoCo，并且希望每个 batch 内稳定包含同类正样本：
    sampler_type = "balanced_batch"

注意事项
========
1) WeightedRandomSampler 只能在统计意义上提高少数类出现概率，不能保证每个 batch 内类别均衡。
2) ClassBalancedBatchSampler 会固定每个 batch 的结构：
       batch_size = classes_per_batch * samples_per_class
3) 本文件不实现 DistributedSampler 版本。若在 DDP 下使用 weighted / balanced_batch，
   需要额外实现 distributed-aware sampler，否则不同 rank 可能出现重复采样或 step 数不一致。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


VALID_TIERS = ("tier1", "tier2", "tier3")


def _require_packed_dataset_attrs(dataset: Any) -> None:
    """
    检查 dataset 是否具备本 sampler 模块所需的轻量属性。

    这里故意不检查 dataset 的具体 class 类型，因为 RGB / MindRove / Fusion 数据集
    只要遵循相同的 map-style 约定，就都可以复用同一套 sampler。
    """
    if not hasattr(dataset, "records"):
        raise TypeError("dataset must have attribute 'records'.")
    if not hasattr(dataset, "label_map"):
        raise TypeError("dataset must have attribute 'label_map'.")
    if not hasattr(dataset, "cfg"):
        raise TypeError("dataset must have attribute 'cfg'.")


def resolve_tier_for_sampling(dataset: Any, tier_for_sampling: Optional[str] = None) -> str:
    """
    决定采样时使用哪个 tier 的标签。

    参数
    ----
    dataset:
        map-style dataset，需包含 dataset.cfg.tier_mode。

    tier_for_sampling:
        显式指定采样 tier。可为 "tier1" / "tier2" / "tier3" / None。

    返回
    ----
    str:
        采样使用的 tier 名称。

    规则
    ----
    1) 如果 tier_for_sampling 不为 None，则直接使用它。
    2) 如果 tier_for_sampling 为 None，且 dataset.cfg.tier_mode 是具体 tier，
       则沿用 dataset.cfg.tier_mode。
    3) 如果 dataset.cfg.tier_mode == "all" 或其他非具体 tier，则默认使用 "tier3"。
       这样可以在多 tier 同时输出时使用最细粒度标签做采样。
    """
    _require_packed_dataset_attrs(dataset)

    if tier_for_sampling is None:
        cfg_tier = getattr(dataset.cfg, "tier_mode", None)
        if cfg_tier in VALID_TIERS:
            tier_for_sampling = str(cfg_tier)
        else:
            tier_for_sampling = "tier3"

    tier_for_sampling = str(tier_for_sampling)
    if tier_for_sampling not in VALID_TIERS:
        raise ValueError(
            f"tier_for_sampling must be one of {VALID_TIERS}, got {tier_for_sampling!r}"
        )

    if tier_for_sampling not in dataset.label_map:
        raise KeyError(f"dataset.label_map does not contain key {tier_for_sampling!r}")

    return tier_for_sampling


def extract_labels_from_packed_dataset(
    dataset: Any,
    tier_for_sampling: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    从 map-style dataset.records 中提取每个样本的整数类别 id。

    这个函数只读取 manifest 中已经加载到内存的 records，不调用 dataset.__getitem__。
    因此它不会加载任何真实模态文件，也不会触发数据增强。

    返回
    ----
    labels:
        LongTensor[N]，第 i 个元素是 dataset.records[i] 对应的 class id。

    info:
        调试信息，包含 tier_for_sampling、class_counts、bad_indices 等。
    """
    _require_packed_dataset_attrs(dataset)
    tier = resolve_tier_for_sampling(dataset, tier_for_sampling)
    tier_label_map = dataset.label_map[tier]

    labels: List[int] = []
    bad_indices: List[int] = []

    for i, rec in enumerate(dataset.records):
        action_name = rec.get(tier, None)
        if action_name is None:
            labels.append(-1)
            bad_indices.append(i)
            continue

        class_id = tier_label_map.get(str(action_name), -1)
        if int(class_id) < 0:
            bad_indices.append(i)

        labels.append(int(class_id))

    if bad_indices:
        raise ValueError(
            f"Found {len(bad_indices)} samples with invalid label for {tier}. "
            f"Example bad indices: {bad_indices[:10]}"
        )

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    if labels_tensor.numel() == 0:
        raise RuntimeError("No valid labels found in dataset.records.")

    class_counts = Counter(labels_tensor.tolist())
    if len(class_counts) == 0:
        raise RuntimeError("No valid class counts can be built from labels.")

    info: Dict[str, Any] = {
        "tier_for_sampling": tier,
        "labels": labels_tensor,
        "class_counts": class_counts,
        "num_samples": int(labels_tensor.numel()),
        "num_classes_in_split": int(len(class_counts)),
        "bad_indices": bad_indices,
    }
    return labels_tensor, info


def build_class_to_indices(labels: torch.Tensor) -> Dict[int, List[int]]:
    """
    建立 class_id -> sample_indices 的索引表。

    参数
    ----
    labels:
        LongTensor[N]，每个样本的类别 id。

    返回
    ----
    dict[int, list[int]]:
        例如：
            {
                0: [0, 5, 9, ...],
                1: [1, 2, 8, ...],
            }
    """
    if not torch.is_tensor(labels):
        raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D tensor, got shape={tuple(labels.shape)}")

    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, cls_id in enumerate(labels.detach().cpu().tolist()):
        cls_id = int(cls_id)
        if cls_id < 0:
            raise ValueError(f"labels contains invalid class id {cls_id} at index {idx}")
        class_to_indices[cls_id].append(int(idx))

    return dict(class_to_indices)


def summarize_sampling_labels(labels: torch.Tensor) -> Dict[str, Any]:
    """
    汇总标签分布，便于日志记录和调试。
    """
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")

    counter = Counter(labels.detach().cpu().tolist())
    counts_sorted = dict(sorted((int(k), int(v)) for k, v in counter.items()))

    return {
        "num_samples": int(labels.numel()),
        "num_classes_in_split": int(len(counter)),
        "class_counts": counter,
        "class_counts_sorted": counts_sorted,
    }


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    直接生成 class-balanced batch 的 BatchSampler。

    与 WeightedRandomSampler 的区别
    =============================
    WeightedRandomSampler 每次只返回一个 sample index，DataLoader 再把若干 index
    组合成 batch。因此它只能在长期统计上改变类别采样概率，不能保证每个 batch 内
    有多少类、每类多少样本。

    ClassBalancedBatchSampler 每次直接返回一个完整的 batch index list，形如：
        [cls0_idx_a, cls0_idx_b, cls1_idx_a, cls1_idx_b, ...]

    因此它可以保证：
        batch_size = classes_per_batch * samples_per_class

    这对 SupCon / supervised contrastive loss 更友好，因为每个类别在 batch 内至少
    有 samples_per_class 个样本，batch 内正样本结构更稳定。

    replacement 的语义
    ==================
    这里的 replacement 主要控制“同一个 batch 内，从某个类别下抽样本时是否有放回”。
    - replacement=False：要求每个被采样类别至少有 samples_per_class 个样本。
    - replacement=True ：即使某个类别样本数少于 samples_per_class，也可以重复抽样。

    类别选择方面：
    - 如果 classes_per_batch <= 当前 split 中类别数，则每个 batch 内默认采样不重复类别。
    - 如果 classes_per_batch > 当前 split 中类别数，则只有 replacement=True 时才允许类别重复。

    注意：
    本实现没有保证一个 epoch 内样本不重复。它的目标是 batch-level class balance，
    不是 strict epoch-level without-replacement traversal。
    """

    def __init__(
        self,
        labels: torch.Tensor,
        classes_per_batch: int,
        samples_per_class: int,
        num_batches: Optional[int] = None,
        replacement: bool = True,
        drop_last: bool = True,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        if not torch.is_tensor(labels):
            raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={tuple(labels.shape)}")
        if int(classes_per_batch) <= 0:
            raise ValueError(f"classes_per_batch must be positive, got {classes_per_batch}")
        if int(samples_per_class) <= 0:
            raise ValueError(f"samples_per_class must be positive, got {samples_per_class}")

        self.labels = labels.detach().cpu().long().clone()
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self.epoch = 0
        self.verbose = bool(verbose)

        self.class_to_indices = build_class_to_indices(self.labels)
        self.class_ids = sorted(int(c) for c in self.class_to_indices.keys())
        self.num_classes = len(self.class_ids)
        self.num_samples = int(self.labels.numel())

        if self.num_classes == 0:
            raise RuntimeError("ClassBalancedBatchSampler received no classes.")

        if self.classes_per_batch > self.num_classes and not self.replacement:
            raise ValueError(
                "classes_per_batch is larger than the number of classes in this split. "
                "Set replacement=True or reduce classes_per_batch. "
                f"classes_per_batch={self.classes_per_batch}, num_classes={self.num_classes}"
            )

        if not self.replacement:
            too_small = {
                cls_id: len(indices)
                for cls_id, indices in self.class_to_indices.items()
                if len(indices) < self.samples_per_class
            }
            if too_small:
                raise ValueError(
                    "replacement=False requires every class to have at least "
                    f"samples_per_class={self.samples_per_class} samples. "
                    f"Too-small classes: {too_small}"
                )

        if num_batches is None:
            if self.drop_last:
                num_batches = self.num_samples // self.batch_size
            else:
                # BatchSampler 每次仍然输出固定大小 batch；这里的 ceil 只决定 epoch 长度。
                num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        self.num_batches = int(num_batches)
        if self.num_batches <= 0:
            raise ValueError(
                f"num_batches must be positive. Got {self.num_batches}. "
                f"num_samples={self.num_samples}, batch_size={self.batch_size}"
            )

        if self.verbose:
            counts = dict(sorted((k, len(v)) for k, v in self.class_to_indices.items()))
            print("[ClassBalancedBatchSampler] enabled")
            print(f"[ClassBalancedBatchSampler] num_samples = {self.num_samples}")
            print(f"[ClassBalancedBatchSampler] num_classes = {self.num_classes}")
            print(f"[ClassBalancedBatchSampler] classes_per_batch = {self.classes_per_batch}")
            print(f"[ClassBalancedBatchSampler] samples_per_class = {self.samples_per_class}")
            print(f"[ClassBalancedBatchSampler] batch_size = {self.batch_size}")
            print(f"[ClassBalancedBatchSampler] num_batches = {self.num_batches}")
            print(f"[ClassBalancedBatchSampler] replacement = {self.replacement}")
            print(f"[ClassBalancedBatchSampler] class_counts = {counts}")

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch，用于可复现地改变每个 epoch 的采样随机性。

        训练主循环中如果已有：
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        则无需额外修改训练循环。
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _make_generator(self) -> torch.Generator:
        """为当前 epoch 创建独立 torch.Generator。"""
        g = torch.Generator()
        if self.seed is None:
            # 不固定 seed 时，仍然让 torch 自己生成随机种子。
            g.seed()
        else:
            # 加上 epoch，保证不同 epoch 的 batch 组合不同，同时可复现。
            g.manual_seed(int(self.seed) + int(self.epoch))
        return g

    def _sample_classes(self, generator: torch.Generator) -> List[int]:
        """采样当前 batch 使用的类别 id。"""
        class_tensor = torch.as_tensor(self.class_ids, dtype=torch.long)

        if self.classes_per_batch <= self.num_classes:
            perm = torch.randperm(self.num_classes, generator=generator)
            chosen = class_tensor[perm[: self.classes_per_batch]]
            return [int(x) for x in chosen.tolist()]

        # classes_per_batch > num_classes 只在 replacement=True 时允许。
        rand_pos = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.classes_per_batch,),
            generator=generator,
        )
        chosen = class_tensor[rand_pos]
        return [int(x) for x in chosen.tolist()]

    def _sample_indices_from_class(self, cls_id: int, generator: torch.Generator) -> List[int]:
        """从某一类中采样 samples_per_class 个样本 index。"""
        indices = self.class_to_indices[int(cls_id)]
        n = len(indices)

        if self.replacement:
            pos = torch.randint(
                low=0,
                high=n,
                size=(self.samples_per_class,),
                generator=generator,
            )
            return [int(indices[int(p)]) for p in pos.tolist()]

        perm = torch.randperm(n, generator=generator)[: self.samples_per_class]
        return [int(indices[int(p)]) for p in perm.tolist()]

    def __iter__(self) -> Iterator[List[int]]:
        generator = self._make_generator()

        for _ in range(self.num_batches):
            batch_indices: List[int] = []
            chosen_classes = self._sample_classes(generator)

            for cls_id in chosen_classes:
                batch_indices.extend(self._sample_indices_from_class(cls_id, generator))

            # 打乱 batch 内顺序，避免同一类别样本总是相邻。这样不会改变类别配比，
            # 但可以减少模型或 BatchNorm 对固定排列模式的依赖。
            perm = torch.randperm(len(batch_indices), generator=generator).tolist()
            batch_indices = [batch_indices[int(i)] for i in perm]

            yield batch_indices


def build_weighted_sampler_for_packed_dataset(
    dataset: Any,
    tier_for_sampling: Optional[str] = None,
    mode: str = "sqrt_inv",
    replacement: bool = True,
    num_samples: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """
    为 map-style dataset 构建 WeightedRandomSampler。

    参数
    ----
    mode:
        类别权重构造方式：
        - "inv"      : class_weight = 1 / count
        - "sqrt_inv" : class_weight = 1 / sqrt(count)

    返回
    ----
    sampler:
        可传给 DataLoader(..., sampler=sampler, shuffle=False)。

    info:
        采样统计信息，用于日志记录。
    """
    labels, info = extract_labels_from_packed_dataset(dataset, tier_for_sampling)
    counter = info["class_counts"]

    max_class_id = max(int(k) for k in counter.keys())
    class_counts_tensor = torch.zeros(max_class_id + 1, dtype=torch.long)
    for cls_id, cnt in counter.items():
        class_counts_tensor[int(cls_id)] = int(cnt)

    class_weights = torch.zeros_like(class_counts_tensor, dtype=torch.double)
    for cls_id, cnt in counter.items():
        cnt = int(cnt)
        if cnt <= 0:
            raise ValueError(f"Invalid class count for class {cls_id}: {cnt}")

        if mode == "inv":
            weight = 1.0 / float(cnt)
        elif mode == "sqrt_inv":
            weight = 1.0 / (float(cnt) ** 0.5)
        else:
            raise ValueError(f"Unsupported mode: {mode!r}. Use 'inv' or 'sqrt_inv'.")

        class_weights[int(cls_id)] = float(weight)

    sample_weights = class_weights[labels].to(torch.double)

    if num_samples is None:
        num_samples = len(dataset)
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=bool(replacement),
    )

    info.update(
        {
            "mode": mode,
            "replacement": bool(replacement),
            "class_counts_tensor": class_counts_tensor,
            "class_weights": class_weights,
            "sample_weights": sample_weights,
            "num_samples": num_samples,
        }
    )

    if verbose:
        print("[WeightedSampler] enabled")
        print(f"[WeightedSampler] tier_for_sampling = {info['tier_for_sampling']}")
        print(f"[WeightedSampler] mode = {mode}")
        print(f"[WeightedSampler] replacement = {bool(replacement)}")
        print(f"[WeightedSampler] num_samples = {num_samples}")
        print(f"[WeightedSampler] num_classes_in_split = {len(counter)}")
        print(f"[WeightedSampler] class_counts = {dict(sorted(counter.items()))}")

    return sampler, info


def build_class_balanced_batch_sampler_for_packed_dataset(
    dataset: Any,
    tier_for_sampling: Optional[str] = None,
    classes_per_batch: int = 16,
    samples_per_class: int = 2,
    num_batches: Optional[int] = None,
    replacement: bool = True,
    drop_last: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[ClassBalancedBatchSampler, Dict[str, Any]]:
    """
    为 map-style dataset 构建 ClassBalancedBatchSampler。

    这个函数是训练脚本推荐调用的入口。它负责：
    1) 从 dataset.records 中提取标签；
    2) 构建 ClassBalancedBatchSampler；
    3) 返回 sampler 和日志信息。
    """
    labels, info = extract_labels_from_packed_dataset(dataset, tier_for_sampling)

    batch_sampler = ClassBalancedBatchSampler(
        labels=labels,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        num_batches=num_batches,
        replacement=replacement,
        drop_last=drop_last,
        seed=seed,
        verbose=verbose,
    )

    info.update(
        {
            "classes_per_batch": int(classes_per_batch),
            "samples_per_class": int(samples_per_class),
            "batch_size": int(classes_per_batch) * int(samples_per_class),
            "num_batches": len(batch_sampler),
            "replacement": bool(replacement),
            "drop_last": bool(drop_last),
            "seed": seed,
        }
    )

    return batch_sampler, info
