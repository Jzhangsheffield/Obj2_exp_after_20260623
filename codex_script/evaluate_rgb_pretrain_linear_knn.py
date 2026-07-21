#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-probe and kNN evaluation for RGB contrastive pretraining checkpoints.

Evaluates both:
  - feat512: encoder_q pooled backbone feature
  - proj128: encoder_q projection-head output

The output is one summary CSV for all N_as_test take/exclude-take-put
checkpoint_0200.pth files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import umap_rgb_pretrain_checkpoint_by_label as pretrain
import umap_rgb_lastpth_by_label as base


def compute_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    reverse_label_map: Dict[int, str],
) -> Dict[str, Any]:
    y_true = torch.as_tensor(y_true, dtype=torch.long).view(-1)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long).view(-1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        ti = int(t.item())
        pi = int(p.item())
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1

    cm_f = cm.float()
    tp = torch.diag(cm_f)
    support = cm_f.sum(dim=1)
    pred_count = cm_f.sum(dim=0)
    present = support > 0

    recall = torch.zeros(num_classes)
    precision = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)
    recall[present] = tp[present] / torch.clamp(support[present], min=1.0)
    pred_nonzero = pred_count > 0
    precision[pred_nonzero] = tp[pred_nonzero] / torch.clamp(pred_count[pred_nonzero], min=1.0)
    denom = precision + recall
    valid = denom > 0
    f1[valid] = 2.0 * precision[valid] * recall[valid] / denom[valid]

    total = int(y_true.numel())
    correct = int((y_true == y_pred).sum().item())
    per_class_acc: Dict[str, Any] = {}
    per_class_f1: Dict[str, Any] = {}
    for c in range(num_classes):
        name = reverse_label_map.get(c, str(c))
        key = f"{c}:{name}"
        per_class_acc[key] = float(recall[c].item()) if support[c] > 0 else None
        per_class_f1[key] = float(f1[c].item()) if support[c] > 0 else None

    return {
        "acc": correct / max(1, total),
        "balanced_acc": float(recall[present].mean().item()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean().item()) if present.any() else 0.0,
        "num_samples": total,
        "correct": correct,
        "per_class_acc_json": json.dumps(per_class_acc, ensure_ascii=False, separators=(",", ":")),
        "per_class_f1_json": json.dumps(per_class_f1, ensure_ascii=False, separators=(",", ":")),
        "confusion_matrix_json": json.dumps(cm.tolist(), ensure_ascii=False, separators=(",", ":")),
    }


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features.float().contiguous()
        self.labels = labels.long().contiguous()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    reverse_label_map: Dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    clf = nn.Linear(int(train_features.shape[1]), num_classes).to(device)
    if args.linear_probe_optimizer == "sgd":
        optimizer = optim.SGD(
            clf.parameters(),
            lr=args.linear_probe_lr,
            momentum=args.linear_probe_momentum,
            weight_decay=args.linear_probe_weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            clf.parameters(),
            lr=args.linear_probe_lr,
            weight_decay=args.linear_probe_weight_decay,
        )
    loader = torch.utils.data.DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=args.linear_probe_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()
    clf.train()
    for _epoch in range(args.linear_probe_epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(clf(xb), yb)
            loss.backward()
            optimizer.step()

    clf.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, test_features.shape[0], args.eval_chunk_size):
            xb = test_features[start:start + args.eval_chunk_size].to(device)
            preds.append(clf(xb).argmax(dim=1).cpu())
    metrics = compute_classification_metrics(test_labels, torch.cat(preds), num_classes, reverse_label_map)
    return {
        "linear_probe_acc": metrics["acc"],
        "linear_probe_balanced_acc": metrics["balanced_acc"],
        "linear_probe_macro_f1": metrics["macro_f1"],
        "linear_probe_correct": metrics["correct"],
        "linear_probe_num_samples": metrics["num_samples"],
        "linear_probe_per_class_acc_json": metrics["per_class_acc_json"],
        "linear_probe_per_class_f1_json": metrics["per_class_f1_json"],
        "linear_probe_confusion_matrix_json": metrics["confusion_matrix_json"],
        "linear_probe_epochs": args.linear_probe_epochs,
        "linear_probe_lr": args.linear_probe_lr,
        "linear_probe_optimizer": args.linear_probe_optimizer,
    }


def compute_knn_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    k: int,
    num_classes: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    k = min(int(k), int(train_features.shape[0]))
    train_n = torch.nn.functional.normalize(train_features, dim=1) if args.knn_metric == "cosine" else train_features
    preds: List[torch.Tensor] = []
    for start in range(0, test_features.shape[0], args.eval_chunk_size):
        x = test_features[start:start + args.eval_chunk_size]
        if args.knn_metric == "cosine":
            x_n = torch.nn.functional.normalize(x, dim=1)
            scores, idx = torch.topk(x_n @ train_n.T, k=k, dim=1, largest=True)
        else:
            dist = torch.cdist(x, train_n, p=2)
            vals, idx = torch.topk(dist, k=k, dim=1, largest=False)
            scores = -vals
        nn_labels = train_labels[idx]
        if args.knn_weighted:
            weights = torch.softmax(scores / max(args.knn_temperature, 1e-12), dim=1)
        else:
            weights = torch.ones_like(scores)
        votes = torch.zeros((x.shape[0], num_classes), dtype=torch.float32)
        votes.scatter_add_(dim=1, index=nn_labels.long(), src=weights.float())
        preds.append(votes.argmax(dim=1).long())
    return torch.cat(preds, dim=0)


def evaluate_representation(
    group: pretrain.PretrainGroup,
    checkpoint_slug: str,
    checkpoint_path: Path,
    load_report: Dict[str, Any],
    rep_name: str,
    train_features_raw: torch.Tensor,
    test_features_raw: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    reverse_label_map: Dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if args.feature_l2_normalize:
        train_features = torch.nn.functional.normalize(train_features_raw, dim=1)
        test_features = torch.nn.functional.normalize(test_features_raw, dim=1)
    else:
        train_features = train_features_raw.float()
        test_features = test_features_raw.float()

    row: Dict[str, Any] = {
        "group": group.key,
        "checkpoint_slug": checkpoint_slug,
        "checkpoint_path": str(checkpoint_path),
        "representation": rep_name,
        "feature_dim": int(train_features.shape[1]),
        "num_reference_train_samples": int(train_features.shape[0]),
        "num_test_samples": int(test_features.shape[0]),
        "num_classes": int(group.num_classes),
        "feature_l2_normalize": bool(args.feature_l2_normalize),
        "epoch": load_report.get("epoch", ""),
        "ablation_mode": load_report.get("ablation_mode", ""),
        "contrastive_loss_mode": load_report.get("contrastive_loss_mode", ""),
        "loaded_encoder_q_keys": load_report.get("loaded_encoder_q_keys", ""),
    }
    row.update(train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=group.num_classes,
        reverse_label_map=reverse_label_map,
        args=args,
        device=device,
    ))

    for k in args.knn_k:
        pred = compute_knn_predictions(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            k=int(k),
            num_classes=group.num_classes,
            args=args,
        )
        metrics = compute_classification_metrics(test_labels, pred, group.num_classes, reverse_label_map)
        prefix = f"knn_k{int(k)}"
        row[f"{prefix}_acc"] = metrics["acc"]
        row[f"{prefix}_balanced_acc"] = metrics["balanced_acc"]
        row[f"{prefix}_macro_f1"] = metrics["macro_f1"]
        row[f"{prefix}_correct"] = metrics["correct"]
        row[f"{prefix}_num_samples"] = metrics["num_samples"]
        row[f"{prefix}_per_class_acc_json"] = metrics["per_class_acc_json"]
        row[f"{prefix}_per_class_f1_json"] = metrics["per_class_f1_json"]
        row[f"{prefix}_confusion_matrix_json"] = metrics["confusion_matrix_json"]
    return row


def parse_group_keys(values: Sequence[str] | None) -> List[str]:
    if not values:
        return [g.key for g in pretrain.GROUPS]
    requested = [str(v).strip().lower() for v in values]
    valid = {g.key for g in pretrain.GROUPS}
    bad = [v for v in requested if v not in valid]
    if bad:
        raise ValueError(f"Unknown group(s): {bad}. Valid: {sorted(valid)}")
    return requested


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RGB pretrain checkpoint_0200 features with linear probe and kNN.")
    parser.add_argument("--dataset_root", type=Path, default=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset"))
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "analysis" / "N_as_test" / "umap_rgb_pretrain")
    parser.add_argument("--summary_name", type=str, default="pretrain_linear_probe_knn_summary.csv")
    parser.add_argument("--groups", nargs="*", default=None, help="Subset of groups: excl take")
    parser.add_argument("--tier_mode", default="tier1", choices=["tier1", "tier2", "tier3"])
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--model_depth", type=int, default=18)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--rgb_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--feature_l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear_probe_epochs", type=int, default=100)
    parser.add_argument("--linear_probe_lr", type=float, default=1e-2)
    parser.add_argument("--linear_probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--linear_probe_momentum", type=float, default=0.9)
    parser.add_argument("--linear_probe_batch_size", type=int, default=256)
    parser.add_argument("--linear_probe_optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--eval_chunk_size", type=int, default=4096)
    parser.add_argument("--knn_k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--knn_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--knn_weighted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--knn_temperature", type=float, default=0.07)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir = args.output_dir.resolve()
    base.ensure_dir(args.output_dir)
    base.seed_everything(args.seed)
    base.save_json(args.output_dir / "pretrain_linear_probe_knn_config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "project_root": str(PROJECT_ROOT),
        "representations": ["feat512", "proj128"],
    })

    selected = set(parse_group_keys(args.groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[output] {args.output_dir}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for group in pretrain.GROUPS:
        if group.key not in selected:
            continue
        print(f"\n[group] {group.key}")
        reverse_label_map = base.build_reverse_label_map(str(group.label_map_json), args.tier_mode)
        loaders = {
            "train": pretrain.build_loader(group, group.train_manifest, args),
            "test": pretrain.build_loader(group, group.test_manifest, args),
        }
        checkpoints = pretrain.find_checkpoints(group)
        print(f"[checkpoints] found {len(checkpoints)}")
        for ckpt_path in checkpoints:
            slug = pretrain.checkpoint_slug(ckpt_path, group)
            print(f"\n[checkpoint] {group.key} | {slug}")
            model = pretrain.make_pretrain_encoder(args, device)
            load_report = pretrain.load_encoder_q_checkpoint(model, ckpt_path)
            print(json.dumps(load_report, indent=2))
            train_reps_np, train_labels_np, _train_ids = pretrain.extract_representations(model, loaders["train"], device, args.tier_mode, "train")
            test_reps_np, test_labels_np, _test_ids = pretrain.extract_representations(model, loaders["test"], device, args.tier_mode, "test")
            train_labels = torch.as_tensor(train_labels_np, dtype=torch.long)
            test_labels = torch.as_tensor(test_labels_np, dtype=torch.long)
            for rep_name in ("feat512", "proj128"):
                print(f"[metric] {group.key} | {slug} | {rep_name} linear_probe + kNN")
                row = evaluate_representation(
                    group=group,
                    checkpoint_slug=slug,
                    checkpoint_path=ckpt_path,
                    load_report=load_report,
                    rep_name=rep_name,
                    train_features_raw=torch.as_tensor(train_reps_np[rep_name], dtype=torch.float32),
                    test_features_raw=torch.as_tensor(test_reps_np[rep_name], dtype=torch.float32),
                    train_labels=train_labels,
                    test_labels=test_labels,
                    reverse_label_map=reverse_label_map,
                    args=args,
                    device=device,
                )
                rows.append(row)
                base.save_csv(args.output_dir / "pretrain_linear_probe_knn_summary_partial.csv", rows)

    summary_csv = args.output_dir / args.summary_name
    summary_json = args.output_dir / "pretrain_linear_probe_knn_summary.json"
    base.save_csv(summary_csv, rows)
    base.save_json(summary_json, {"rows": rows, "elapsed_seconds": time.time() - t0})
    print(f"\n[done] elapsed seconds: {time.time() - t0:.1f}")
    print(f"[summary csv] {summary_csv}")
    print(f"[summary json] {summary_json}")


if __name__ == "__main__":
    main()
