#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP visualisation for RGB finetuned last.pth checkpoints.

This script scans the two N_as_test RGB finetuning result folders, extracts
frozen features from every last.pth checkpoint, and saves label-coloured UMAP
plots for train, test, and combined train+test splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backbone.resnet as resnet3d
from utils_.mapstype_dataloader_with_index_mindrove_modified_varlen import (
    PackedMultiModalConfig,
    build_packed_mapstyle_dataset,
    build_packed_mapstyle_loader_from_dataset,
    load_label_map_json,
)


@dataclass(frozen=True)
class GroupConfig:
    key: str
    result_dir: Path
    label_map_json: Path
    train_manifest: str
    test_manifest: str
    num_classes: int
    rgb_mean: Tuple[float, float, float]
    rgb_std: Tuple[float, float, float]


GROUPS: Tuple[GroupConfig, ...] = (
    GroupConfig(
        key="excl",
        result_dir=PROJECT_ROOT / "results" / "ft_rgb_N_except_take_put_adamw_22_seed1",
        label_map_json=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset\label_map_except_take_put.json"),
        train_manifest="N_as_test/train_manifest_except_take_put.jsonl",
        test_manifest="N_as_test/test_manifest_except_take_put.jsonl",
        num_classes=15,
        rgb_mean=(0.3752, 0.3864, 0.3960),
        rgb_std=(0.2934, 0.2724, 0.2644),
    ),
    GroupConfig(
        key="take",
        result_dir=PROJECT_ROOT / "results" / "ft_rgb_N_take_put_adamw_22_seed1",
        label_map_json=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset\label_map_take_put.json"),
        train_manifest="N_as_test/train_manifest_take_put.jsonl",
        test_manifest="N_as_test/test_manifest_take_put.jsonl",
        num_classes=2,
        rgb_mean=(0.3725, 0.3828, 0.3921),
        rgb_std=(0.2923, 0.2715, 0.2640),
    ),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("._-")
    cleaned = cleaned or "item"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._-")
    return cleaned or "item"


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_reverse_label_map(label_map_json_path: Path, tier_mode: str) -> Dict[int, str]:
    label_map = load_label_map_json(str(label_map_json_path))
    if tier_mode not in label_map:
        raise KeyError(f"tier_mode '{tier_mode}' not found in {label_map_json_path}")
    return {int(v): str(k) for k, v in label_map[tier_mode].items()}


def label_names_from_ids(labels: np.ndarray, reverse_label_map: Dict[int, str]) -> np.ndarray:
    return np.array([reverse_label_map.get(int(x), str(int(x))) for x in labels], dtype=str)


def make_cfg(group: GroupConfig, args: argparse.Namespace) -> PackedMultiModalConfig:
    return PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=False,
        use_modalities=("rgb",),
        missing_policy="skip",
        load_labels=True,
        label_map_path=str(group.label_map_json),
        tier_mode=args.tier_mode,
        is_train=False,
        rgb_out_hw=(args.rgb_size, args.rgb_size),
        rgb_mean=group.rgb_mean,
        rgb_std=group.rgb_std,
        rrc_scale=(0.6, 1.0),
        rrc_ratio=(0.75, 1.3333333333),
        rgb_apply_spatial_aug=False,
        rgb_hflip_p=0.0,
        rgb_vflip_p=0.0,
        rgb_jitter_p=0.0,
        rgb_gray_p=0.0,
        rgb_blur_p=0.0,
        rgb_blur_kernel=5,
        rgb_blur_sigma=(0.1, 1.0),
        depth_out_hw=(args.rgb_size, args.rgb_size),
        default_rgb_hw=(256, 256),
        default_depth_hw=(args.rgb_size, args.rgb_size),
    )


def build_loader(group: GroupConfig, manifest: str, args: argparse.Namespace):
    label_map = load_label_map_json(str(group.label_map_json))
    dataset = build_packed_mapstyle_dataset(
        dataset_root=str(args.dataset_root),
        manifest_name=manifest,
        cfg=make_cfg(group, args),
        label_map=label_map,
        verify_paths_on_init=True,
    )
    loader = build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
        sampler=None,
        pin_memory=args.pin_memory,
    )
    return loader


def strip_prefixes_from_key(key: str) -> str:
    prefixes = (
        "module.", "model.", "backbone.", "encoder_q.", "encoder_k.",
        "encoder.", "base_encoder.", "online_encoder.", "network.",
        "student.", "teacher.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict_from_checkpoint(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint object must be dict-like.")
    for key in ("model_state_dict", "state_dict", "model", "net", "network"):
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    if any(isinstance(k, str) and torch.is_tensor(v) for k, v in ckpt_obj.items()):
        return ckpt_obj
    raise ValueError("Could not find a state_dict in checkpoint.")


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    raw_state = extract_state_dict_from_checkpoint(ckpt)
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    dropped_missing: List[str] = []
    dropped_shape: List[str] = []
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        new_key = strip_prefixes_from_key(key)
        if new_key not in model_state:
            dropped_missing.append(new_key)
            continue
        if tuple(value.shape) != tuple(model_state[new_key].shape):
            dropped_shape.append(new_key)
            continue
        filtered[new_key] = value
    msg = model.load_state_dict(filtered, strict=False)
    return {
        "checkpoint": str(ckpt_path),
        "loaded_keys": len(filtered),
        "raw_tensor_keys": sum(1 for v in raw_state.values() if torch.is_tensor(v)),
        "dropped_missing_count": len(dropped_missing),
        "dropped_shape_count": len(dropped_shape),
        "missing_after_load_count": len(msg.missing_keys),
        "unexpected_after_load_count": len(msg.unexpected_keys),
    }


def extract_inputs_labels_ids(batch: Dict[str, Any], tier_mode: str):
    labels = batch["tier_ids"][tier_mode]
    sample_ids = batch.get("key", None) or batch.get("sample_name", None)
    if sample_ids is None:
        sample_ids = [str(i) for i in range(int(labels.shape[0]))]
    if not isinstance(sample_ids, list):
        sample_ids = list(sample_ids)
    return batch["rgb"], labels, sample_ids


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
    tier_mode: str,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    feats_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    ids_all: List[str] = []
    for batch in tqdm.tqdm(loader, desc=f"extract:{split}", dynamic_ncols=True):
        inputs, labels, sample_ids = extract_inputs_labels_ids(batch, tier_mode)
        inputs = inputs.to(device, non_blocking=True).float()
        if inputs.ndim != 5:
            raise ValueError(f"Expected RGB tensor [B,T,C,H,W], got {tuple(inputs.shape)}")
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        feats = model.forward_features(inputs) if hasattr(model, "forward_features") else model(inputs)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        feats_all.append(feats.detach().cpu().float())
        labels_all.append(labels.detach().cpu().long().view(-1))
        ids_all.extend([str(x) for x in sample_ids])
    features = torch.cat(feats_all, dim=0).numpy().astype(np.float32)
    labels_np = torch.cat(labels_all, dim=0).numpy().astype(np.int64)
    return features, labels_np, np.array(ids_all, dtype=str)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (x / denom).astype(np.float32)


def maybe_pca(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    if n_components <= 0 or x.shape[1] <= n_components:
        return x.astype(np.float32, copy=False)
    from sklearn.decomposition import PCA
    n_comp = min(n_components, x.shape[0], x.shape[1])
    if n_comp < 2:
        return x.astype(np.float32, copy=False)
    return PCA(n_components=n_comp, random_state=seed).fit_transform(x).astype(np.float32)


def compute_umap(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap
    if x.shape[0] < 3:
        raise ValueError("UMAP needs at least 3 samples.")
    n_neighbors = min(args.umap_n_neighbors, x.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def plot_by_label(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    title: str,
    out_path: Path,
    point_size: float,
    alpha: float,
) -> None:
    ensure_dir(out_path.parent)
    unique_labels = list(dict.fromkeys(labels.astype(int).tolist()))
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels))) if len(unique_labels) <= 20 else plt.get_cmap("gist_ncar", len(unique_labels))
    color_map = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    fig_w = 11.0 if len(unique_labels) > 4 else 9.0
    fig, ax = plt.subplots(figsize=(fig_w, 7.0), dpi=180)
    for lab in unique_labels:
        mask = labels.astype(int) == lab
        legend_name = str(label_names[mask][0]) if np.any(mask) else str(lab)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color_map[lab]],
            label=legend_name,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_coords_csv(
    path: Path,
    coords: np.ndarray,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> None:
    rows = []
    for i in range(coords.shape[0]):
        rows.append({
            "sample_id": str(sample_ids[i]),
            "split": str(splits[i]),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "true_label_id": int(labels[i]),
            "true_label_name": str(label_names[i]),
        })
    save_csv(path, rows)


def find_last_checkpoints(group: GroupConfig) -> List[Path]:
    weights_dir = group.result_dir / "weights"
    paths = sorted(p for p in weights_dir.rglob("last.pth") if "_batch_test" not in p.parts)
    return paths


def run_slug(weight_path: Path, group: GroupConfig) -> str:
    rel_parent = weight_path.parent.relative_to(group.result_dir / "weights")
    raw = "_".join(rel_parent.parts)
    digest = hashlib.sha1(str(weight_path).encode("utf-8", errors="ignore")).hexdigest()[:6]
    return f"{sanitize_name(raw, 42)}_{digest}"


def make_model(num_classes: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = resnet3d.generate_model(
        args.model_depth,
        n_input_channels=3,
        num_classes=num_classes,
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize_split(
    out_dir: Path,
    group_key: str,
    weight_slug: str,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    label_names: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, str]:
    x = l2_normalize(features) if args.feature_l2_normalize else features.astype(np.float32, copy=False)
    x = maybe_pca(x, args.pca_dim, args.seed)
    coords = compute_umap(x, args)
    coords_csv = out_dir / f"{split_name}_umap_coords.csv"
    png = out_dir / f"{split_name}_umap_by_label_legend.png"
    splits = np.array([split_name] * len(labels), dtype=str)
    if split_name == "combined":
        split_marker = np.array(["train"] * args._combined_train_n + ["test"] * (len(labels) - args._combined_train_n), dtype=str)
        splits = split_marker
    save_coords_csv(coords_csv, coords, sample_ids, splits, labels, label_names)
    plot_by_label(
        coords,
        labels,
        label_names,
        title=f"{group_key} | {weight_slug} | {split_name} | UMAP by true label",
        out_path=png,
        point_size=args.plot_point_size,
        alpha=args.plot_alpha,
    )
    return {"coords_csv": str(coords_csv), "png": str(png)}


def analyze_weight(
    group: GroupConfig,
    weight_path: Path,
    loaders: Dict[str, Any],
    reverse_label_map: Dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    slug = run_slug(weight_path, group)
    out_dir = args.output_dir / group.key / slug
    ensure_dir(out_dir)
    print(f"\n[weight] {group.key} | {slug}")
    print(f"[path] {weight_path}")
    model = make_model(group.num_classes, args, device)
    load_report = load_checkpoint(model, weight_path)
    print(json.dumps(load_report, indent=2))

    train_features, train_labels, train_ids = extract_features(model, loaders["train"], device, args.tier_mode, "train")
    test_features, test_labels, test_ids = extract_features(model, loaders["test"], device, args.tier_mode, "test")
    train_names = label_names_from_ids(train_labels, reverse_label_map)
    test_names = label_names_from_ids(test_labels, reverse_label_map)

    outputs: Dict[str, Any] = {}
    outputs["train"] = visualize_split(out_dir, group.key, slug, "train", train_features, train_labels, train_ids, train_names, args)
    outputs["test"] = visualize_split(out_dir, group.key, slug, "test", test_features, test_labels, test_ids, test_names, args)

    args._combined_train_n = int(train_features.shape[0])
    combined_features = np.concatenate([train_features, test_features], axis=0)
    combined_labels = np.concatenate([train_labels, test_labels], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)
    combined_names = np.concatenate([train_names, test_names], axis=0)
    outputs["combined"] = visualize_split(out_dir, group.key, slug, "combined", combined_features, combined_labels, combined_ids, combined_names, args)

    meta = {
        "group": group.key,
        "weight_path": str(weight_path),
        "weight_slug": slug,
        "num_classes": group.num_classes,
        "rgb_mean": group.rgb_mean,
        "rgb_std": group.rgb_std,
        "train_manifest": group.train_manifest,
        "test_manifest": group.test_manifest,
        "load_report": load_report,
        "num_train_samples": int(train_features.shape[0]),
        "num_test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "outputs": outputs,
    }
    save_json(out_dir / "umap_meta.json", meta)
    return meta


def parse_group_keys(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return [g.key for g in GROUPS]
    requested = [v.strip().lower() for v in values]
    valid = {g.key for g in GROUPS}
    bad = [v for v in requested if v not in valid]
    if bad:
        raise ValueError(f"Unknown group(s): {bad}. Valid: {sorted(valid)}")
    return requested


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RGB last.pth UMAP visualisation by true label.")
    p.add_argument("--dataset_root", type=Path, default=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset"))
    p.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "analysis" / "umap_rgb_last")
    p.add_argument("--groups", nargs="*", default=None, help="Subset of groups: excl take")
    p.add_argument("--tier_mode", default="tier1", choices=["tier1", "tier2", "tier3"])
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--model_depth", type=int, default=18)
    p.add_argument("--rgb_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--feature_l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="euclidean")
    p.add_argument("--plot_point_size", type=float, default=8.0)
    p.add_argument("--plot_alpha", type=float, default=0.82)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir = args.output_dir.resolve()
    ensure_dir(args.output_dir)
    seed_everything(args.seed)
    save_json(args.output_dir / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "project_root": str(PROJECT_ROOT),
    })

    selected = set(parse_group_keys(args.groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[output] {args.output_dir}")

    summary: List[Dict[str, Any]] = []
    t0 = time.time()
    for group in GROUPS:
        if group.key not in selected:
            continue
        print(f"\n[group] {group.key}")
        reverse_label_map = build_reverse_label_map(group.label_map_json, args.tier_mode)
        loaders = {
            "train": build_loader(group, group.train_manifest, args),
            "test": build_loader(group, group.test_manifest, args),
        }
        weights = find_last_checkpoints(group)
        if not weights:
            raise FileNotFoundError(f"No last.pth files found under {group.result_dir / 'weights'}")
        print(f"[weights] found {len(weights)} last.pth files")
        for weight_path in weights:
            meta = analyze_weight(group, weight_path, loaders, reverse_label_map, args, device)
            summary.append({
                "group": meta["group"],
                "weight_slug": meta["weight_slug"],
                "weight_path": meta["weight_path"],
                "num_train_samples": meta["num_train_samples"],
                "num_test_samples": meta["num_test_samples"],
                "feature_dim": meta["feature_dim"],
                "train_png": meta["outputs"]["train"]["png"],
                "test_png": meta["outputs"]["test"]["png"],
                "combined_png": meta["outputs"]["combined"]["png"],
            })
            save_csv(args.output_dir / "summary_partial.csv", summary)

    save_csv(args.output_dir / "summary.csv", summary)
    save_json(args.output_dir / "summary.json", {"rows": summary, "elapsed_seconds": time.time() - t0})
    print(f"\n[done] elapsed seconds: {time.time() - t0:.1f}")
    print(f"[summary] {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
