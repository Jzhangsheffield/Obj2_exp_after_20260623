#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D and 3D UMAP for RGB contrastive pretraining checkpoint_0200.pth files.

For each checkpoint, this script extracts two representations:
  - feat512: pooled backbone feature from encoder_q.forward_features()
  - proj128: projection-head output from encoder_q.forward_head(feat512)

Both representations are visualised with true-label colouring for train, test,
and train+test combined splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import umap_rgb_lastpth_by_label as base
import umap_rgb_lastpth_by_label_3d as base3d


@dataclass(frozen=True)
class PretrainGroup:
    key: str
    result_dir: Path
    label_map_json: Path
    train_manifest: str
    test_manifest: str
    num_classes: int
    rgb_mean: Tuple[float, float, float]
    rgb_std: Tuple[float, float, float]


GROUPS: Tuple[PretrainGroup, ...] = (
    PretrainGroup(
        key="excl",
        result_dir=PROJECT_ROOT / "results" / "rgb_N_except_take_put_adamw_22",
        label_map_json=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset\label_map_except_take_put.json"),
        train_manifest="N_as_test/train_manifest_except_take_put.jsonl",
        test_manifest="N_as_test/test_manifest_except_take_put.jsonl",
        num_classes=15,
        rgb_mean=(0.3752, 0.3864, 0.3960),
        rgb_std=(0.2934, 0.2724, 0.2644),
    ),
    PretrainGroup(
        key="take",
        result_dir=PROJECT_ROOT / "results" / "rgb_N_take_put_adamw_22",
        label_map_json=Path(r"C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset\label_map_take_put.json"),
        train_manifest="N_as_test/train_manifest_take_put.jsonl",
        test_manifest="N_as_test/test_manifest_take_put.jsonl",
        num_classes=2,
        rgb_mean=(0.3725, 0.3828, 0.3921),
        rgb_std=(0.2923, 0.2715, 0.2640),
    ),
)


def make_cfg(group: PretrainGroup, args: argparse.Namespace):
    adapted = base.GroupConfig(
        key=group.key,
        result_dir=group.result_dir,
        label_map_json=group.label_map_json,
        train_manifest=group.train_manifest,
        test_manifest=group.test_manifest,
        num_classes=group.num_classes,
        rgb_mean=group.rgb_mean,
        rgb_std=group.rgb_std,
    )
    return base.make_cfg(adapted, args)


def build_loader(group: PretrainGroup, manifest: str, args: argparse.Namespace):
    label_map = base.load_label_map_json(str(group.label_map_json))
    dataset = base.build_packed_mapstyle_dataset(
        dataset_root=str(args.dataset_root),
        manifest_name=manifest,
        cfg=make_cfg(group, args),
        label_map=label_map,
        verify_paths_on_init=True,
    )
    return base.build_packed_mapstyle_loader_from_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
        sampler=None,
        pin_memory=args.pin_memory,
    )


def make_pretrain_encoder(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = base.resnet3d.generate_model(
        args.model_depth,
        n_input_channels=3,
        num_classes=args.proj_dim,
    )
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp),
        nn.ReLU(),
        nn.Linear(dim_mlp, args.proj_dim),
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_encoder_q_checkpoint(model: nn.Module, ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    raw_state = base.extract_state_dict_from_checkpoint(ckpt)
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped_non_q = 0
    dropped_missing: List[str] = []
    dropped_shape: List[str] = []

    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module.encoder_q."):
            new_key = key[len("module.encoder_q."):]
        elif key.startswith("encoder_q."):
            new_key = key[len("encoder_q."):]
        else:
            skipped_non_q += 1
            continue
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
        "loaded_encoder_q_keys": len(filtered),
        "raw_tensor_keys": sum(1 for v in raw_state.values() if torch.is_tensor(v)),
        "skipped_non_encoder_q_count": skipped_non_q,
        "dropped_missing_count": len(dropped_missing),
        "dropped_shape_count": len(dropped_shape),
        "missing_after_load_count": len(msg.missing_keys),
        "unexpected_after_load_count": len(msg.unexpected_keys),
        "epoch": ckpt.get("epoch", None) if isinstance(ckpt, dict) else None,
        "ablation_mode": ckpt.get("ablation_mode", None) if isinstance(ckpt, dict) else None,
        "contrastive_loss_mode": ckpt.get("contrastive_loss_mode", None) if isinstance(ckpt, dict) else None,
    }


@torch.no_grad()
def extract_representations(
    model: nn.Module,
    loader,
    device: torch.device,
    tier_mode: str,
    split: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    model.eval()
    feat512_all: List[torch.Tensor] = []
    proj128_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    ids_all: List[str] = []

    for batch in tqdm.tqdm(loader, desc=f"extract:{split}", dynamic_ncols=True):
        inputs, labels, sample_ids = base.extract_inputs_labels_ids(batch, tier_mode)
        inputs = inputs.to(device, non_blocking=True).float()
        if inputs.ndim != 5:
            raise ValueError(f"Expected RGB tensor [B,T,C,H,W], got {tuple(inputs.shape)}")
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        feat512 = model.forward_features(inputs)
        proj128 = model.forward_head(feat512)
        feat512_all.append(feat512.detach().cpu().float())
        proj128_all.append(proj128.detach().cpu().float())
        labels_all.append(labels.detach().cpu().long().view(-1))
        ids_all.extend([str(x) for x in sample_ids])

    reps = {
        "feat512": torch.cat(feat512_all, dim=0).numpy().astype(np.float32),
        "proj128": torch.cat(proj128_all, dim=0).numpy().astype(np.float32),
    }
    labels_np = torch.cat(labels_all, dim=0).numpy().astype(np.int64)
    sample_ids_np = np.array(ids_all, dtype=str)
    return reps, labels_np, sample_ids_np


def find_checkpoints(group: PretrainGroup) -> List[Path]:
    return sorted(group.result_dir.rglob("checkpoint_0200.pth"))


def checkpoint_slug(path: Path, group: PretrainGroup) -> str:
    rel_parent = path.parent.relative_to(group.result_dir)
    raw = "_".join(rel_parent.parts)
    digest = hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:6]
    return f"{base.sanitize_name(raw, 42)}_{digest}"


def visualize_2d(
    out_dir: Path,
    group_key: str,
    slug: str,
    rep_name: str,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    label_names: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, str]:
    x = base.l2_normalize(features) if args.feature_l2_normalize else features.astype(np.float32, copy=False)
    x = base.maybe_pca(x, args.pca_dim, args.seed)
    coords = base.compute_umap(x, args)
    coords_csv = out_dir / f"{split_name}_umap2d_coords.csv"
    png = out_dir / f"{split_name}_umap2d_by_label_legend.png"
    splits = np.array([split_name] * len(labels), dtype=str)
    if split_name == "combined":
        splits = np.array(["train"] * args._combined_train_n + ["test"] * (len(labels) - args._combined_train_n), dtype=str)
    base.save_coords_csv(coords_csv, coords, sample_ids, splits, labels, label_names)
    base.plot_by_label(
        coords,
        labels,
        label_names,
        title=f"{group_key} | {slug} | {rep_name} | {split_name} | 2D UMAP",
        out_path=png,
        point_size=args.plot_point_size,
        alpha=args.plot_alpha,
    )
    return {"coords_csv": str(coords_csv), "png": str(png)}


def visualize_3d(
    out_dir: Path,
    group_key: str,
    slug: str,
    rep_name: str,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    label_names: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, str]:
    x = base.l2_normalize(features) if args.feature_l2_normalize else features.astype(np.float32, copy=False)
    x = base.maybe_pca(x, args.pca_dim, args.seed)
    coords = base3d.compute_umap3d(x, args)
    coords_csv = out_dir / f"{split_name}_umap3d_coords.csv"
    html_path = out_dir / f"{split_name}_umap3d_by_label_interactive.html"
    splits = np.array([split_name] * len(labels), dtype=str)
    if split_name == "combined":
        splits = np.array(["train"] * args._combined_train_n + ["test"] * (len(labels) - args._combined_train_n), dtype=str)
    base3d.save_coords_csv_3d(coords_csv, coords, sample_ids, splits, labels, label_names)
    base3d.write_interactive_html(
        html_path,
        title=f"{group_key} | {slug} | {rep_name} | {split_name} | 3D UMAP",
        coords=coords,
        sample_ids=sample_ids,
        splits=splits,
        labels=labels,
        label_names=label_names,
    )
    return {"coords_csv": str(coords_csv), "html": str(html_path)}


def run_visualizations_for_rep(
    root_out: Path,
    group: PretrainGroup,
    slug: str,
    rep_name: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    train_names: np.ndarray,
    test_names: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    for mode in ("2d", "3d"):
        mode_out = root_out / f"{mode}_{rep_name}" / group.key / slug
        base.ensure_dir(mode_out)
        vis_fn = visualize_2d if mode == "2d" else visualize_3d
        outputs[f"{mode}_train"] = vis_fn(mode_out, group.key, slug, rep_name, "train", train_features, train_labels, train_ids, train_names, args)
        outputs[f"{mode}_test"] = vis_fn(mode_out, group.key, slug, rep_name, "test", test_features, test_labels, test_ids, test_names, args)
        args._combined_train_n = int(train_features.shape[0])
        combined_features = np.concatenate([train_features, test_features], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)
        combined_ids = np.concatenate([train_ids, test_ids], axis=0)
        combined_names = np.concatenate([train_names, test_names], axis=0)
        outputs[f"{mode}_combined"] = vis_fn(mode_out, group.key, slug, rep_name, "combined", combined_features, combined_labels, combined_ids, combined_names, args)
    return outputs


def analyze_checkpoint(
    group: PretrainGroup,
    ckpt_path: Path,
    loaders: Dict[str, Any],
    reverse_label_map: Dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    slug = checkpoint_slug(ckpt_path, group)
    print(f"\n[checkpoint] {group.key} | {slug}")
    print(f"[path] {ckpt_path}")
    model = make_pretrain_encoder(args, device)
    load_report = load_encoder_q_checkpoint(model, ckpt_path)
    print(json.dumps(load_report, indent=2))

    train_reps, train_labels, train_ids = extract_representations(model, loaders["train"], device, args.tier_mode, "train")
    test_reps, test_labels, test_ids = extract_representations(model, loaders["test"], device, args.tier_mode, "test")
    train_names = base.label_names_from_ids(train_labels, reverse_label_map)
    test_names = base.label_names_from_ids(test_labels, reverse_label_map)

    all_outputs: Dict[str, Any] = {}
    for rep_name in ("feat512", "proj128"):
        all_outputs[rep_name] = run_visualizations_for_rep(
            args.output_dir,
            group,
            slug,
            rep_name,
            train_reps[rep_name],
            test_reps[rep_name],
            train_labels,
            test_labels,
            train_ids,
            test_ids,
            train_names,
            test_names,
            args,
        )
        meta_dir = args.output_dir / "meta" / group.key / slug
        base.ensure_dir(meta_dir)
        base.save_json(meta_dir / f"{rep_name}_meta.json", {
            "group": group.key,
            "checkpoint": str(ckpt_path),
            "checkpoint_slug": slug,
            "representation": rep_name,
            "load_report": load_report,
            "num_train_samples": int(train_labels.shape[0]),
            "num_test_samples": int(test_labels.shape[0]),
            "feature_dim": int(train_reps[rep_name].shape[1]),
            "outputs": all_outputs[rep_name],
        })

    return {
        "group": group.key,
        "checkpoint_slug": slug,
        "checkpoint": str(ckpt_path),
        "load_report": load_report,
        "num_train_samples": int(train_labels.shape[0]),
        "num_test_samples": int(test_labels.shape[0]),
        "feat512_dim": int(train_reps["feat512"].shape[1]),
        "proj128_dim": int(train_reps["proj128"].shape[1]),
        "outputs": all_outputs,
    }


def parse_group_keys(values):
    if not values:
        return [g.key for g in GROUPS]
    requested = [v.strip().lower() for v in values]
    valid = {g.key for g in GROUPS}
    bad = [v for v in requested if v not in valid]
    if bad:
        raise ValueError(f"Unknown group(s): {bad}. Valid: {sorted(valid)}")
    return requested


def build_argparser() -> argparse.ArgumentParser:
    p = base.build_argparser()
    p.description = "2D/3D UMAP for RGB contrastive pretraining checkpoint_0200.pth files."
    p.set_defaults(output_dir=PROJECT_ROOT / "analysis" / "umap_rgb_pretrain")
    p.add_argument("--proj_dim", type=int, default=128)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir = args.output_dir.resolve()
    base.ensure_dir(args.output_dir)
    base.seed_everything(args.seed)
    base.save_json(args.output_dir / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "project_root": str(PROJECT_ROOT),
        "checkpoint_name": "checkpoint_0200.pth",
        "representations": ["feat512", "proj128"],
        "umap_outputs": ["2d", "3d"],
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
        reverse_label_map = base.build_reverse_label_map(group.label_map_json, args.tier_mode)
        loaders = {
            "train": build_loader(group, group.train_manifest, args),
            "test": build_loader(group, group.test_manifest, args),
        }
        checkpoints = find_checkpoints(group)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint_0200.pth found under {group.result_dir}")
        print(f"[checkpoints] found {len(checkpoints)}")
        for ckpt_path in checkpoints:
            meta = analyze_checkpoint(group, ckpt_path, loaders, reverse_label_map, args, device)
            summary.append({
                "group": meta["group"],
                "checkpoint_slug": meta["checkpoint_slug"],
                "checkpoint": meta["checkpoint"],
                "num_train_samples": meta["num_train_samples"],
                "num_test_samples": meta["num_test_samples"],
                "feat512_dim": meta["feat512_dim"],
                "proj128_dim": meta["proj128_dim"],
                "epoch": meta["load_report"].get("epoch", ""),
                "ablation_mode": meta["load_report"].get("ablation_mode", ""),
                "contrastive_loss_mode": meta["load_report"].get("contrastive_loss_mode", ""),
            })
            base.save_csv(args.output_dir / "summary_partial.csv", summary)

    base.save_csv(args.output_dir / "summary.csv", summary)
    base.save_json(args.output_dir / "summary.json", {"rows": summary, "elapsed_seconds": time.time() - t0})
    print(f"\n[done] elapsed seconds: {time.time() - t0:.1f}")
    print(f"[summary] {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
