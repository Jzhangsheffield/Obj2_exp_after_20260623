#!/usr/bin/env python3
"""Run one full-fine-tuning task from the round-2 shortlist."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "round2_config.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    matches = [x for x in cfg["finetune_plan"] if int(x["index"]) == args.index]
    if len(matches) != 1:
        raise ValueError(f"Unknown fine-tune plan index: {args.index}")
    plan = matches[0]
    task = cfg["task"]
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    data = Path(args.dataset_root or os.environ.get("DATASET_ROOT", cfg["dataset_root"]))
    pretrain_root = root / cfg["pretrain_output_rel"]
    output_root = root / cfg["finetune_output_rel"]
    save_root = output_root / "weights" / plan["id"]
    datamap_root = output_root / "datamaps" / plan["id"]
    original = root / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    wrapper = root / "codex_script" / "rgb_round2_20260717" / "rgb_round2_finetune_entry.py"

    existing_last = list(save_root.rglob("last.pth")) if save_root.is_dir() else []
    if existing_last:
        print(f"[Skip] completed fine-tune exists: {existing_last[0]}")
        return

    pretrained = None
    if plan["pretrain_id"] is not None:
        pretrained = pretrain_root / plan["pretrain_id"] / "checkpoint_0200.pth"
        if not pretrained.is_file() and not args.dry_run and not args.validate_command:
            print(f"[Skip] prerequisite pretrain checkpoint is missing: {pretrained}")
            return

    cmd = [
        args.python_bin, "-u", str(wrapper),
        "--round2-original-script", str(original),
        "--round2-auto-resume",
        "--run_mode", "train",
        "--save_path", str(save_root),
        "--datamap_csv_path", str(datamap_root),
        "--dataset_root", str(data),
        "--label_map_json", str(data / task["label_map_rel"]),
        "--train_manifest", str(data / task["train_manifest_rel"]),
        "--val_manifest", str(data / task["val_manifest_rel"]),
        "--tier_mode", task["tier_mode"],
        "--n_frames", str(task["n_frames"]),
        "--use_modality", "rgb",
        "--num_classes", str(task["num_classes"]),
        "--model_depth", "18",
        "--rgb_camera_id", task["rgb_camera_id"],
        "--rgb_size", "224",
        "--rgb_mean", *map(str, task["rgb_mean"]),
        "--rgb_std", *map(str, task["rgb_std"]),
        "--rrc_scale_min", "0.85",
        "--rrc_scale_max", "1.0",
        "--rrc_ratio_min", "0.9",
        "--rrc_ratio_max", "1.1",
        "--rgb_hflip_p", "0.5",
        "--rgb_vflip_p", "0.0",
        "--rgb_jitter_p", "0.0",
        "--rgb_gray_p", "0.0",
        "--rgb_blur_p", "0.0",
        "--rgb_apply_spatial_aug",
        "--epochs", "100",
        "--batch_size", "64",
        "--num_workers_train", "8",
        "--num_workers_val", "6",
        "--prefetch_factor_train", "2",
        "--prefetch_factor_val", "2",
        "--optimizer", "adamw",
        "--learning_rate", "0.001",
        "--weight_decay", "0.0001",
        "--schedules", "50", "75",
        "--seed", "1",
        "--finetune_mode", "full",
        "--save_period", "10",
        "--best_after_epoch", "0",
    ]
    # A randomly initialized backbone needs the standard full-network LR;
    # reducing it to the pretrained-backbone LR would make scratch unfairly weak.
    if pretrained is not None:
        cmd.extend([
            "--use_discriminative_lr",
            "--backbone_learning_rate", str(plan["backbone_lr"]),
            "--head_learning_rate", "0.001",
        ])
    if args.validate_command:
        cmd.insert(5, "--round2-parse-only")
    if pretrained is not None:
        cmd.extend(["--pretrained_weight_paths", str(pretrained)])

    print(json.dumps({"plan": plan, "pretrained": str(pretrained) if pretrained else None}, indent=2))
    print("[Command]", shlex.join(cmd))
    if args.dry_run:
        return
    if not args.validate_command:
        save_root.mkdir(parents=True, exist_ok=True)
        datamap_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
