#!/usr/bin/env python3
"""Batch-test all completed round-2 fine-tuned RGB checkpoints."""

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
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task = cfg["task"]
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    data = Path(args.dataset_root or os.environ.get("DATASET_ROOT", cfg["dataset_root"]))
    output_root = root / cfg["finetune_output_rel"]
    weights_root = output_root / "weights"
    selected = sorted(weights_root.rglob("best_val.pth")) if weights_root.is_dir() else []
    if args.include_last:
        selected += sorted(weights_root.rglob("last.pth"))
    selected = sorted(set(selected))
    if args.validate_command and not selected:
        selected = [weights_root / "round2_parse_only_dummy.pth"]
    if not selected and not args.dry_run:
        raise FileNotFoundError(f"No selected weights under {weights_root}")

    batch_root = weights_root / "_batch_test" / "round2_fixed"
    summary_root = batch_root / "summary"
    summary_csv = summary_root / "rgb_test_results.csv"
    if not args.dry_run and not args.validate_command:
        if summary_csv.is_file():
            summary_csv.unlink()
        summary_root.mkdir(parents=True, exist_ok=True)

    original = root / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    if args.validate_command:
        wrapper = root / "codex_script" / "rgb_round2_20260717" / "rgb_round2_finetune_entry.py"
        cmd = [
            args.python_bin, "-u", str(wrapper),
            "--round2-original-script", str(original),
            "--round2-parse-only",
        ]
    else:
        cmd = [args.python_bin, "-u", str(original)]
    cmd.extend([
        "--run_mode", "test",
        "--dataset_root", str(data),
        "--label_map_json", str(data / task["label_map_rel"]),
        "--test_manifest", str(data / task["test_manifest_rel"]),
        "--save_path", str(summary_root / "rgb"),
        "--test_results_csv", str(summary_csv),
        "--tier_mode", task["tier_mode"],
        "--num_classes", str(task["num_classes"]),
        "--use_modality", "rgb",
        "--model_depth", "18",
        "--n_frames", str(task["n_frames"]),
        "--rgb_camera_id", task["rgb_camera_id"],
        "--rgb_size", "224",
        "--rgb_mean", *map(str, task["rgb_mean"]),
        "--rgb_std", *map(str, task["rgb_std"]),
        "--no-rgb_apply_spatial_aug",
        "--rgb_hflip_p", "0.0",
        "--rgb_vflip_p", "0.0",
        "--rgb_jitter_p", "0.0",
        "--rgb_gray_p", "0.0",
        "--rgb_blur_p", "0.0",
        "--batch_size", "64",
        "--num_workers_test", "8",
        "--prefetch_factor_test", "2",
        "--seed", "1",
        "--test_weight_paths", *map(str, selected),
    ])
    print(f"Selected weights: {len(selected)}")
    print("[Command]", shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=root, check=True)
    if args.validate_command:
        print("Round-2 test command validation: OK")
        return
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
