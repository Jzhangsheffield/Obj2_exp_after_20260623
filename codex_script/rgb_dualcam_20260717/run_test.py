#!/usr/bin/env python3
"""Test every completed fine-tuned model on both RGB cameras."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "dualcam_config.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--camera", choices=["00143", "00152", "both"], default="both")
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task = cfg["task"]
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    data = Path(args.dataset_root or os.environ.get("DATASET_ROOT", cfg["dataset_root"]))
    weights_root = root / cfg["finetune_output_rel"] / "weights"
    selected = sorted(weights_root.rglob("best_val.pth")) if weights_root.is_dir() else []
    if args.include_last and weights_root.is_dir():
        selected += sorted(weights_root.rglob("last.pth"))
    selected = sorted(set(selected))
    if args.validate_command and not selected:
        selected = [weights_root / "parse_only_dummy" / "best_val.pth"]
    if not selected and not args.dry_run:
        raise FileNotFoundError(f"No completed weights under {weights_root}")

    summary_root = weights_root / "_batch_test" / "dualcam" / "summary"
    summary_csv = summary_root / "rgb_dualcam_test_results.csv"
    if not args.dry_run and not args.validate_command:
        summary_root.mkdir(parents=True, exist_ok=True)
        if summary_csv.is_file():
            summary_csv.unlink()
    cameras = ["00143", "00152"] if args.camera == "both" else [args.camera]
    original = root / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    wrapper = root / "codex_script" / "rgb_dualcam_20260717" / "dualcam_finetune_entry.py"

    for camera_id in cameras:
        cam = cfg["cameras"][camera_id]
        cmd = [
            args.python_bin, "-u", str(wrapper),
            "--dualcam-original-script", str(original),
            "--dualcam-test-tag", f"cam{camera_id}",
        ]
        if args.validate_command:
            cmd.append("--dualcam-parse-only")
        cmd.extend([
            "--run_mode", "test",
            "--dataset_root", str(data),
            "--label_map_json", str(data / task["label_map_rel"]),
            "--test_manifest", str(data / task["test_manifest_rel"]),
            "--save_path", str(summary_root / f"cam{camera_id}"),
            "--test_results_csv", str(summary_csv),
            "--tier_mode", task["tier_mode"], "--num_classes", str(task["num_classes"]),
            "--use_modality", "rgb", "--model_depth", "18", "--n_frames", str(task["n_frames"]),
            "--rgb_camera_id", camera_id, "--rgb_size", "224",
            "--rgb_mean", *map(str, cam["mean"]), "--rgb_std", *map(str, cam["std"]),
            "--no-rgb_apply_spatial_aug",
            "--rgb_hflip_p", "0.0", "--rgb_vflip_p", "0.0",
            "--rgb_jitter_p", "0.0", "--rgb_gray_p", "0.0", "--rgb_blur_p", "0.0",
            "--batch_size", "64", "--num_workers_test", "8", "--prefetch_factor_test", "2",
            "--seed", "1", "--test_weight_paths", *map(str, selected),
        ])
        print(f"Camera {camera_id}; selected weights: {len(selected)}")
        print("[Command]", shlex.join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=root, check=True)
    if args.validate_command:
        print("Dual-camera test commands: OK")
    elif not args.dry_run:
        print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
