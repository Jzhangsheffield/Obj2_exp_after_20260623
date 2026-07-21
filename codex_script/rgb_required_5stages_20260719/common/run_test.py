#!/usr/bin/env python3
"""Test one fine-tuned run using validation-balanced checkpoint selection."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from config_utils import choose, load_stage, roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--experiment-id")
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    master, stage, _ = load_stage(config_path)
    plan = choose(stage.get("finetune_experiments", []), args.index, args.experiment_id)
    project, dataset = roots(master, args.project_root, args.dataset_root)
    lock = master["test_lock"]
    if not (args.dry_run or args.validate_command) and os.environ.get(lock["environment_variable"]) != lock["required_value"]:
        raise RuntimeError(
            "Test is locked. Export %s=%s only after all validation decisions are frozen."
            % (lock["environment_variable"], lock["required_value"])
        )
    task = master["task"]
    common = master["test_common"]
    weights = project / stage["finetune_output_rel"] / "weights" / plan["id"]
    candidates = sorted(weights.rglob("best_val_balanced.pth")) if weights.is_dir() else []
    if args.validate_command and not candidates:
        candidates = [weights / "parse_only" / "best_val_balanced.pth"]
    if len(candidates) != 1 and not args.dry_run:
        raise FileNotFoundError("Expected exactly one best_val_balanced.pth under %s; found %d" % (weights, len(candidates)))
    selected = candidates[0] if candidates else weights / "best_val_balanced.pth"
    summary_root = project / stage["finetune_output_rel"] / "test" / plan["id"]
    summary_csv = summary_root / "rgb_test_results.csv"
    original = project / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    entry = project / "codex_script" / "rgb_required_5stages_20260719" / "common" / "finetune_entry.py"
    cmd = [
        args.python_bin, "-u", str(entry),
        "--required-original-script", str(original),
        "--required-backbone-temporal-mode", plan.get("backbone_temporal_mode", "current"),
    ]
    if args.validate_command:
        cmd.append("--required-parse-only")
    values = {
        "--run_mode": "test", "--dataset_root": dataset,
        "--label_map_json": dataset / task["label_map_rel"],
        "--test_manifest": dataset / task["test_manifest_rel"],
        "--save_path": summary_root / "artifacts", "--test_results_csv": summary_csv,
        "--tier_mode": task["tier_mode"], "--num_classes": task["num_classes"],
        "--use_modality": "rgb", "--model_depth": master["finetune_common"]["model_depth"],
        "--n_frames": task["n_frames"], "--rgb_camera_id": task["rgb_camera_id"],
        "--rgb_size": master["finetune_augmentation"]["rgb_size"],
        "--rgb_hflip_p": 0.0, "--rgb_vflip_p": 0.0, "--rgb_jitter_p": 0.0,
        "--rgb_gray_p": 0.0, "--rgb_blur_p": 0.0, "--batch_size": common["batch_size"],
        "--num_workers_test": common["num_workers"], "--prefetch_factor_test": common["prefetch_factor"],
        "--seed": plan.get("seed", 1),
    }
    for flag, value in values.items():
        cmd.extend([flag, str(value)])
    for flag, seq in {"--rgb_mean": task["rgb_mean"], "--rgb_std": task["rgb_std"]}.items():
        cmd.append(flag)
        cmd.extend(map(str, seq))
    cmd.extend(["--no-rgb_apply_spatial_aug", "--test_weight_paths", str(selected)])
    print(json.dumps({"plan": plan, "selected": str(selected), "output": str(summary_csv)}, indent=2))
    print("[Command] %s" % shlex.join(cmd))
    if args.dry_run:
        return
    if not args.validate_command:
        summary_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(project), check=True)


if __name__ == "__main__":
    main()
