#!/usr/bin/env python3
"""Build and launch one configured RGB fine-tuning run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from config_utils import choose, load_stage, resolve_pretrained, roots, write_provenance


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
    task = master["task"]
    common = master["finetune_common"]
    aug = master["finetune_augmentation"]
    pretrained = resolve_pretrained(project, master, stage, plan)
    output_root = project / stage["finetune_output_rel"]
    save_root = output_root / "weights" / plan["id"]
    datamap_root = output_root / "datamaps" / plan["id"]
    original = project / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    entry = project / "codex_script" / "rgb_required_5stages_20260719" / "common" / "finetune_entry.py"

    completed = list(save_root.rglob("last.pth")) if save_root.is_dir() else []
    if completed and not args.validate_command:
        print("[Skip] completed fine-tune exists: %s" % completed[0])
        return
    if pretrained is not None and not pretrained.is_file() and not args.dry_run and not args.validate_command:
        raise FileNotFoundError("Required pretrain checkpoint is missing: %s" % pretrained)
    cmd = [
        args.python_bin, "-u", str(entry),
        "--required-original-script", str(original),
        "--required-backbone-temporal-mode", plan.get("backbone_temporal_mode", "current"),
        "--required-auto-resume",
    ]
    if args.validate_command:
        cmd.append("--required-parse-only")
    values = {
        "--run_mode": "train", "--save_path": save_root, "--datamap_csv_path": datamap_root,
        "--dataset_root": dataset, "--label_map_json": dataset / task["label_map_rel"],
        "--train_manifest": dataset / task["train_manifest_rel"],
        "--val_manifest": dataset / task["val_manifest_rel"], "--tier_mode": task["tier_mode"],
        "--n_frames": task["n_frames"], "--use_modality": "rgb", "--num_classes": task["num_classes"],
        "--model_depth": common["model_depth"], "--rgb_camera_id": task["rgb_camera_id"],
        "--rgb_size": aug["rgb_size"], "--rrc_scale_min": aug["rrc_scale"][0],
        "--rrc_scale_max": aug["rrc_scale"][1], "--rrc_ratio_min": aug["rrc_ratio"][0],
        "--rrc_ratio_max": aug["rrc_ratio"][1], "--rgb_hflip_p": aug["hflip_p"],
        "--rgb_vflip_p": aug["vflip_p"], "--rgb_jitter_p": aug["jitter_p"],
        "--rgb_gray_p": aug["gray_p"], "--rgb_blur_p": aug["blur_p"],
        "--epochs": common["epochs"], "--batch_size": common["batch_size"],
        "--num_workers_train": common["num_workers_train"], "--num_workers_val": common["num_workers_val"],
        "--prefetch_factor_train": common["prefetch_factor"], "--prefetch_factor_val": common["prefetch_factor"],
        "--optimizer": common["optimizer"], "--learning_rate": plan.get("head_lr", common["head_lr"]),
        "--weight_decay": common["weight_decay"], "--seed": plan.get("seed", 1),
        "--finetune_mode": plan.get("finetune_mode", "full"), "--save_period": common["save_period"],
        "--best_after_epoch": common["best_after_epoch"],
    }
    for flag, value in values.items():
        cmd.extend([flag, str(value)])
    for flag, seq in {"--rgb_mean": task["rgb_mean"], "--rgb_std": task["rgb_std"], "--schedules": common["schedule"]}.items():
        cmd.append(flag)
        cmd.extend(map(str, seq))
    cmd.append("--rgb_apply_spatial_aug")
    if pretrained is not None:
        cmd.extend(["--pretrained_weight_paths", str(pretrained)])
    if pretrained is not None and plan.get("finetune_mode", "full") == "full":
        cmd.extend(["--use_discriminative_lr", "--backbone_learning_rate", str(plan["backbone_lr"]),
                    "--head_learning_rate", str(plan.get("head_lr", common["head_lr"]))])
    payload = {"stage": stage["id"], "plan": plan, "pretrained": str(pretrained) if pretrained else None,
               "output": str(save_root), "command": cmd}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("[Command] %s" % shlex.join(cmd))
    if args.dry_run:
        return
    if not args.validate_command:
        write_provenance(
            save_root, payload, project,
            [original, entry, entry.parent / "temporal_backbone.py", config_path],
        )
        datamap_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(project), check=True, env=dict(os.environ))


if __name__ == "__main__":
    main()
