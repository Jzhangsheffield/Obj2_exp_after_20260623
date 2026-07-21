#!/usr/bin/env python3
"""Build and launch one configured RGB round-2 pretraining experiment."""

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


def add(cmd: list[str], flag: str, value) -> None:
    cmd.extend([flag, str(value)])


def choose(config: dict, index: int | None, experiment_id: str | None) -> dict:
    experiments = config["experiments"]
    if experiment_id is not None:
        matches = [x for x in experiments if x["id"] == experiment_id]
    else:
        matches = [x for x in experiments if int(x["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely select experiment index={index}, id={experiment_id}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--experiment-id")
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    exp = choose(config, args.index, args.experiment_id)
    common = config["pretrain_common"]
    task = config["task"]
    project_root = Path(args.project_root or os.environ.get("PROJECT_ROOT", config["project_root"]))
    dataset_root = Path(args.dataset_root or os.environ.get("DATASET_ROOT", config["dataset_root"]))
    original = project_root / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    wrapper = project_root / "codex_script" / "rgb_round2_20260717" / "rgb_round2_pretrain_entry.py"
    output = project_root / config["pretrain_output_rel"] / exp["id"]

    cmd = [
        args.python_bin, "-u", str(wrapper),
        "--round2-original-script", str(original),
        "--round2-temporal-mode", exp["temporal_mode"],
        "--round2-min-temporal-overlap", str(exp["min_temporal_overlap"]),
        "--round2-aux-ce-weight", str(exp["aux_ce_weight"]),
    ]
    if not args.no_auto_resume:
        cmd.append("--round2-auto-resume")
    if args.validate_command:
        cmd.append("--round2-parse-only")
    cmd.extend([
        "--dataset_root", str(dataset_root),
        "--train_manifest_name", str(dataset_root / task["train_manifest_rel"]),
        "--label_map_json", str(dataset_root / task["label_map_rel"]),
        "--weight_save_path", str(output),
        "--tier_mode", task["tier_mode"],
        "--n_frames", str(task["n_frames"]),
        "--rgb_camera_id", task["rgb_camera_id"],
        "--rgb_mean", *map(str, task["rgb_mean"]),
        "--rgb_std", *map(str, task["rgb_std"]),
        "--rgb_out_hw", "224", "224",
        "--rrc_scale", *map(str, exp["rrc_scale"]),
        "--rrc_ratio", *map(str, exp["rrc_ratio"]),
        "--rgb_hflip_p", str(exp["hflip_p"]),
        "--rgb_vflip_p", str(exp["vflip_p"]),
        "--rgb_jitter_p", str(exp["jitter_p"]),
        "--rgb_jitter_brightness", str(exp["jitter_strength"][0]),
        "--rgb_jitter_contrast", str(exp["jitter_strength"][1]),
        "--rgb_jitter_saturation", str(exp["jitter_strength"][2]),
        "--rgb_jitter_hue", str(exp["jitter_strength"][3]),
        "--rgb_gray_p", str(exp["gray_p"]),
        "--rgb_blur_p", str(exp["blur_p"]),
        "--rgb_blur_kernel", "5",
        "--rgb_blur_sigma", "0.1", "1.0",
        "--batch_size", str(common["batch_size"]),
        "--num_workers", str(common["num_workers"]),
        "--model_depth", str(common["model_depth"]),
        "--proj_dim", str(common["proj_dim"]),
        "--K_queue", str(common["K_queue"]),
        "--temperature", str(common["temperature"]),
        "--contrastive_loss", common["contrastive_loss"],
        "--num_positive", str(common["num_positive"]),
        "--ablation_mode", exp["ablation_mode"],
        "--warmup_epochs", str(exp["warmup_epochs"]),
        "--recluster_interval", str(common["recluster_interval"]),
        "--default_num_prototypes", str(common["default_num_prototypes"]),
        "--lambda_proto", "1.0",
        "--proto_temperature", str(common["proto_temperature"]),
        "--proto_refresh_batch_size", str(common["batch_size"]),
        "--proto_refresh_num_workers", str(common["num_workers"]),
        "--lambda_rel", str(exp["lambda_rel"]),
        "--proto_ema_momentum", str(common["proto_ema_momentum"]),
        "--preview_ema_momentum", str(common["preview_ema_momentum"]),
        "--rel_same_margin", str(common["rel_same_margin"]),
        "--rel_diff_margin", str(common["rel_diff_margin"]),
        "--rel_topk_diff_classes", str(exp["rel_topk_diff_classes"]),
        "--proto_loss_start_epoch", str(exp["proto_loss_start_epoch"]),
        "--rel_loss_start_epoch", str(exp["rel_loss_start_epoch"]),
        "--rel_loss_end_epoch", str(common["rel_loss_end_epoch"]),
        "--rel_lambda_schedule", "cosine",
        "--epochs", str(common["epochs"]),
        "--learning_rate", str(common["learning_rate"]),
        "--schedule", *map(str, common["schedule"]),
        "--weight_decay", str(common["weight_decay"]),
        "--optimizer", common["optimizer"],
        "--seed", str(common["seed"]),
        "--print_freq", "10",
        "--save_interval", str(common["save_interval"]),
        "--sampler_type", "none",
        "--rgb_apply_spatial_aug",
        "--mlp",
        "--no_ddp",
        "--no-use_syncbn",
        "--verify_paths_on_init",
        "--proto_refresh_verify_paths_on_init",
        "--enable_loss_stage_schedule",
    ])

    print(json.dumps({"experiment": exp, "output": str(output)}, indent=2))
    print("[Command]", shlex.join(cmd))
    if args.dry_run:
        return
    if not args.validate_command:
        output.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
