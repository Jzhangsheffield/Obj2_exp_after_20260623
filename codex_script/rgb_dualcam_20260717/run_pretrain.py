#!/usr/bin/env python3
"""Build and launch one configured dual-camera RGB pretraining experiment."""

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


def select(items, index, experiment_id):
    matches = [x for x in items if x["id"] == experiment_id] if experiment_id else [x for x in items if int(x["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely select index={index}, id={experiment_id}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--experiment-id")
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-command", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    exp = select(cfg["experiments"], args.index, args.experiment_id)
    common, aug, task = cfg["pretrain_common"], cfg["augmentation"], cfg["task"]
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    data = Path(args.dataset_root or os.environ.get("DATASET_ROOT", cfg["dataset_root"]))
    original = root / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    wrapper = root / "codex_script" / "rgb_dualcam_20260717" / "dualcam_pretrain_entry.py"
    output = root / cfg["pretrain_output_rel"] / exp["id"]
    cam_a, cam_b = cfg["cameras"]["00143"], cfg["cameras"]["00152"]

    cmd = [
        args.python_bin, "-u", str(wrapper),
        "--dualcam-original-script", str(original),
        "--dualcam-view-mode", exp["view_mode"],
        "--dualcam-camera-a", "00143",
        "--dualcam-camera-b", "00152",
        "--dualcam-mean-b", *map(str, cam_b["mean"]),
        "--dualcam-std-b", *map(str, cam_b["std"]),
        "--dualcam-temporal-span-min", str(aug["temporal_span_min"]),
        "--dualcam-hybrid-cross-probability", str(exp["hybrid_cross_probability"]),
        "--dualcam-aux-ce-weight", str(exp["aux_ce_weight"]),
    ]
    if not args.no_auto_resume:
        cmd.append("--dualcam-auto-resume")
    if args.validate_command:
        cmd.append("--dualcam-parse-only")
    cmd.extend([
        "--dataset_root", str(data),
        "--train_manifest_name", str(data / task["train_manifest_rel"]),
        "--label_map_json", str(data / task["label_map_rel"]),
        "--weight_save_path", str(output),
        "--tier_mode", task["tier_mode"],
        "--n_frames", str(task["n_frames"]),
        "--rgb_camera_id", "00143",
        "--rgb_mean", *map(str, cam_a["mean"]),
        "--rgb_std", *map(str, cam_a["std"]),
        "--rgb_out_hw", "224", "224",
        "--rrc_scale", *map(str, aug["rrc_scale"]),
        "--rrc_ratio", *map(str, aug["rrc_ratio"]),
        "--rgb_hflip_p", str(aug["hflip_p"]),
        "--rgb_vflip_p", str(aug["vflip_p"]),
        "--rgb_jitter_p", str(aug["jitter_p"]),
        "--rgb_jitter_brightness", str(aug["jitter_strength"][0]),
        "--rgb_jitter_contrast", str(aug["jitter_strength"][1]),
        "--rgb_jitter_saturation", str(aug["jitter_strength"][2]),
        "--rgb_jitter_hue", str(aug["jitter_strength"][3]),
        "--rgb_gray_p", str(aug["gray_p"]),
        "--rgb_blur_p", str(aug["blur_p"]),
        "--rgb_blur_kernel", "5", "--rgb_blur_sigma", "0.1", "1.0",
        "--batch_size", str(common["batch_size"]),
        "--num_workers", str(common["num_workers"]),
        "--model_depth", str(common["model_depth"]),
        "--proj_dim", str(common["proj_dim"]),
        "--K_queue", str(common["K_queue"]),
        "--temperature", str(common["temperature"]),
        "--contrastive_loss", common["contrastive_loss"],
        "--num_positive", str(common["num_positive"]),
        "--ablation_mode", exp["ablation_mode"],
        "--warmup_epochs", "50",
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
        "--proto_loss_start_epoch", "50",
        "--rel_loss_start_epoch", "50",
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
        "--rgb_apply_spatial_aug", "--mlp", "--no_ddp", "--no-use_syncbn",
        "--verify_paths_on_init", "--proto_refresh_verify_paths_on_init",
        "--enable_loss_stage_schedule",
    ])
    print(json.dumps({"experiment": exp, "output": str(output)}, indent=2))
    print("[Command]", shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
