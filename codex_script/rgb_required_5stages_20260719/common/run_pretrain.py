#!/usr/bin/env python3
"""Build and launch one configured contrastive-pretraining experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from config_utils import choose, load_stage, roots, write_provenance


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
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    master, stage, _ = load_stage(config_path)
    exp = choose(stage.get("pretrain_experiments", []), args.index, args.experiment_id)
    project, dataset = roots(master, args.project_root, args.dataset_root)
    common = master["pretrain_common"]
    task = master["task"]
    aug = master["pretrain_augmentation"]
    original = project / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    entry = project / "codex_script" / "rgb_required_5stages_20260719" / "common" / "pretrain_entry.py"
    output = project / stage["pretrain_output_rel"] / exp["id"]

    cmd = [
        args.python_bin, "-u", str(entry),
        "--required-original-script", str(original),
        "--required-temporal-mode", exp.get("temporal_mode", "shared"),
        "--required-min-temporal-overlap", str(exp.get("min_temporal_overlap", 1.0)),
        "--required-proto-positive-mode", exp.get("proto_positive_mode", "all"),
        "--required-backbone-temporal-mode", exp.get("backbone_temporal_mode", "current"),
    ]
    if not args.no_auto_resume:
        cmd.append("--required-auto-resume")
    if args.validate_command:
        cmd.append("--required-parse-only")
    values = {
        "--dataset_root": dataset,
        "--train_manifest_name": dataset / task["train_manifest_rel"],
        "--label_map_json": dataset / task["label_map_rel"],
        "--weight_save_path": output,
        "--tier_mode": task["tier_mode"],
        "--n_frames": task["n_frames"],
        "--rgb_camera_id": task["rgb_camera_id"],
        "--rgb_hflip_p": aug["hflip_p"],
        "--rgb_vflip_p": aug["vflip_p"],
        "--rgb_jitter_p": aug["jitter_p"],
        "--rgb_jitter_brightness": aug["jitter_strength"][0],
        "--rgb_jitter_contrast": aug["jitter_strength"][1],
        "--rgb_jitter_saturation": aug["jitter_strength"][2],
        "--rgb_jitter_hue": aug["jitter_strength"][3],
        "--rgb_gray_p": aug["gray_p"],
        "--rgb_blur_p": aug["blur_p"],
        "--rgb_blur_kernel": aug["blur_kernel"],
        "--batch_size": common["batch_size"],
        "--num_workers": common["num_workers"],
        "--model_depth": common["model_depth"],
        "--proj_dim": common["proj_dim"],
        "--K_queue": common["K_queue"],
        "--temperature": common["temperature"],
        "--contrastive_loss": common["contrastive_loss"],
        "--num_positive": common["num_positive"],
        "--ablation_mode": exp["ablation_mode"],
        "--warmup_epochs": exp.get("warmup_epochs", 50),
        "--recluster_interval": exp.get("recluster_interval", common["recluster_interval"]),
        "--default_num_prototypes": exp.get("num_prototypes", 1),
        "--lambda_proto": exp.get("lambda_proto", 0.0),
        "--proto_temperature": exp.get("proto_temperature", common["proto_temperature"]),
        "--proto_refresh_batch_size": common["batch_size"],
        "--proto_refresh_num_workers": common["num_workers"],
        "--lambda_rel": exp.get("lambda_rel", 0.0),
        "--proto_ema_momentum": exp.get("proto_ema_momentum", common["proto_ema_momentum"]),
        "--preview_ema_momentum": exp.get("preview_ema_momentum", common["preview_ema_momentum"]),
        "--rel_same_margin": exp.get("rel_same_margin", common["rel_same_margin"]),
        "--rel_diff_margin": exp.get("rel_diff_margin", common["rel_diff_margin"]),
        "--rel_same_weight": exp.get("rel_same_weight", 1.0),
        "--rel_diff_weight": exp.get("rel_diff_weight", 1.0),
        "--rel_topk_diff_classes": exp.get("rel_topk_diff_classes", 0),
        "--proto_loss_start_epoch": exp.get("proto_start", 50),
        "--rel_loss_start_epoch": exp.get("rel_start", 200),
        "--rel_loss_end_epoch": exp.get("rel_end", common["epochs"]),
        "--rel_lambda_schedule": exp.get("rel_lambda_schedule", "constant"),
        "--epochs": common["epochs"],
        "--learning_rate": common["learning_rate"],
        "--weight_decay": common["weight_decay"],
        "--optimizer": common["optimizer"],
        "--seed": exp.get("seed", 1),
        "--print_freq": common["print_freq"],
        "--save_interval": common["save_interval"],
        "--sampler_type": exp.get("sampler_type", "none"),
    }
    for flag, value in values.items():
        cmd.extend([flag, str(value)])
    for flag, seq in {
        "--rgb_mean": task["rgb_mean"], "--rgb_std": task["rgb_std"],
        "--rgb_out_hw": aug["out_hw"], "--rrc_scale": aug["rrc_scale"],
        "--rrc_ratio": aug["rrc_ratio"], "--rgb_blur_sigma": aug["blur_sigma"],
        "--schedule": common["schedule"],
    }.items():
        cmd.append(flag)
        cmd.extend(map(str, seq))
    cmd.extend([
        "--rgb_apply_spatial_aug", "--mlp", "--no_ddp", "--no-use_syncbn",
        "--verify_paths_on_init", "--proto_refresh_verify_paths_on_init",
        "--enable_loss_stage_schedule", "--debug_mode", "--debug_write_jsonl",
        "--exclude_invalid_queue",
    ])
    payload = {"stage": stage["id"], "experiment": exp, "output": str(output), "command": cmd}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("[Command] %s" % shlex.join(cmd))
    if args.dry_run:
        return
    if not args.validate_command:
        write_provenance(
            output, payload, project,
            [original, entry, entry.parent / "temporal_backbone.py", entry.parent / "proto_loss_modes.py", config_path],
        )
    subprocess.run(cmd, cwd=str(project), check=True, env=dict(os.environ))


if __name__ == "__main__":
    main()
