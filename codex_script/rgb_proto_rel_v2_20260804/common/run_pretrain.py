#!/usr/bin/env python3
"""Build and launch one V2 pretraining experiment."""

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
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--index", type=int); sel.add_argument("--experiment-id")
    p.add_argument("--project-root"); p.add_argument("--dataset-root")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--validate-command", action="store_true")
    args = p.parse_args()
    config_path = Path(args.config).resolve(); master, stage, _ = load_stage(config_path)
    exp = choose(stage["pretrain_experiments"], args.index, args.experiment_id)
    project, dataset = roots(master, args.project_root, args.dataset_root)
    task, aug, common = master["task"], master["pretrain_augmentation"], master["pretrain_common"]
    original = project / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    entry = project / "codex_script" / "rgb_proto_rel_v2_20260804" / "common" / "pretrain_v2_entry.py"
    output = project / stage["pretrain_output_rel"] / exp["id"]
    v2 = dict(master["v2_defaults"]); v2.update(exp.get("v2", {}))
    cmd = [args.python_bin, "-u", str(entry), "--v2-original-script", str(original),
           "--v2-auto-resume",
           "--v2-temporal-mode", exp.get("temporal_mode", "shared"), "--v2-min-temporal-overlap", str(exp.get("min_temporal_overlap", 1.0))]
    for key, flag in {
        "assignment_mode": "--v2-assignment-mode", "assignment_temperature": "--v2-assignment-temperature",
        "prediction_temperature": "--v2-prediction-temperature", "sinkhorn_iterations": "--v2-sinkhorn-iterations",
        "balance_weight": "--v2-balance-weight", "diversity_weight": "--v2-diversity-weight",
        "diversity_margin": "--v2-diversity-margin", "preview_momentum": "--v2-preview-momentum",
        "bank_momentum": "--v2-bank-momentum", "rel_mode": "--v2-rel-mode", "rel_topk": "--v2-rel-topk",
        "rel_margin": "--v2-rel-margin", "rel_temperature": "--v2-rel-temperature",
        "direction_weight": "--v2-direction-weight", "direction_delta": "--v2-direction-delta",
        "diagnostic_interval": "--v2-diagnostic-interval",
    }.items(): cmd.extend([flag, str(v2[key])])
    if args.validate_command: cmd.append("--v2-parse-only")
    values = {
        "--dataset_root": dataset, "--train_manifest_name": dataset / task["train_manifest_rel"],
        "--label_map_json": dataset / task["label_map_rel"], "--weight_save_path": output,
        "--tier_mode": task["tier_mode"], "--n_frames": task["n_frames"], "--rgb_camera_id": task["rgb_camera_id"],
        "--rgb_hflip_p": aug["hflip_p"], "--rgb_vflip_p": aug["vflip_p"], "--rgb_jitter_p": aug["jitter_p"],
        "--rgb_jitter_brightness": aug["jitter_strength"][0], "--rgb_jitter_contrast": aug["jitter_strength"][1],
        "--rgb_jitter_saturation": aug["jitter_strength"][2], "--rgb_jitter_hue": aug["jitter_strength"][3],
        "--rgb_gray_p": aug["gray_p"], "--rgb_blur_p": aug["blur_p"], "--rgb_blur_kernel": aug["blur_kernel"],
        "--batch_size": common["batch_size"], "--num_workers": common["num_workers"], "--model_depth": common["model_depth"],
        "--proj_dim": common["proj_dim"], "--K_queue": common["K_queue"], "--temperature": common["temperature"],
        "--contrastive_loss": common["contrastive_loss"], "--num_positive": common["num_positive"],
        "--ablation_mode": exp["ablation_mode"], "--warmup_epochs": exp.get("proto_start", 50),
        "--recluster_interval": exp.get("recluster_interval", 10000), "--default_num_prototypes": exp.get("num_prototypes", 2),
        "--lambda_proto": exp.get("lambda_proto", 0.0), "--proto_temperature": v2["prediction_temperature"],
        "--proto_refresh_batch_size": common["batch_size"], "--proto_refresh_num_workers": common["num_workers"],
        "--lambda_rel": exp.get("lambda_rel", 0.0), "--proto_ema_momentum": v2["bank_momentum"],
        "--preview_ema_momentum": v2["preview_momentum"], "--rel_same_margin": 0.0, "--rel_diff_margin": 0.0,
        "--rel_same_weight": 0.0, "--rel_diff_weight": 1.0, "--rel_topk_diff_classes": v2["rel_topk"],
        "--proto_loss_start_epoch": exp.get("proto_start", 50), "--rel_loss_start_epoch": exp.get("rel_start", 75),
        "--rel_loss_end_epoch": exp.get("rel_end", 100), "--rel_lambda_schedule": exp.get("rel_schedule", "cosine"),
        "--epochs": exp.get("epochs", common["epochs"]), "--learning_rate": common["learning_rate"],
        "--weight_decay": common["weight_decay"], "--optimizer": common["optimizer"], "--seed": exp.get("seed", 1),
        "--print_freq": common["print_freq"], "--save_interval": common["save_interval"],
        "--prototype_diagnostic_interval": common["prototype_diagnostic_interval"],
        "--rel_checkpoint_after_epochs": common["rel_checkpoint_after_epochs"], "--sampler_type": exp.get("sampler_type", "none"),
    }
    for flag, value in values.items(): cmd.extend([flag, str(value)])
    for flag, seq in {"--rgb_mean": task["rgb_mean"], "--rgb_std": task["rgb_std"], "--rgb_out_hw": aug["out_hw"], "--rrc_scale": aug["rrc_scale"], "--rrc_ratio": aug["rrc_ratio"], "--rgb_blur_sigma": aug["blur_sigma"], "--schedule": common["schedule"]}.items():
        cmd.append(flag); cmd.extend(map(str, seq))
    cmd.extend(["--rgb_apply_spatial_aug", "--mlp", "--no_ddp", "--no-use_syncbn", "--verify_paths_on_init", "--proto_refresh_verify_paths_on_init", "--enable_loss_stage_schedule", "--debug_mode", "--debug_write_jsonl", "--exclude_invalid_queue"])
    payload = {"stage": stage["id"], "experiment": exp, "v2_effective": v2, "output": str(output), "command": cmd}
    print(json.dumps(payload, indent=2, ensure_ascii=False)); print("[Command] " + shlex.join(cmd))
    if args.dry_run: return
    if not args.validate_command: write_provenance(output, payload, project, [original, entry, entry.parent / "v2_losses.py", config_path])
    subprocess.run(cmd, cwd=str(project), check=True, env=dict(os.environ))


if __name__ == "__main__": main()
