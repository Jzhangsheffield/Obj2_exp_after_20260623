#!/usr/bin/env python3
"""Unified Windows/HPC launcher driven by config/experiment_config.json."""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.config import (
    PACKAGE_ROOT,
    command_text,
    experiment_row,
    load_config,
    manifest_dir,
    resolve_paths,
    task_for_stage,
    write_json,
)


PRETRAIN_STAGES = (
    "takeput_pretrain",
    "middle_backbone_pretrain",
    "middle_aug",
    "middle_loss_screen",
    "middle_rel_topk",
    "middle_rel_start",
    "middle_combined",
    "middle_p2_sentinel",
)
DIRECT_STAGES = ("takeput_direct", "middle_direct")
ALL_STAGES = (*DIRECT_STAGES, *PRETRAIN_STAGES)


def add_values(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.append(flag)
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_config(args.config)
        overrides = {
            "project_root": args.project_root,
            "dataset_root": args.dataset_root,
            "results_root": args.results_root,
            "python_bin": args.python_bin,
        }
        self.paths = resolve_paths(self.cfg, args.platform, overrides)
        self.project: Path = self.paths["project_root"]  # type: ignore[assignment]
        self.dataset: Path = self.paths["dataset_root"]  # type: ignore[assignment]
        self.results: Path = self.paths["results_root"]  # type: ignore[assignment]
        configured_python = str(self.paths["python_bin"])
        self.python = configured_python if configured_python != "python" else sys.executable
        self.source = self.project / self.cfg["sources"]["old_pretrain_script"]
        self.classifier_source = self.project / self.cfg["sources"]["classifier_script"]
        self.classifier_src = self.project / self.cfg["sources"]["classifier_src_root"]
        self.pretrain_entry = PACKAGE_ROOT / "common" / "pretrain_entry.py"
        self.classifier_entry = PACKAGE_ROOT / "common" / "classifier_entry.py"
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(self.project) + (
            os.pathsep + self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else ""
        )

    def execute(self, command: list[str], metadata: dict | None = None, output: Path | None = None) -> None:
        print("[Command]", command_text(command))
        if output is not None:
            resolved = {
                "command": command,
                "config": self.cfg,
                "paths": {key: str(value) for key, value in self.paths.items()},
                "metadata": metadata or {},
            }
            if not self.args.dry_run:
                write_json(output / "resolved_experiment.json", resolved)
        if not self.args.dry_run:
            subprocess.run(command, cwd=str(self.project), env=self.env, check=True)

    def manifests(self, stage: str, fold: str) -> tuple[Path, Path, Path]:
        task = task_for_stage(stage)
        root = manifest_dir(self.paths, self.cfg, task, fold)
        return root / "train.jsonl", root / "test.jsonl", root / "label_map.json"

    def prepare(self) -> None:
        output = PACKAGE_ROOT / self.cfg["data"]["manifest_output_rel"]
        command = [
            self.python,
            str(PACKAGE_ROOT / "tools" / "prepare_manifests.py"),
            "--config",
            str(self.paths["config_path"]),
            "--dataset-root",
            str(self.dataset),
            "--output",
            str(output),
        ]
        self.execute(command)

    def pretrain_dir(self, stage: str, fold: str, row: dict) -> Path:
        return self.results / "pretrain" / task_for_stage(stage) / fold / stage / row["id"]

    def pretrain_checkpoint(self, stage: str, fold: str, row: dict) -> Path:
        epochs = int(self.cfg["training"]["pretrain"]["epochs"])
        return self.pretrain_dir(stage, fold, row) / f"checkpoint_{epochs:04d}.pth"

    def effective_row(self, stage: str, row: dict) -> dict:
        result = dict(row)
        if stage.startswith("middle_"):
            result.setdefault("backbone", "tv_r3d18" if stage == "middle_aug" else "mvit_v2_s")
            result.setdefault("backbone_init", "kinetics400")
            result.setdefault("augmentation", self.cfg["selected"]["middle_augmentation"])
            result.setdefault("ablation_mode", "contrastive_only")
            result.setdefault("num_prototypes", 1)
            result.setdefault("lambda_proto", 0.0)
            result.setdefault("lambda_rel", 0.0)
        if self.args.augmentation:
            result["augmentation"] = self.args.augmentation
        return result

    def pretrain(self, stage: str, fold: str, original_row: dict) -> None:
        if stage not in PRETRAIN_STAGES:
            raise ValueError(f"{stage} has no pretraining phase")
        row = self.effective_row(stage, original_row)
        train_manifest, _, label_map = self.manifests(stage, fold)
        common = self.cfg["training"]["pretrain"]
        model = self.cfg["model"]
        aug = self.cfg["augmentation_profiles"][row["augmentation"]]
        out = self.pretrain_dir(stage, fold, row)
        command = [
            self.python,
            "-u",
            str(self.pretrain_entry),
            "--package-source",
            str(self.source),
            "--project-root",
            str(self.project),
            "--exp-backbone",
            row["backbone"],
            "--exp-backbone-init",
            row["backbone_init"],
            "--proto-positive-mode",
            row.get("proto_positive_mode", "all"),
            "--lr-warmup-epochs",
            str(common["warmup_epochs"]),
            "--auto-resume",
        ]
        if self.args.parse_only:
            command.append("--parse-only")
        rel_start = int(row.get("rel_start", self.cfg["selected"]["middle_loss_start_epoch"]))
        proto_start = int(row.get("proto_start", self.cfg["selected"]["middle_loss_start_epoch"]))
        rel_topk = int(row.get("rel_topk", self.cfg["selected"]["middle_rel_topk"]))
        values = {
            "--dataset_root": self.dataset,
            "--train_manifest_name": train_manifest,
            "--label_map_json": label_map,
            "--weight_save_path": out,
            "--tier_mode": model["tier_mode"],
            "--n_frames": model["n_frames"],
            "--rgb_camera_id": model["rgb_camera_id"],
            "--batch_size": common["batch_size"],
            "--num_workers": common["num_workers"],
            "--model_depth": 18,
            "--proj_dim": model["projection_dim"],
            "--K_queue": model["queue_size"],
            "--temperature": common["temperature"],
            "--contrastive_loss": "suploss",
            "--num_positive": 6,
            "--ablation_mode": row["ablation_mode"],
            "--warmup_epochs": proto_start,
            "--recluster_interval": common["recluster_interval"],
            "--default_num_prototypes": row["num_prototypes"],
            "--lambda_proto": row["lambda_proto"],
            "--proto_temperature": common["proto_temperature"],
            "--proto_kmeans_random_state": common["proto_kmeans_random_state"],
            "--proto_kmeans_n_init": common["proto_kmeans_n_init"],
            "--proto_kmeans_max_iter": common["proto_kmeans_max_iter"],
            "--proto_refresh_batch_size": common["batch_size"],
            "--proto_refresh_num_workers": common["num_workers"],
            "--lambda_rel": row["lambda_rel"],
            "--proto_ema_momentum": common["proto_ema_momentum"],
            "--preview_ema_momentum": common["preview_ema_momentum"],
            "--rel_same_margin": common["rel_same_margin"],
            "--rel_diff_margin": common["rel_diff_margin"],
            "--rel_same_weight": 1.0,
            "--rel_diff_weight": 1.0,
            "--rel_topk_diff_classes": rel_topk,
            "--proto_loss_start_epoch": proto_start,
            "--rel_loss_start_epoch": rel_start,
            "--rel_loss_end_epoch": common["epochs"],
            "--rel_lambda_schedule": "cosine",
            "--epochs": common["epochs"],
            "--learning_rate": common["learning_rate"],
            "--weight_decay": common["weight_decay"],
            "--optimizer": common["optimizer"],
            "--seed": common.get("seed", self.cfg["training"]["seed"]),
            "--print_freq": 20,
            "--save_interval": common["save_interval"],
            "--prototype_diagnostic_interval": common["prototype_diagnostic_interval"],
            "--rel_checkpoint_after_epochs": common["rel_checkpoint_after_epochs"],
            "--sampler_type": "none",
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
        }
        for flag, value in values.items():
            add_values(command, flag, value)
        for flag, value in {
            "--rgb_mean": model["rgb_mean"],
            "--rgb_std": model["rgb_std"],
            "--rgb_out_hw": [model["image_size"], model["image_size"]],
            "--rrc_scale": aug["rrc_scale"],
            "--rrc_ratio": aug["rrc_ratio"],
            "--rgb_blur_sigma": aug["blur_sigma"],
            "--schedule": [50, 100, 150],
        }.items():
            add_values(command, flag, value)
        command.extend(
            [
                "--rgb_apply_spatial_aug",
                "--mlp",
                "--cos",
                "--no_ddp",
                "--no-use_syncbn",
                "--verify_paths_on_init",
                "--proto_refresh_verify_paths_on_init",
                "--enable_loss_stage_schedule",
                "--debug_mode",
                "--debug_write_jsonl",
                "--debug_grad_stats",
                "--debug_param_update_stats",
                "--debug_batch_label_stats",
                "--debug_proto_stats",
                "--debug_feature_stats",
                "--debug_nonfinite_check",
                "--exclude_invalid_queue",
            ]
        )
        add_values(command, "--debug_log_interval", common["debug_log_interval"])
        add_values(command, "--debug_grad_topk", common["debug_grad_topk"])
        self.execute(command, {"stage": stage, "fold": fold, "row": row}, out)

    def classifier_dir(self, stage: str, fold: str, row: dict, policy: str) -> Path:
        return self.results / "classifier" / task_for_stage(stage) / fold / stage / row["id"] / policy

    def classifier(self, stage: str, fold: str, original_row: dict, policy_override: str | None = None) -> None:
        row = self.effective_row(stage, original_row)
        train_manifest, test_manifest, label_map = self.manifests(stage, fold)
        is_direct = stage in DIRECT_STAGES
        policy = policy_override or row.get("finetune_policy")
        if policy not in {"full", "head_only"}:
            raise ValueError("Specify --policy full/head_only for a pretrained experiment")
        common = self.cfg["training"]["direct_classifier" if is_direct else "downstream_finetune"]
        model = self.cfg["model"]
        out = self.classifier_dir(stage, fold, row, policy)
        init = row["backbone_init"]
        command = [
            self.python,
            "-u",
            str(self.classifier_entry),
            "--package-source",
            str(self.classifier_source),
            "--src-root",
            str(self.classifier_src),
            "--backbone-init",
            init,
        ]
        if self.args.parse_only:
            command.append("--parse-only")
        aug = self.cfg["training"]["classifier_augmentation"]
        values = {
            "--run_mode": "train",
            "--save_path": out,
            "--datamap_csv_path": out / "datamaps",
            "--dataset_root": self.dataset,
            "--label_map_json": label_map,
            "--train_manifest": train_manifest,
            "--val_manifest": test_manifest,
            "--tier_mode": model["tier_mode"],
            "--n_frames": model["n_frames"],
            "--use_modality": "rgb",
            "--num_classes": len(self.cfg["data"]["tasks"][task_for_stage(stage)]),
            "--backbone": row["backbone"],
            "--model_depth": 18,
            "--rgb_camera_id": model["rgb_camera_id"],
            "--rgb_size": model["image_size"],
            "--rrc_scale_min": aug["rrc_scale"][0],
            "--rrc_scale_max": aug["rrc_scale"][1],
            "--rrc_ratio_min": aug["rrc_ratio"][0],
            "--rrc_ratio_max": aug["rrc_ratio"][1],
            "--rgb_hflip_p": aug["hflip_p"],
            "--rgb_vflip_p": aug["vflip_p"],
            "--rgb_jitter_p": aug["jitter_p"],
            "--rgb_gray_p": aug["gray_p"],
            "--rgb_blur_p": aug["blur_p"],
            "--epochs": common["epochs"],
            "--batch_size": common["batch_size"],
            "--num_workers_train": common["num_workers_train"],
            "--num_workers_val": common.get("num_workers_test", 8),
            "--optimizer": common["optimizer"],
            "--learning_rate": common["head_learning_rate"],
            "--backbone_learning_rate": common["backbone_learning_rate"],
            "--head_learning_rate": common["head_learning_rate"],
            "--weight_decay": common["weight_decay"],
            "--seed": self.cfg["training"]["seed"],
            "--finetune_mode": policy,
            "--save_period": common["save_period"],
            "--best_after_epoch": 0,
        }
        for flag, value in values.items():
            add_values(command, flag, value)
        add_values(command, "--rgb_mean", model["rgb_mean"])
        add_values(command, "--rgb_std", model["rgb_std"])
        add_values(command, "--schedules", common["schedule"])
        command.extend(["--rgb_apply_spatial_aug", "--enable_amp"])
        if policy == "full":
            command.append("--use_discriminative_lr")
        if not is_direct:
            add_values(command, "--pretrained_weight_paths", self.pretrain_checkpoint(stage, fold, row))
        self.execute(
            command,
            {"stage": stage, "fold": fold, "row": row, "policy": policy, "held_out_used_each_epoch": True},
            out,
        )

    def diagnostics(self, stage: str, fold: str, row: dict) -> None:
        row = self.effective_row(stage, row)
        pretrain_dir = self.pretrain_dir(stage, fold, row)
        out = self.results / "analysis" / task_for_stage(stage) / fold / stage / row["id"] / "training_diagnostics"
        command = [
            self.python,
            str(PACKAGE_ROOT / "tools" / "analyze_training_diagnostics.py"),
            "--pretrain-dir",
            str(pretrain_dir),
            "--output",
            str(out),
        ]
        self.execute(command)

    def features(self, stage: str, fold: str, original_row: dict, checkpoint_kind: str, policy: str | None) -> None:
        row = self.effective_row(stage, original_row)
        train_manifest, test_manifest, label_map = self.manifests(stage, fold)
        if checkpoint_kind == "pretrain":
            if stage in DIRECT_STAGES:
                raise ValueError("Direct-classification rows have no pretrain checkpoint")
            checkpoint = self.pretrain_checkpoint(stage, fold, row)
            tag = "pretrain_epoch200"
        else:
            selected_policy = policy or row.get("finetune_policy")
            if selected_policy not in {"full", "head_only"}:
                raise ValueError("Classifier feature analysis requires --policy full/head_only")
            root = self.classifier_dir(stage, fold, row, selected_policy)
            found = sorted(root.rglob("best_val_balanced.pth"))
            if len(found) != 1:
                if self.args.dry_run:
                    checkpoint = root / "<run_name>" / "best_val_balanced.pth"
                else:
                    raise FileNotFoundError(f"Expected one best_val_balanced.pth under {root}, found {len(found)}")
            else:
                checkpoint = found[0]
            tag = f"classifier_{selected_policy}"
        output = self.results / "analysis" / task_for_stage(stage) / fold / stage / row["id"] / tag
        model = self.cfg["model"]
        command = [
            self.python,
            str(PACKAGE_ROOT / "tools" / "analyze_features.py"),
            "--src-root", str(self.classifier_src),
            "--dataset-root", str(self.dataset),
            "--train-manifest", str(train_manifest),
            "--test-manifest", str(test_manifest),
            "--label-map", str(label_map),
            "--checkpoint", str(checkpoint),
            "--checkpoint-kind", checkpoint_kind,
            "--backbone", row["backbone"],
            "--backbone-init", row["backbone_init"],
            "--num-classes", str(len(self.cfg["data"]["tasks"][task_for_stage(stage)])),
            "--rgb-camera-id", str(model["rgb_camera_id"]),
            "--n-frames", str(model["n_frames"]),
            "--image-size", str(model["image_size"]),
            "--queue-size", str(model["queue_size"]),
            "--proj-dim", str(model["projection_dim"]),
            "--batch-size", str(self.cfg["training"]["pretrain"]["batch_size"]),
            "--num-workers", str(self.cfg["training"]["pretrain"]["num_workers"]),
            "--seed", str(self.cfg["training"]["seed"]),
            "--output", str(output),
        ]
        add_values(command, "--rgb-mean", model["rgb_mean"])
        add_values(command, "--rgb-std", model["rgb_std"])
        self.execute(command)

    def validate(self) -> None:
        errors = []
        for path in (self.source, self.classifier_source, self.pretrain_entry, self.classifier_entry):
            if not path.is_file():
                errors.append(f"Missing: {path}")
        forbidden = "rgb_proto_rel_v2_20260804"
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    module = " ".join(alias.name for alias in node.names)
                if forbidden in module:
                    errors.append(f"Forbidden V2 import: {path}")
        pre = self.cfg["training"]["pretrain"]
        if pre["save_interval"] != 50 or pre["epochs"] != 200:
            errors.append("Pretraining must be 200 epochs with save_interval=50")
        if pre["prototype_diagnostic_interval"] != 1:
            errors.append("prototype_diagnostic_interval must be 1")
        if pre["rel_checkpoint_after_epochs"] != 0:
            errors.append("rel_checkpoint_after_epochs must be 0 to avoid extra checkpoints")
        source_text = self.source.read_text(encoding="utf-8") if self.source.is_file() else ""
        classifier_text = self.classifier_source.read_text(encoding="utf-8") if self.classifier_source.is_file() else ""
        for flag in (
            "--prototype_diagnostic_interval", "--rel_checkpoint_after_epochs", "--debug_write_jsonl",
            "--debug_grad_topk", "--preview_ema_momentum", "--rel_topk_diff_classes",
        ):
            if flag not in source_text:
                errors.append(f"Old pretrain source misses required option {flag}")
        for flag in (
            "--train_manifest", "--val_manifest", "--finetune_mode", "--pretrained_weight_paths",
            "--num_workers_val", "--schedules", "--save_period", "--best_after_epoch",
        ):
            if flag not in classifier_text:
                errors.append(f"Classifier source misses required option {flag}")
        if self.cfg["experiment_grids"].get("middle_direct") != self.cfg["experiment_grids"].get("takeput_direct"):
            errors.append("middle_direct must exactly mirror the takeput_direct backbone/init/policy grid")
        if self.cfg["experiment_grids"].get("middle_backbone_pretrain") != self.cfg["experiment_grids"].get("takeput_pretrain"):
            errors.append("middle_backbone_pretrain must exactly mirror the takeput_pretrain SupLoss grid")
        prototypes = [
            int(row.get("num_prototypes", 1))
            for stage, rows in self.cfg["experiment_grids"].items()
            for row in rows
            if stage.startswith("middle_")
        ]
        if max(prototypes, default=1) > 2 or prototypes.count(2) != 2:
            errors.append("Middle grid must contain only the two P2 sentinel rows and no P3 rows")
        audit_path = PACKAGE_ROOT / self.cfg["data"]["manifest_output_rel"] / "manifest_audit.json"
        manifest_totals = {}
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            manifest_totals = {name: value["total"] for name, value in audit["tasks"].items()}
            if manifest_totals != {"take_put": 2920, "middle": 1457, "full": 4543}:
                errors.append(f"Unexpected manifest totals: {manifest_totals}")
        if errors:
            raise RuntimeError("\n".join(errors))
        print(
            json.dumps(
                {
                    "status": "OK",
                    "package": str(PACKAGE_ROOT),
                    "platform": self.paths["platform"],
                    "grids": {name: len(rows) for name, rows in self.cfg["experiment_grids"].items()},
                    "checkpoint_epochs": [50, 100, 150, 200],
                    "classifier_test_each_epoch": True,
                    "manifest_totals": manifest_totals or "not_generated_yet",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "prepare", "list", "pretrain", "classify", "pipeline", "diagnostics", "features", "summarize"])
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=ALL_STAGES)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--index", type=int)
    select.add_argument("--experiment-id")
    parser.add_argument("--fold", choices=["dev_N", "test_M", "test_J", "test_MR"], default="dev_N")
    parser.add_argument("--policy", choices=["full", "head_only"])
    parser.add_argument("--checkpoint-kind", choices=["pretrain", "classifier"], default="classifier")
    parser.add_argument("--augmentation")
    parser.add_argument("--platform", choices=["auto", "windows", "hpc"], default="auto")
    parser.add_argument("--project-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--results-root")
    parser.add_argument("--python-bin")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    runner = Runner(args)
    if args.action == "validate":
        runner.validate()
        return
    if args.action == "prepare":
        runner.prepare()
        return
    if args.action == "summarize":
        runner.execute([runner.python, str(PACKAGE_ROOT / "tools" / "summarize_results.py"), "--results-root", str(runner.results)])
        return
    if not args.stage:
        parser.error(f"{args.action} requires --stage")
    if args.action == "list":
        print(json.dumps(runner.cfg["experiment_grids"][args.stage], ensure_ascii=False, indent=2))
        return
    row = experiment_row(runner.cfg, args.stage, args.index, args.experiment_id)
    if args.action == "pretrain":
        runner.pretrain(args.stage, args.fold, row)
    elif args.action == "classify":
        runner.classifier(args.stage, args.fold, row, args.policy)
    elif args.action == "diagnostics":
        runner.diagnostics(args.stage, args.fold, row)
    elif args.action == "features":
        runner.features(args.stage, args.fold, row, args.checkpoint_kind, args.policy)
    elif args.action == "pipeline":
        if args.stage not in DIRECT_STAGES:
            runner.pretrain(args.stage, args.fold, row)
            policies = [args.policy] if args.policy else ["full", "head_only"]
            for policy in policies:
                runner.classifier(args.stage, args.fold, row, policy)
            runner.diagnostics(args.stage, args.fold, row)
        else:
            runner.classifier(args.stage, args.fold, row, args.policy)


if __name__ == "__main__":
    main()
