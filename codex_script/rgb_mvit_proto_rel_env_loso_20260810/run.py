#!/usr/bin/env python3
"""Unified Windows/Stanage launcher for the MViT old Proto/Rel LOSO package."""
from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common.config import PACKAGE_ROOT, SELECTION_PATH, append, load_plan, read_json, roots, select


SELECTION_STAGES = ("stage1", "stage2a", "stage2b", "stage3a", "stage3b", "stage4")
STATIC_STAGES = (*SELECTION_STAGES, "stage8")
ALL_STAGES = (*SELECTION_STAGES, "stage5", "stage6", "stage7", "stage8")


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.plan = load_plan()
        self.project, self.dataset = roots(self.plan, args.platform, args.project_root, args.dataset_root)
        self.python = args.python_bin or os.environ.get("PYTHON_BIN") or sys.executable
        self.results = self.project / self.plan["output_rel"]
        self.splits = self.results / "runtime" / "splits"
        self.source = self.project / self.plan["sources"]["pretrain_script_rel"]
        self.classifier_source = self.project / self.plan["sources"]["classifier_script_rel"]
        self.classifier_src = self.project / self.plan["sources"]["classifier_src_root_rel"]
        self.classifier_entry = self.project / "codex_script" / "rgb_supcon_repair_20260806" / "common" / "classifier_entry.py"
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(self.project) + (os.pathsep + self.env["PYTHONPATH"] if self.env.get("PYTHONPATH") else "")

    def run(self, command: list[object]) -> None:
        cmd = [str(x) for x in command]
        print("[Command]", shlex.join(cmd))
        if not self.args.dry_run:
            subprocess.run(cmd, cwd=str(self.project), env=self.env, check=True)

    def static_catalog(self) -> dict[str, tuple[str, dict]]:
        catalog: dict[str, tuple[str, dict]] = {}
        for stage in SELECTION_STAGES:
            for row in self.plan["stages"][stage]:
                if row["id"] in catalog:
                    raise ValueError(f"Duplicate experiment ID: {row['id']}")
                catalog[row["id"]] = (stage, dict(row))
        return catalog

    def stage5_rows(self) -> list[dict]:
        selection = read_json(SELECTION_PATH)
        if not selection.get("stage5_ready"):
            raise RuntimeError("Stage 5 is locked. Fill best_p2/p3 Proto and Rel IDs in config/selection.json, then set stage5_ready=true.")
        catalog = self.static_catalog()

        def chosen(key: str, expected_p: int) -> dict:
            exp_id = selection.get(key, "")
            if exp_id not in catalog:
                raise ValueError(f"selection.{key} is not a Stage 1-4 experiment ID: {exp_id!r}")
            row = dict(catalog[exp_id][1])
            if int(row.get("num_prototypes", -1)) != expected_p:
                raise ValueError(f"selection.{key} must use P{expected_p}, got {row.get('num_prototypes')}")
            return row

        p2, p3 = chosen("best_p2_proto", 2), chosen("best_p3_proto", 3)
        r2, r3 = chosen("best_p2_rel", 2), chosen("best_p3_rel", 3)

        def combine(proto: dict, rel: dict, exp_id: str, description: str) -> dict:
            row = dict(proto)
            row.update({key: value for key, value in rel.items() if key.startswith("rel_") or key in {"lambda_rel", "preview_ema_momentum"}})
            row.update({"id": exp_id, "description": description, "ablation_mode": "contrastive_proto_rel", "pretrain": True, "seed": 1})
            return row

        c2 = combine(p2, r2, "c2_rgb_both", "Best selected P2 RGB Proto + P2 RGB Rel")
        c3 = combine(p3, r3, "c3_rgb_both", "Best selected P3 RGB Proto + P3 RGB Rel")
        s2 = combine(p2, r2, "c2_sensor_schedule", "Best P2 Proto with sensor-style Rel schedule")
        s3 = combine(p3, r3, "c3_sensor_schedule", "Best P3 Proto with sensor-style Rel schedule")
        for row in (s2, s3):
            row.update({"lambda_rel": 1.0, "rel_start": 50, "preview_ema_momentum": 0.5, "rel_same_weight": 1.0, "rel_diff_weight": 1.0, "rel_topk_diff_classes": 3, "rel_lambda_schedule": "cosine"})
        n2 = combine(p2, r2, "cn2_matched_null", "Matched P2 Proto+Rel state-path Null")
        n3 = combine(p3, r3, "cn3_matched_null", "Matched P3 Proto+Rel state-path Null")
        for row in (n2, n3):
            row.update({"lambda_proto": 0.0, "lambda_rel": 0.0})
        rows = [c2, c3, s2, s3, n2, n3]
        for index, row in enumerate(rows):
            row["index"] = index
        return rows

    def catalog_with_stage5(self) -> dict[str, tuple[str, dict]]:
        catalog = self.static_catalog()
        try:
            rows = self.stage5_rows()
        except RuntimeError:
            rows = []
        for row in rows:
            catalog[row["id"]] = ("stage5", dict(row))
        return catalog

    def stage6_rows(self) -> list[dict]:
        selection = read_json(SELECTION_PATH)
        if not selection.get("stage6_ready"):
            raise RuntimeError("Stage 6 is locked. Fill Stage-6 selections and set stage6_ready=true.")
        catalog = self.catalog_with_stage5()
        sources = [
            ("l0_direct", "d0_k400_direct"), ("l1_sup", "s0_sup"),
            ("l2_best_p2", selection.get("best_p2", "")),
            ("l3_best_p3", selection.get("best_p3", "")),
            ("l4_best_overall", selection.get("best_overall", "")),
            ("l5_best_p1", selection.get("best_p1", "")),
        ]
        rows = []
        for index, (new_id, source_id) in enumerate(sources):
            if source_id not in catalog:
                raise ValueError(f"Stage-6 source is invalid: {source_id!r}")
            row = dict(catalog[source_id][1])
            row.update({"index": index, "id": new_id, "source_experiment": source_id, "description": f"LOSO confirmation of {source_id}", "seed": 1})
            rows.append(row)
        return rows

    def stage7_rows(self) -> list[dict]:
        selection = read_json(SELECTION_PATH)
        if not selection.get("stage7_ready"):
            raise RuntimeError("Stage 7 is locked. Set stage7_best_overall and stage7_ready=true.")
        catalog = self.catalog_with_stage5()
        best = selection.get("stage7_best_overall", "")
        if best not in catalog:
            raise ValueError(f"Invalid stage7_best_overall: {best!r}")
        sup = dict(catalog["s0_sup"][1])
        winner = dict(catalog[best][1])
        rows = []
        for source_id, source in (("s0_sup", sup), (best, winner)):
            for seed in (2, 3):
                row = dict(source)
                row.update({"index": len(rows), "id": f"f_{'sup' if source_id == 's0_sup' else 'best'}_s{seed}", "source_experiment": source_id, "description": f"Seed-{seed} four-fold confirmation of {source_id}", "seed": seed})
                rows.append(row)
        return rows

    def rows(self, stage: str) -> list[dict]:
        if stage in STATIC_STAGES:
            rows = [dict(row) for row in self.plan["stages"][stage]]
        elif stage == "stage5":
            rows = self.stage5_rows()
        elif stage == "stage6":
            rows = self.stage6_rows()
        elif stage == "stage7":
            rows = self.stage7_rows()
        else:
            raise ValueError(stage)
        for row in rows:
            row.setdefault("pretrain", True)
        return rows

    def fold_names(self) -> list[str]:
        return list(self.plan["protocol"]["people"])

    def manifests(self, fold: str) -> tuple[Path, Path, Path]:
        root = self.splits / f"fold_{fold}"
        return root / "train.jsonl", root / "inner_val.jsonl", root / "outer_test.jsonl"

    def pretrain_dir(self, stage: str, fold: str, row: dict) -> Path:
        return self.results / "pretrain" / f"fold_{fold}" / stage / row["id"]

    def checkpoint(self, stage: str, fold: str, row: dict) -> Path:
        return self.pretrain_dir(stage, fold, row) / f"checkpoint_{self.plan['pretrain_common']['epochs']:04d}.pth"

    def prepare(self) -> None:
        cfg, manifests = self.plan["protocol"], self.plan["manifests"]
        self.run([
            self.python, PACKAGE_ROOT / "tools" / "prepare_protocol.py",
            "--dataset-root", self.dataset, "--train", manifests["train"], "--val", manifests["val"], "--test", manifests["test"],
            "--people", *cfg["people"], "--inner-val-fraction", cfg["inner_val_fraction"], "--seed", cfg["inner_split_seed"], "--output", self.splits,
        ])

    def pretrain(self, stage: str, fold: str, row: dict) -> None:
        if not row.get("pretrain", True):
            print(f"[Skip] {row['id']} is direct fine-tuning and has no pretraining phase")
            return
        train_manifest, _, _ = self.manifests(fold)
        common, aug, model = self.plan["pretrain_common"], self.plan["pretrain_augmentation"], self.plan["model"]
        out = self.pretrain_dir(stage, fold, row)
        command: list[object] = [
            self.python, "-u", PACKAGE_ROOT / "common" / "pretrain_entry.py",
            "--package-source", self.source, "--project-root", self.project,
            "--proto-positive-mode", row.get("proto_positive_mode", "all"),
            "--lr-warmup-epochs", common["warmup_epochs"], "--auto-resume",
        ]
        if self.args.validate_command:
            command.append("--parse-only")
        values = {
            "--dataset_root": self.dataset, "--train_manifest_name": train_manifest,
            "--label_map_json": self.dataset / self.plan["manifests"]["label_map"],
            "--weight_save_path": out, "--tier_mode": model["tier_mode"], "--n_frames": model["n_frames"],
            "--rgb_camera_id": model["rgb_camera_id"], "--batch_size": common["batch_size"],
            "--num_workers": common["num_workers"], "--model_depth": 18,
            "--proj_dim": common["projection_dim"], "--K_queue": common["queue_size"],
            "--temperature": common["temperature"], "--contrastive_loss": "suploss",
            "--num_positive": common["num_positive"], "--ablation_mode": row.get("ablation_mode", "contrastive_only"),
            "--warmup_epochs": row.get("warmup_epochs", row.get("proto_start", 50)),
            "--recluster_interval": common["recluster_interval"], "--default_num_prototypes": row.get("num_prototypes", 1),
            "--lambda_proto": row.get("lambda_proto", 0.0), "--proto_temperature": common["proto_temperature"],
            "--proto_kmeans_random_state": common["proto_kmeans_random_state"], "--proto_kmeans_n_init": common["proto_kmeans_n_init"],
            "--proto_kmeans_max_iter": common["proto_kmeans_max_iter"], "--proto_refresh_batch_size": common["batch_size"],
            "--proto_refresh_num_workers": common["num_workers"], "--lambda_rel": row.get("lambda_rel", 0.0),
            "--proto_ema_momentum": common["proto_ema_momentum"], "--preview_ema_momentum": row.get("preview_ema_momentum", 0.5),
            "--rel_same_margin": common["rel_same_margin"], "--rel_diff_margin": common["rel_diff_margin"],
            "--rel_same_weight": row.get("rel_same_weight", 0.0), "--rel_diff_weight": row.get("rel_diff_weight", 1.0),
            "--rel_topk_diff_classes": row.get("rel_topk_diff_classes", 3), "--proto_loss_start_epoch": row.get("proto_start", 50),
            "--rel_loss_start_epoch": row.get("rel_start", 200), "--rel_loss_end_epoch": common["epochs"],
            "--rel_lambda_schedule": row.get("rel_lambda_schedule", "constant"), "--epochs": common["epochs"],
            "--learning_rate": common["learning_rate"], "--weight_decay": common["weight_decay"], "--optimizer": common["optimizer"],
            "--seed": row.get("seed", 1), "--print_freq": common["print_freq"], "--save_interval": common["save_interval"],
            "--prototype_diagnostic_interval": common["prototype_diagnostic_interval"],
            "--rel_checkpoint_after_epochs": common["rel_checkpoint_after_epochs"],
            "--sampler_type": row.get("pretrain_sampler", "none"),
            "--rgb_hflip_p": aug["hflip_p"], "--rgb_vflip_p": aug["vflip_p"], "--rgb_jitter_p": aug["jitter_p"],
            "--rgb_jitter_brightness": aug["jitter_strength"][0], "--rgb_jitter_contrast": aug["jitter_strength"][1],
            "--rgb_jitter_saturation": aug["jitter_strength"][2], "--rgb_jitter_hue": aug["jitter_strength"][3],
            "--rgb_gray_p": aug["gray_p"], "--rgb_blur_p": aug["blur_p"], "--rgb_blur_kernel": aug["blur_kernel"],
        }
        for flag, value in values.items():
            append(command, flag, value)
        if row.get("pretrain_sampler") == "weighted":
            append(command, "--weighted_sampler_mode", row.get("weighted_sampler_mode", "sqrt_inv"))
        elif row.get("pretrain_sampler") == "balanced_batch":
            append(command, "--balanced_classes_per_batch", row.get("balanced_classes_per_batch", 16))
            append(command, "--balanced_samples_per_class", row.get("balanced_samples_per_class", 2))
        for flag, value in {
            "--rgb_mean": model["rgb_mean"], "--rgb_std": model["rgb_std"],
            "--rgb_out_hw": [model["image_size"], model["image_size"]], "--rrc_scale": aug["rrc_scale"],
            "--rrc_ratio": aug["rrc_ratio"], "--rgb_blur_sigma": aug["blur_sigma"], "--schedule": [50, 100, 150],
        }.items():
            append(command, flag, value)
        command.extend(["--rgb_apply_spatial_aug", "--mlp", "--cos", "--no_ddp", "--no-use_syncbn", "--verify_paths_on_init", "--proto_refresh_verify_paths_on_init", "--enable_loss_stage_schedule", "--debug_mode", "--debug_write_jsonl", "--exclude_invalid_queue"])
        self.run(command)

    def proto_environment(self, stage: str, fold: str, row: dict) -> None:
        if not row.get("pretrain", True) or row.get("ablation_mode") == "contrastive_only":
            print(f"[Skip] no prototype state expected for {row['id']}")
            return
        train_manifest, _, _ = self.manifests(fold)
        self.run([
            self.python, PACKAGE_ROOT / "tools" / "analyze_environment_prototypes.py",
            "--manifest", train_manifest,
            "--diagnostic-dir", self.pretrain_dir(stage, fold, row) / "prototype_diagnostics",
            "--output", self.results / "prototype_environment" / f"fold_{fold}" / stage / row["id"],
        ])

    def classifier_dir(self, stage: str, fold: str, row: dict) -> Path:
        return self.results / "classifier" / f"fold_{fold}" / stage / row["id"]

    def finetune(self, stage: str, fold: str, row: dict) -> None:
        train_manifest, val_manifest, _ = self.manifests(fold)
        out = self.classifier_dir(stage, fold, row)
        if not self.args.dry_run and list(out.rglob("best_val_balanced.pth")):
            print(f"[Skip] completed classifier: {out}")
            return
        model, common = self.plan["model"], self.plan["finetune_common"]
        command: list[object] = [
            self.python, "-u", self.classifier_entry,
            "--repair-source", self.classifier_source, "--repair-src-root", self.classifier_src,
            "--repair-representation", "rgb", "--repair-temporal-mode", "current",
            "--repair-backbone-init", "kinetics400", "--repair-finetune-policy", "full",
        ]
        if self.args.validate_command:
            command.append("--repair-parse-only")
        values = {
            "--run_mode": "train", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--dataset_root": self.dataset, "--label_map_json": self.dataset / self.plan["manifests"]["label_map"],
            "--train_manifest": train_manifest, "--val_manifest": val_manifest, "--tier_mode": model["tier_mode"],
            "--n_frames": model["n_frames"], "--use_modality": "rgb", "--num_classes": model["num_classes"],
            "--backbone": model["backbone"], "--model_depth": 18, "--rgb_camera_id": model["rgb_camera_id"],
            "--rgb_size": model["image_size"], "--rrc_scale_min": common["rrc_scale"][0], "--rrc_scale_max": common["rrc_scale"][1],
            "--rrc_ratio_min": common["rrc_ratio"][0], "--rrc_ratio_max": common["rrc_ratio"][1],
            "--rgb_hflip_p": common["hflip_p"], "--rgb_vflip_p": common["vflip_p"], "--rgb_jitter_p": common["jitter_p"],
            "--rgb_gray_p": common["gray_p"], "--rgb_blur_p": common["blur_p"], "--epochs": common["epochs"],
            "--batch_size": common["batch_size"], "--num_workers_train": common["num_workers_train"],
            "--num_workers_val": common["num_workers_val"], "--optimizer": common["optimizer"],
            "--learning_rate": common["head_learning_rate"], "--weight_decay": common["weight_decay"],
            "--seed": row.get("seed", 1), "--finetune_mode": "full", "--save_period": common["save_period"],
            "--best_after_epoch": 0, "--backbone_learning_rate": common["backbone_learning_rate"],
            "--head_learning_rate": common["head_learning_rate"],
        }
        for flag, value in values.items():
            append(command, flag, value)
        append(command, "--rgb_mean", model["rgb_mean"]); append(command, "--rgb_std", model["rgb_std"]); append(command, "--schedules", common["schedule"])
        command.extend(["--rgb_apply_spatial_aug", "--enable_amp", "--use_discriminative_lr"])
        if row.get("pretrain", True):
            append(command, "--pretrained_weight_paths", self.checkpoint(stage, fold, row))
        self.run(command)

    def best_classifier(self, stage: str, fold: str, row: dict) -> Path:
        found = list(self.classifier_dir(stage, fold, row).rglob("best_val_balanced.pth"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one best_val_balanced.pth, found {len(found)} under {self.classifier_dir(stage, fold, row)}")
        return found[0]

    def test(self, stage: str, fold: str, row: dict) -> None:
        _, _, test_manifest = self.manifests(fold)
        model = self.plan["model"]
        out = self.results / "test" / f"fold_{fold}" / stage / row["id"]
        result_csv = out / "test_results.csv"
        if not self.args.dry_run and result_csv.is_file() and result_csv.stat().st_size > 0:
            print(f"[Skip] completed outer test: {result_csv}")
            return
        command: list[object] = [
            self.python, "-u", self.classifier_entry,
            "--repair-source", self.classifier_source, "--repair-src-root", self.classifier_src,
            "--repair-representation", "rgb", "--repair-temporal-mode", "current",
            "--repair-backbone-init", "kinetics400", "--repair-finetune-policy", "full",
        ]
        values = {
            "--run_mode": "test", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--test_results_csv": result_csv, "--dataset_root": self.dataset,
            "--label_map_json": self.dataset / self.plan["manifests"]["label_map"], "--test_manifest": test_manifest,
            "--test_weight_paths": self.best_classifier(stage, fold, row) if not self.args.dry_run else self.classifier_dir(stage, fold, row) / "<run>" / "best_val_balanced.pth",
            "--tier_mode": model["tier_mode"], "--n_frames": model["n_frames"], "--use_modality": "rgb",
            "--num_classes": model["num_classes"], "--backbone": model["backbone"], "--model_depth": 18,
            "--rgb_camera_id": model["rgb_camera_id"], "--rgb_size": model["image_size"],
            "--batch_size": self.plan["finetune_common"]["batch_size"],
            "--num_workers_test": self.plan["finetune_common"]["num_workers_test"],
        }
        for flag, value in values.items():
            append(command, flag, value)
        append(command, "--rgb_mean", model["rgb_mean"]); append(command, "--rgb_std", model["rgb_std"]); command.append("--enable_amp")
        self.run(command)

    def pipeline(self, stage: str, fold: str, row: dict) -> None:
        self.pretrain(stage, fold, row)
        self.proto_environment(stage, fold, row)
        self.finetune(stage, fold, row)
        if self.args.outer_test:
            self.test(stage, fold, row)

    def summarize(self) -> None:
        self.run([self.python, PACKAGE_ROOT / "tools" / "summarize.py", "--results-root", self.results])

    def validate(self) -> None:
        forbidden_import = "rgb_proto_rel_" + "v2_20260804"
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                if any(forbidden_import in module for module in modules):
                    raise RuntimeError(f"Forbidden V2 import found in {path}: {modules}")
        expected = self.plan["stage_counts"]
        actual = {stage: len(self.rows(stage)) for stage in STATIC_STAGES}
        for stage, count in actual.items():
            if count != expected[stage]:
                raise RuntimeError(f"Count mismatch {stage}: {count} != {expected[stage]}")
        for path in (self.source, self.classifier_source, self.classifier_entry):
            if not path.is_file():
                raise FileNotFoundError(path)
        print(json.dumps({"status": "OK", "static_counts": actual, "package": str(PACKAGE_ROOT), "results": str(self.results)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "prepare", "pretrain", "proto-env", "finetune", "test", "pipeline", "summarize", "list"])
    parser.add_argument("--stage", choices=ALL_STAGES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int)
    selection.add_argument("--experiment")
    parser.add_argument("--fold", choices=["M", "J", "MR", "N", "all"])
    parser.add_argument("--outer-test", action="store_true", help="Evaluate the outer held-out person; use only for frozen configurations")
    parser.add_argument("--platform", choices=["auto", "windows", "hpc"], default="auto")
    parser.add_argument("--project-root"); parser.add_argument("--dataset-root"); parser.add_argument("--python-bin")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--validate-command", action="store_true")
    args = parser.parse_args()
    runner = Runner(args)
    if args.action == "validate":
        runner.validate(); return
    if args.action == "prepare":
        runner.prepare(); return
    if args.action == "summarize":
        runner.summarize(); return
    if args.action == "list":
        if not args.stage: parser.error("list requires --stage")
        print(json.dumps(runner.rows(args.stage), indent=2, ensure_ascii=False)); return
    if not args.stage or (args.index is None and args.experiment is None):
        parser.error(f"{args.action} requires --stage and --index/--experiment")
    row = select(runner.rows(args.stage), args.index, args.experiment)
    folds = runner.fold_names() if args.fold == "all" else [args.fold or runner.plan["protocol"]["screen_fold"]]
    for fold in folds:
        if args.action == "pretrain": runner.pretrain(args.stage, fold, row)
        elif args.action == "proto-env": runner.proto_environment(args.stage, fold, row)
        elif args.action == "finetune": runner.finetune(args.stage, fold, row)
        elif args.action == "test": runner.test(args.stage, fold, row)
        elif args.action == "pipeline": runner.pipeline(args.stage, fold, row)


if __name__ == "__main__":
    main()
