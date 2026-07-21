#!/usr/bin/env python3
"""One launcher for Windows and HPC stages in this isolated package."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT / "src"
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "experiments.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select(items: list[dict[str, Any]], experiment: str | None, index: int | None) -> dict[str, Any]:
    if index is not None:
        if index < 0 or index >= len(items):
            raise IndexError(f"index {index} outside 0..{len(items)-1}")
        return items[index]
    if experiment is None:
        raise ValueError("Provide --experiment or --index")
    matches = [x for x in items if x["id"] == experiment]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate experiment id: {experiment}")
    return matches[0]


def append_values(cmd: list[str], flag: str, values: Any) -> None:
    cmd.append(flag)
    if isinstance(values, (list, tuple)):
        cmd.extend(str(x) for x in values)
    else:
        cmd.append(str(values))


class Launcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = load_json(args.config.resolve())
        platform = args.platform
        if platform == "auto":
            platform = "windows" if os.name == "nt" else "hpc"
        self.platform = platform
        roots = self.cfg["roots"]
        self.project = Path(args.project_root or roots[f"{platform}_project"]).resolve()
        self.dataset = Path(args.dataset_root or roots[f"{platform}_dataset"]).resolve()
        self.python = args.python_bin or sys.executable
        self.output = self.project / self.cfg["output_rel"]
        self.runtime = self.output / "runtime"
        self.task = self.cfg["task"]
        self.env = dict(os.environ)
        old_pythonpath = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
        self.env.setdefault("PYTHONHASHSEED", str(self.cfg["seed"]))
        self.env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @property
    def crop_stats_path(self) -> Path:
        return self.runtime / "motion_crop_train_rgb_stats.json"

    def run(self, cmd: list[str]) -> None:
        print("[Command]", shlex.join(cmd))
        if self.args.dry_run:
            return
        subprocess.run(cmd, cwd=self.project, env=self.env, check=True)

    def rgb(self, source: str, allow_missing_stats: bool = False) -> dict[str, Any]:
        if source == "original":
            return {
                "camera": self.task["original_rgb_key"],
                "mean": self.task["original_mean"],
                "std": self.task["original_std"],
                "preserve": False,
            }
        if source != "motion_crop":
            raise ValueError(f"Unknown rgb_source={source!r}")
        if self.crop_stats_path.is_file():
            stats = load_json(self.crop_stats_path)
            mean, std = stats["mean"], stats["std"]
        elif allow_missing_stats or self.args.dry_run:
            mean, std = self.task["original_mean"], self.task["original_std"]
            print(f"[Dry/validate fallback] crop stats not yet present: {self.crop_stats_path}")
        else:
            raise FileNotFoundError(
                f"Crop statistics are required: {self.crop_stats_path}. Run action=stats first."
            )
        return {
            "camera": self.task["motion_crop_rgb_key"],
            "mean": mean,
            "std": std,
            "preserve": True,
        }

    def validate(self) -> None:
        failures = []
        warnings = []
        audit = {}
        for rel in (self.task["train_manifest"], self.task["val_manifest"], self.task["test_manifest"], self.task["label_map"]):
            path = self.dataset / rel
            if not path.is_file():
                failures.append(f"missing: {path}")
        for exp in self.cfg["pretrain_experiments"]:
            batch = int(exp["batch_size"])
            queue = int(self.cfg["pretrain_common"]["queue_size"])
            if queue % batch:
                failures.append(f"queue_size={queue} is not divisible by batch_size={batch}: {exp['id']}")
        required = {"rgb_cam_00143", "rgb_cam_00143_motion_crop_m32"}
        for split_name in ("train_manifest", "val_manifest", "test_manifest"):
            manifest = self.dataset / self.task[split_name]
            if not manifest.is_file():
                continue
            total = fallback = 0
            missing_paths = []
            labels = set()
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    total += 1
                    labels.add(rec.get(self.task["tier_mode"]))
                    if rec.get("rgb_cam_00143_motion_crop_m32_motion_found") is False:
                        fallback += 1
                    for key in required:
                        rel = rec.get(key)
                        if not rel or not (self.dataset / rel).is_file():
                            missing_paths.append((rec.get("sample_name"), key, rel))
            audit[split_name] = {"samples": total, "classes": len(labels), "motion_fallback": fallback}
            if missing_paths:
                failures.append(f"{split_name}: {len(missing_paths)} missing RGB paths; first={missing_paths[:3]}")
            if len(labels) != int(self.task["num_classes"]):
                message = f"{split_name}: configured {self.task['num_classes']} classes, split contains {len(labels)}"
                if split_name == "train_manifest":
                    failures.append(message)
                else:
                    warnings.append(message)
        runtime_cmd = [self.python, str(PACKAGE_ROOT / "tools" / "validate_runtime.py"), "--src-root", str(SRC_ROOT)]
        if failures:
            raise RuntimeError("Package validation failed:\n- " + "\n- ".join(failures))
        self.run(runtime_cmd)
        print("[Dataset audit]", json.dumps(audit, ensure_ascii=False))
        for message in warnings:
            print("[Warning]", message)
        print("[OK] configuration, dataset paths and runtime validation passed")

    def stats(self) -> None:
        cmd = [
            self.python, str(PACKAGE_ROOT / "tools" / "compute_rgb_stats.py"),
            "--dataset-root", str(self.dataset),
            "--manifest", self.task["train_manifest"],
            "--rgb-key", "rgb_cam_00143_motion_crop_m32",
            "--output", str(self.crop_stats_path),
        ]
        self.run(cmd)

    def pretrain_checkpoint(self, experiment_id: str) -> Path:
        epochs = int(self.cfg["pretrain_common"]["epochs"])
        return self.output / "pretrain" / experiment_id / f"checkpoint_{epochs:04d}.pth"

    def pretrain(self, exp: dict[str, Any]) -> None:
        common, aug = self.cfg["pretrain_common"], self.cfg["augmentation"]
        rgb = self.rgb(exp["rgb_source"])
        out = self.output / "pretrain" / exp["id"]
        cmd = [self.python, "-u", str(SRC_ROOT / "train" / "pretrain_supcon.py")]
        values = {
            "--dataset_root": self.dataset,
            "--train_manifest_name": self.dataset / self.task["train_manifest"],
            "--label_map_json": self.dataset / self.task["label_map"],
            "--weight_save_path": out,
            "--tier_mode": self.task["tier_mode"],
            "--n_frames": self.task["n_frames"],
            "--backbone": exp["backbone"],
            "--model_depth": 18,
            "--rgb_camera_id": rgb["camera"],
            "--temporal_view_mode": aug["temporal_view_mode"],
            "--batch_size": exp["batch_size"],
            "--num_workers": common["num_workers"],
            "--proj_dim": common["proj_dim"],
            "--K_queue": common["queue_size"],
            "--temperature": common["temperature"],
            "--contrastive_loss": "suploss",
            "--num_positive": common["num_positive"],
            "--ablation_mode": "contrastive_only",
            "--epochs": common["epochs"],
            "--learning_rate": exp["learning_rate"],
            "--weight_decay": common["weight_decay"],
            "--optimizer": common["optimizer"],
            "--seed": self.cfg["seed"],
            "--save_interval": common["save_interval"],
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
            append_values(cmd, flag, value)
        for flag, value in {
            "--rgb_mean": rgb["mean"], "--rgb_std": rgb["std"],
            "--rgb_out_hw": [self.task["image_size"], self.task["image_size"]],
            "--rrc_scale": aug["rrc_scale"], "--rrc_ratio": aug["rrc_ratio"],
            "--rgb_blur_sigma": aug["blur_sigma"], "--schedule": common["schedule"],
        }.items():
            append_values(cmd, flag, value)
        cmd.extend(["--rgb_apply_spatial_aug", "--mlp", "--no_ddp", "--no-use_syncbn",
                    "--verify_paths_on_init", "--exclude_invalid_queue"])
        if rgb["preserve"]:
            cmd.append("--rgb_preserve_aspect_pad")
        final_checkpoint = self.pretrain_checkpoint(exp["id"])
        if final_checkpoint.is_file() and not self.args.dry_run:
            print(f"[Skip] completed pretraining checkpoint exists: {final_checkpoint}")
            return
        candidates = sorted(out.glob("checkpoint_*.pth")) if out.is_dir() else []
        if candidates:
            append_values(cmd, "--resume_checkpoint", candidates[-1])
        self.run(cmd)

    def classifier(self, exp: dict[str, Any]) -> None:
        common, aug = self.cfg["classifier_common"], self.cfg["augmentation"]
        rgb = self.rgb(exp["rgb_source"])
        out = self.output / "classifier" / exp["id"]
        completed = sorted(out.rglob("last.pth")) if out.is_dir() else []
        if completed and not self.args.dry_run:
            print(f"[Skip] completed classifier checkpoint exists: {completed[0]}")
            return
        cmd = [self.python, "-u", str(SRC_ROOT / "ft_and_test" / "train_classifier.py")]
        values = {
            "--run_mode": "train", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--dataset_root": self.dataset, "--label_map_json": self.dataset / self.task["label_map"],
            "--train_manifest": self.dataset / self.task["train_manifest"],
            "--val_manifest": self.dataset / self.task["val_manifest"],
            "--tier_mode": self.task["tier_mode"], "--n_frames": self.task["n_frames"],
            "--use_modality": "rgb", "--num_classes": self.task["num_classes"],
            "--backbone": exp["backbone"], "--model_depth": 18,
            "--rgb_camera_id": rgb["camera"], "--rgb_size": self.task["image_size"],
            "--rrc_scale_min": aug["rrc_scale"][0], "--rrc_scale_max": aug["rrc_scale"][1],
            "--rrc_ratio_min": aug["rrc_ratio"][0], "--rrc_ratio_max": aug["rrc_ratio"][1],
            "--rgb_hflip_p": aug["hflip_p"], "--rgb_vflip_p": aug["vflip_p"],
            "--rgb_jitter_p": 0.0, "--rgb_gray_p": 0.0, "--rgb_blur_p": 0.0,
            "--epochs": common["epochs"], "--batch_size": exp["batch_size"],
            "--num_workers_train": common["num_workers_train"],
            "--num_workers_val": common["num_workers_val"],
            "--optimizer": common["optimizer"], "--learning_rate": exp["learning_rate"],
            "--weight_decay": common["weight_decay"], "--seed": self.cfg["seed"],
            "--finetune_mode": exp["mode"], "--save_period": common["save_period"],
            "--best_after_epoch": 0,
        }
        for flag, value in values.items():
            append_values(cmd, flag, value)
        append_values(cmd, "--rgb_mean", rgb["mean"])
        append_values(cmd, "--rgb_std", rgb["std"])
        append_values(cmd, "--schedules", common["schedule"])
        cmd.extend(["--rgb_apply_spatial_aug", "--enable_amp"])
        if rgb["preserve"]:
            cmd.append("--rgb_preserve_aspect_pad")
        if exp["pretrain_id"]:
            checkpoint = self.pretrain_checkpoint(exp["pretrain_id"])
            if not checkpoint.is_file() and not self.args.dry_run:
                raise FileNotFoundError(f"Required pretrain checkpoint missing: {checkpoint}")
            append_values(cmd, "--pretrained_weight_paths", checkpoint)
        self.run(cmd)

    def classifier_checkpoint(self, experiment_id: str) -> Path:
        root = self.output / "classifier" / experiment_id
        found = sorted(root.rglob("best_val_balanced.pth")) if root.is_dir() else []
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one best_val_balanced.pth under {root}, found {len(found)}")
        return found[0]

    def test(self, exp: dict[str, Any]) -> None:
        rgb = self.rgb(exp["rgb_source"])
        checkpoint = (
            self.output / "classifier" / exp["id"] / "<run_dir>" / "best_val_balanced.pth"
            if self.args.dry_run else self.classifier_checkpoint(exp["id"])
        )
        out = self.output / "test" / exp["id"]
        cmd = [self.python, "-u", str(SRC_ROOT / "ft_and_test" / "train_classifier.py")]
        values = {
            "--run_mode": "test", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--test_results_csv": out / "test_results.csv",
            "--dataset_root": self.dataset, "--label_map_json": self.dataset / self.task["label_map"],
            "--test_manifest": self.dataset / self.task["test_manifest"],
            "--test_weight_paths": checkpoint,
            "--tier_mode": self.task["tier_mode"], "--n_frames": self.task["n_frames"],
            "--use_modality": "rgb", "--num_classes": self.task["num_classes"],
            "--backbone": exp["backbone"], "--model_depth": 18,
            "--rgb_camera_id": rgb["camera"], "--rgb_size": self.task["image_size"],
            "--batch_size": exp["batch_size"],
            "--num_workers_test": self.cfg["classifier_common"]["num_workers_test"],
        }
        for flag, value in values.items():
            append_values(cmd, flag, value)
        append_values(cmd, "--rgb_mean", rgb["mean"])
        append_values(cmd, "--rgb_std", rgb["std"])
        cmd.append("--enable_amp")
        if rgb["preserve"]:
            cmd.append("--rgb_preserve_aspect_pad")
        self.run(cmd)

    def features(self, exp: dict[str, Any]) -> None:
        rgb = self.rgb(exp["rgb_source"])
        checkpoint = self.pretrain_checkpoint(exp["id"])
        if not checkpoint.is_file() and not self.args.dry_run:
            raise FileNotFoundError(checkpoint)
        cmd = [
            self.python, "-u", str(PACKAGE_ROOT / "tools" / "evaluate_features.py"),
            "--src-root", str(SRC_ROOT), "--dataset-root", str(self.dataset),
            "--train-manifest", self.task["train_manifest"],
            "--test-manifest", self.task["test_manifest"],
            "--label-map", self.task["label_map"], "--tier-mode", self.task["tier_mode"],
            "--num-classes", str(self.task["num_classes"]), "--n-frames", str(self.task["n_frames"]),
            "--image-size", str(self.task["image_size"]), "--rgb-camera-id", rgb["camera"],
            "--backbone", exp["backbone"], "--checkpoint", str(checkpoint),
            "--queue-size", str(self.cfg["pretrain_common"]["queue_size"]),
            "--proj-dim", str(self.cfg["pretrain_common"]["proj_dim"]),
            "--temperature", str(self.cfg["pretrain_common"]["temperature"]),
            "--batch-size", str(exp["batch_size"]),
            "--num-workers", str(self.cfg["pretrain_common"]["num_workers"]),
            "--output", str(self.output / "features" / exp["id"]),
            "--seed", str(self.cfg["seed"]), "--rgb-mean", *map(str, rgb["mean"]),
            "--rgb-std", *map(str, rgb["std"]),
        ]
        if rgb["preserve"]:
            cmd.append("--rgb-preserve-aspect-pad")
        self.run(cmd)

    def summarize(self) -> None:
        self.run([
            self.python, str(PACKAGE_ROOT / "tools" / "summarize_results.py"),
            "--results-root", str(self.output),
        ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["validate", "stats", "pretrain", "classifier", "test", "features", "summarize"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--platform", choices=["auto", "windows", "hpc"], default="auto")
    p.add_argument("--project-root")
    p.add_argument("--dataset-root")
    p.add_argument("--python-bin")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--experiment")
    group.add_argument("--index", type=int)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    launcher = Launcher(args)
    if args.action == "validate":
        launcher.validate()
    elif args.action == "stats":
        launcher.stats()
    elif args.action == "pretrain":
        launcher.pretrain(select(launcher.cfg["pretrain_experiments"], args.experiment, args.index))
    elif args.action == "classifier":
        launcher.classifier(select(launcher.cfg["classifier_experiments"], args.experiment, args.index))
    elif args.action == "test":
        launcher.test(select(launcher.cfg["classifier_experiments"], args.experiment, args.index))
    elif args.action == "features":
        launcher.features(select(launcher.cfg["pretrain_experiments"], args.experiment, args.index))
    elif args.action == "summarize":
        launcher.summarize()


if __name__ == "__main__":
    main()
