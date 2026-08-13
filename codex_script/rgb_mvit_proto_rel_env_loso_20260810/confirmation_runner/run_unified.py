#!/usr/bin/env python3
"""Flexible runner for 15/17-class confirmation, augmentation, and final-refit experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from common.config import append  # noqa: E402
from run import Runner as ParentRunner  # noqa: E402


REGISTRY_PATH = ROOT / "config" / "unified_experiment_registry.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not result:
        raise ValueError("Name contains no safe characters")
    return result


def base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        platform=args.platform, project_root=args.project_root, dataset_root=args.dataset_root,
        python_bin=args.python_bin, dry_run=getattr(args, "dry_run", False),
        validate_command=getattr(args, "validate_command", False),
    )


def expand(registry: dict, value: str, group_key: str, item_key: str) -> list[str]:
    output: list[str] = []
    for token in split_csv(value):
        for item in registry.get(group_key, {}).get(token, [token]):
            if item not in registry[item_key]:
                raise ValueError(f"Unknown selection {item!r} for {item_key}")
            if item not in output:
                output.append(item)
    if not output:
        raise ValueError(f"No values selected for {item_key}")
    return output


def locate_source(parent: ParentRunner, stage: str, source_id: str) -> dict:
    matches = [dict(row) for row in parent.rows(stage) if row["id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve {stage}/{source_id}: {len(matches)} matches")
    return matches[0]


def resolve_loss(parent: ParentRunner, registry: dict, loss_id: str, seed: int) -> dict:
    spec = registry["loss_configs"][loss_id]
    row = locate_source(parent, spec["source_stage"], spec["source_id"])
    row.update(spec.get("overrides", {}))
    row.update({
        "id": loss_id, "seed": int(seed), "full_id": spec["full_id"], "role": spec["role"],
        "source_stage": spec["source_stage"], "source_id": spec["source_id"],
    })
    row.setdefault("pretrain", True)
    return row


def normalized_loss(row: dict) -> dict:
    defaults = {
        "ablation_mode": "contrastive_only", "proto_positive_mode": "all", "num_prototypes": 1,
        "lambda_proto": 0.0, "lambda_rel": 0.0, "proto_start": 50, "rel_start": 200,
        "preview_ema_momentum": 0.5, "rel_same_weight": 0.0, "rel_diff_weight": 1.0,
        "rel_topk_diff_classes": 3, "rel_lambda_schedule": "constant", "warmup_epochs": None,
    }
    return {key: row.get(key, value) for key, value in defaults.items()}


class UnifiedRunner(ParentRunner):
    def __init__(self, args: argparse.Namespace, registry: dict):
        super().__init__(args)
        self.registry = registry
        self.results = self.project / registry["output_rel"]
        self.splits = self.project / registry["split_output_rel"]
        self.detailed_entry = ROOT / "common" / "detailed_classifier_entry.py"
        self.context: dict | None = None

    def configure(self, row: dict) -> None:
        self.context = row
        task = self.registry["tasks"][row["task"]]
        aug = self.registry["augmentations"][row["augmentation_id"]]
        train_cfg = self.registry["training"]
        self.plan["model"]["num_classes"] = int(task["num_classes"])
        self.plan["manifests"]["label_map"] = task["label_map"]
        self.plan["pretrain_augmentation"] = dict(aug)
        self.plan["pretrain_common"]["epochs"] = int(train_cfg["pretrain_epochs"])
        self.plan["pretrain_common"]["save_interval"] = int(train_cfg["pretrain_save_interval"])
        self.plan["pretrain_common"]["prototype_diagnostic_interval"] = int(train_cfg["prototype_diagnostic_interval"])
        sampling = self.registry["sampling_profiles"][row["sampling_id"]]
        row["pretrain_sampler"] = sampling["pretrain_sampler"]
        for key in ("balanced_classes_per_batch", "balanced_samples_per_class", "weighted_sampler_mode"):
            if key in sampling:
                row[key] = sampling[key]

    def split_dir(self, row: dict) -> Path:
        name = ("holdout_" if row["protocol"] == "subject_dev" else "test_") + row["subject"]
        return self.splits / row["task"] / row["protocol"] / name

    def manifests_for(self, row: dict) -> tuple[Path, Path | None, Path | None]:
        root = self.split_dir(row)
        train = root / "train.jsonl"
        val = root / "val.jsonl" if row["protocol"] == "subject_dev" else None
        test = root / "test.jsonl" if row["protocol"] == "final_refit" else None
        return train, val, test

    def manifests(self, fold: str) -> tuple[Path, Path, Path]:
        if self.context is None:
            raise RuntimeError("Runner context is not configured")
        train, val, test = self.manifests_for(self.context)
        return train, val or Path("<no-validation>"), test or Path("<no-test>")

    @staticmethod
    def seed_tag(row: dict) -> str:
        return f"s{int(row['seed'])}"

    def base_output(self, kind: str, row: dict) -> Path:
        prefix = "v" if row["protocol"] == "subject_dev" else "t"
        return (self.results / kind / row["task"] / row["protocol"] /
                f"{prefix}{row['subject']}" / row["id"] / row["augmentation_id"] /
                row["sampling_id"] / self.seed_tag(row))

    def pretrain_dir(self, stage: str, fold: str, row: dict) -> Path:
        return self.base_output("pretrain", row)

    def classifier_dir(self, stage: str, fold: str, row: dict) -> Path:
        return self.base_output("finetune", row)

    def evaluation_dir(self, phase: str, row: dict) -> Path:
        return self.base_output("dev_eval" if phase == "evaluate" else "test", row)

    def checkpoint(self, stage: str, fold: str, row: dict) -> Path:
        epochs = int(self.registry["training"]["pretrain_epochs"])
        return self.pretrain_dir(stage, fold, row) / f"checkpoint_{epochs:04d}.pth"

    def classifier_epoch_checkpoint(self, row: dict) -> Path:
        epoch = int(self.registry["training"]["test_checkpoint_epoch"])
        root = self.classifier_dir("unified", row["subject"], row)
        found = list(root.rglob(f"epoch_{epoch:03d}.pth"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one epoch_{epoch:03d}.pth, found {len(found)} under {root}")
        return found[0]

    def finetune(self, stage: str, fold: str, row: dict) -> None:
        train_manifest, val_manifest, _ = self.manifests_for(row)
        out = self.classifier_dir(stage, fold, row)
        epoch = int(self.registry["training"]["test_checkpoint_epoch"])
        if not self.args.dry_run and len(list(out.rglob(f"epoch_{epoch:03d}.pth"))) == 1:
            print(f"[Skip] completed epoch-{epoch} classifier: {out}")
            return
        model = self.plan["model"]
        common = dict(self.plan["finetune_common"])
        training = self.registry["training"]
        common.update({"epochs": training["finetune_epochs"], "save_period": training["finetune_save_period"], "schedule": training["finetune_lr_milestones"]})
        command: list[object] = [
            self.python, "-u", self.classifier_entry,
            "--repair-source", self.classifier_source, "--repair-src-root", self.classifier_src,
            "--repair-representation", "rgb", "--repair-temporal-mode", "current",
            "--repair-backbone-init", "kinetics400", "--repair-finetune-policy", "full",
        ]
        values = {
            "--run_mode": "train", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--dataset_root": self.dataset, "--label_map_json": self.dataset / self.registry["tasks"][row["task"]]["label_map"],
            "--train_manifest": train_manifest, "--tier_mode": model["tier_mode"], "--n_frames": model["n_frames"],
            "--use_modality": "rgb", "--num_classes": self.registry["tasks"][row["task"]]["num_classes"],
            "--backbone": model["backbone"], "--model_depth": 18, "--rgb_camera_id": model["rgb_camera_id"],
            "--rgb_size": model["image_size"], "--rrc_scale_min": common["rrc_scale"][0],
            "--rrc_scale_max": common["rrc_scale"][1], "--rrc_ratio_min": common["rrc_ratio"][0],
            "--rrc_ratio_max": common["rrc_ratio"][1], "--rgb_hflip_p": common["hflip_p"],
            "--rgb_vflip_p": common["vflip_p"], "--rgb_jitter_p": common["jitter_p"],
            "--rgb_gray_p": common["gray_p"], "--rgb_blur_p": common["blur_p"],
            "--epochs": common["epochs"], "--batch_size": common["batch_size"],
            "--num_workers_train": common["num_workers_train"], "--num_workers_val": common["num_workers_val"],
            "--optimizer": common["optimizer"], "--learning_rate": common["head_learning_rate"],
            "--weight_decay": common["weight_decay"], "--seed": row["seed"], "--finetune_mode": "full",
            "--save_period": common["save_period"], "--best_after_epoch": 0,
            "--backbone_learning_rate": common["backbone_learning_rate"], "--head_learning_rate": common["head_learning_rate"],
        }
        if val_manifest is not None:
            values["--val_manifest"] = val_manifest
        for flag, value in values.items():
            append(command, flag, value)
        append(command, "--rgb_mean", model["rgb_mean"])
        append(command, "--rgb_std", model["rgb_std"])
        append(command, "--schedules", common["schedule"])
        command.extend(["--rgb_apply_spatial_aug", "--enable_amp", "--use_discriminative_lr"])
        if val_manifest is None:
            command.append("--disable_val")
        sampling = self.registry["sampling_profiles"][row["sampling_id"]]
        if sampling.get("finetune_weighted_sampler", False):
            command.append("--use_weighted_sampler")
            append(command, "--sampler_mode", sampling.get("finetune_sampler_mode", "sqrt_inv"))
        if row.get("pretrain", True):
            append(command, "--pretrained_weight_paths", self.checkpoint(stage, fold, row))
        self.run(command)

    def evaluate(self, phase: str, row: dict) -> None:
        _, val, test = self.manifests_for(row)
        if phase == "evaluate":
            if row["protocol"] != "subject_dev" or val is None:
                raise ValueError("evaluate is only valid for subject_dev")
            target = val
        else:
            if row["protocol"] != "final_refit" or test is None:
                raise ValueError("test is only valid for final_refit")
            target = test
        out = self.evaluation_dir(phase, row)
        checkpoint = self.classifier_epoch_checkpoint(row) if not self.args.dry_run else self.classifier_dir("unified", row["subject"], row) / "<run>" / "epoch_050.pth"
        model = self.plan["model"]
        command: list[object] = [
            self.python, "-u", self.detailed_entry,
            "--repair-source", self.classifier_source, "--repair-src-root", self.classifier_src,
            "--repair-representation", "rgb", "--repair-temporal-mode", "current",
            "--repair-backbone-init", "kinetics400", "--repair-finetune-policy", "full",
        ]
        values = {
            "--run_mode": "test", "--save_path": out, "--datamap_csv_path": out / "datamaps",
            "--test_results_csv": out / "test_results.csv", "--dataset_root": self.dataset,
            "--label_map_json": self.dataset / self.registry["tasks"][row["task"]]["label_map"],
            "--test_manifest": target, "--test_weight_paths": checkpoint, "--tier_mode": model["tier_mode"],
            "--n_frames": model["n_frames"], "--use_modality": "rgb",
            "--num_classes": self.registry["tasks"][row["task"]]["num_classes"],
            "--backbone": model["backbone"], "--model_depth": 18, "--rgb_camera_id": model["rgb_camera_id"],
            "--rgb_size": model["image_size"], "--batch_size": self.plan["finetune_common"]["batch_size"],
            "--num_workers_test": self.plan["finetune_common"]["num_workers_test"],
        }
        for flag, value in values.items():
            append(command, flag, value)
        append(command, "--rgb_mean", model["rgb_mean"]); append(command, "--rgb_std", model["rgb_std"])
        command.append("--enable_amp")
        self.run(command)
        if self.args.dry_run:
            return
        candidates = sorted(out.glob("*_per_sample_test.csv"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No per-sample test CSV under {out}")
        self.run([
            self.python, ROOT / "tools" / "enrich_predictions.py", "--input", candidates[-1],
            "--manifest", target, "--output", out / "predictions.csv", "--config", row["id"],
            "--full-id", row["full_id"], "--fold", row["subject"], "--seed", row["seed"],
            "--checkpoint", checkpoint,
        ])


def ensure_splits(runner: UnifiedRunner, rows: list[dict]) -> None:
    audit = runner.splits / "protocol_audit.json"
    if not audit.is_file():
        raise FileNotFoundError(f"Prepared splits are missing: {audit}. Run prepare first.")
    for row in rows:
        for path in runner.manifests_for(row):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)


def selection_from_args(args: argparse.Namespace, registry: dict) -> dict:
    if args.preset:
        if args.preset not in registry["presets"]:
            raise ValueError(f"Unknown preset: {args.preset}")
        preset = registry["presets"][args.preset]
        task = args.task or preset["task"]
        protocol = args.protocol or preset["protocol"]
        losses = split_csv(args.configs) if args.configs else list(preset["losses"])
        augmentations = split_csv(args.augmentations) if args.augmentations else list(preset["augmentations"])
        samplings = split_csv(args.samplings) if args.samplings else list(preset["samplings"])
        subjects = split_csv(args.subjects) if args.subjects else list(preset["subjects"])
        seeds = [int(value) for value in split_csv(args.seeds)] if args.seeds else list(preset["seeds"])
    else:
        task = args.task or registry["default_task"]
        protocol = args.protocol or registry["default_protocol"]
        losses = expand(registry, args.configs or "confirm15_min", "loss_groups", "loss_configs")
        augmentations = expand(registry, args.augmentations or registry["default_augmentation"], "augmentation_groups", "augmentations")
        samplings = expand(registry, args.samplings or registry["default_sampling"], "sampling_groups", "sampling_profiles")
        subjects = split_csv(args.subjects) or list(registry["protocols"][protocol]["allowed_subjects"])
        seeds = [int(value) for value in split_csv(args.seeds or "2,3")]
    if task not in registry["tasks"] or protocol not in registry["protocols"]:
        raise ValueError(f"Unknown task/protocol: {task}/{protocol}")
    for values, key in ((losses, "loss_configs"), (augmentations, "augmentations"), (samplings, "sampling_profiles")):
        unknown = [value for value in values if value not in registry[key]]
        if unknown:
            raise ValueError(f"Unknown {key}: {unknown}")
    allowed = set(registry["protocols"][protocol]["allowed_subjects"])
    if any(subject not in allowed for subject in subjects):
        raise ValueError(f"Subjects {subjects} are invalid for {protocol}; allowed={sorted(allowed)}")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    return {"task": task, "protocol": protocol, "losses": losses, "augmentations": augmentations, "samplings": samplings, "subjects": subjects, "seeds": seeds}


def build_manifest(args: argparse.Namespace) -> Path:
    registry = read_json(REGISTRY_PATH)
    runner = UnifiedRunner(base_args(args), registry)
    selected = selection_from_args(args, registry)
    rows: list[dict] = []; signatures: set[tuple] = set()
    for subject in selected["subjects"]:
        for seed in selected["seeds"]:
            for loss_id in selected["losses"]:
                for aug_id in selected["augmentations"]:
                    for sampling_id in selected["samplings"]:
                        loss = resolve_loss(runner, registry, loss_id, seed)
                        effective_aug = aug_id if loss.get("pretrain", True) else "a0"
                        sampling = registry["sampling_profiles"][sampling_id]
                        effective_sampling = sampling_id if loss.get("pretrain", True) else ("weighted_ft" if sampling.get("finetune_weighted_sampler") else "natural")
                        signature = (subject, seed, loss_id, effective_aug, effective_sampling)
                        if signature in signatures:
                            continue
                        signatures.add(signature)
                        rows.append({
                            "index": len(rows),
                            "run_id": f"{selected['task']}_{selected['protocol']}_{subject}_{loss_id}_{effective_aug}_{effective_sampling}_s{seed}",
                            "task": selected["task"], "protocol": selected["protocol"], "subject": subject,
                            "seed": seed, "config_id": loss_id, "augmentation_id": effective_aug,
                            "sampling_id": effective_sampling, "full_id": loss["full_id"],
                            "pretrain": bool(loss.get("pretrain", True)),
                        })
    if not rows:
        raise ValueError("Selection produced no runs")
    ensure_splits(runner, rows)
    name = safe_name(args.name or datetime.now().strftime("unified_%Y%m%d_%H%M%S"))
    manifest = Path(args.manifest_path).resolve() if args.manifest_path else runner.results / "manifests" / f"{name}.csv"
    if manifest.exists() and not args.overwrite_manifest:
        raise FileExistsError(f"Manifest already exists: {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(manifest.with_suffix(".meta.json"), {
        "schema_version": 2, "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "registry_sha256": sha256(REGISTRY_PATH), "selection": selected, "num_runs": len(rows),
    })
    print(manifest)
    return manifest


def load_row(path: Path, index: int) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} outside 0..{len(rows)-1}")
    row = rows[index]; row["index"] = int(row["index"]); row["seed"] = int(row["seed"])
    row["pretrain"] = str(row["pretrain"]).lower() in {"1", "true", "yes"}
    return row


def manifest_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def resolved_row(runner: UnifiedRunner, manifest_row: dict) -> dict:
    row = resolve_loss(runner, runner.registry, manifest_row["config_id"], manifest_row["seed"])
    row.update({key: manifest_row[key] for key in ("task", "protocol", "subject", "augmentation_id", "sampling_id")})
    runner.configure(row)
    return row


def snapshot(runner: UnifiedRunner, row: dict, manifest_row: dict) -> None:
    train, val, test = runner.manifests_for(row)
    files = {"train": train, "validation": val, "test": test}
    write_json(runner.base_output("run_meta", row) / "resolved_config.json", {
        "run": manifest_row, "resolved_loss": row,
        "task": runner.registry["tasks"][row["task"]], "protocol": runner.registry["protocols"][row["protocol"]],
        "augmentation": runner.registry["augmentations"][row["augmentation_id"]],
        "sampling": runner.registry["sampling_profiles"][row["sampling_id"]], "training": runner.registry["training"],
        "manifests": {name: ({"path": str(path), "sha256": sha256(path)} if path else None) for name, path in files.items()},
    })


def completed(runner: UnifiedRunner, phase: str, row: dict) -> bool:
    if phase == "pretrain":
        return (not row.get("pretrain", True)) or runner.checkpoint("unified", row["subject"], row).is_file()
    if phase == "proto-env":
        if not row.get("pretrain", True) or row.get("ablation_mode") == "contrastive_only": return True
        target = runner.base_output("prototype_environment", row)
        return target.is_dir() and any(target.iterdir())
    if phase == "finetune":
        epoch = runner.registry["training"]["test_checkpoint_epoch"]
        return len(list(runner.classifier_dir("unified", row["subject"], row).rglob(f"epoch_{epoch:03d}.pth"))) == 1
    if phase in {"evaluate", "test"}:
        target = runner.evaluation_dir(phase, row)
        return (target / "test_results.csv").is_file() and (target / "predictions.csv").is_file()
    return False


def proto_environment(runner: UnifiedRunner, row: dict) -> None:
    if not row.get("pretrain", True) or row.get("ablation_mode") == "contrastive_only":
        print(f"[Skip] no prototype state expected for {row['id']}"); return
    train, _, _ = runner.manifests_for(row)
    runner.run([
        runner.python, PARENT / "tools" / "analyze_environment_prototypes.py", "--manifest", train,
        "--diagnostic-dir", runner.pretrain_dir("unified", row["subject"], row) / "prototype_diagnostics",
        "--output", runner.base_output("prototype_environment", row),
    ])


def run_task(args: argparse.Namespace) -> None:
    manifest_row = load_row(Path(args.manifest).resolve(), args.index)
    registry = read_json(REGISTRY_PATH); runner = UnifiedRunner(base_args(args), registry)
    row = resolved_row(runner, manifest_row); protocol = registry["protocols"][row["protocol"]]
    if args.phase == "test" and not protocol["allows_test_phase"]:
        raise ValueError("subject_dev cannot run test; use evaluate")
    if args.phase == "evaluate" and not protocol["has_validation"]:
        raise ValueError("final_refit has no validation subject; use test")
    if not args.dry_run: snapshot(runner, row, manifest_row)
    if args.resume and completed(runner, args.phase, row):
        print(f"[Skip complete] {args.phase}: {manifest_row['run_id']}"); return
    if args.phase == "pretrain": runner.pretrain("unified", row["subject"], row); proto_environment(runner, row)
    elif args.phase == "proto-env": proto_environment(runner, row)
    elif args.phase == "finetune": runner.finetune("unified", row["subject"], row)
    elif args.phase in {"evaluate", "test"}: runner.evaluate(args.phase, row)


def run_all(args: argparse.Namespace) -> None:
    phases = split_csv(args.phases); valid = {"pretrain", "proto-env", "finetune", "evaluate", "test"}
    if not phases or any(phase not in valid for phase in phases): raise ValueError(f"Invalid phases {phases}")
    for phase in phases:
        for index in range(manifest_count(Path(args.manifest))):
            current = argparse.Namespace(**vars(args)); current.phase = phase; current.index = index; run_task(current)


def prepare(args: argparse.Namespace) -> None:
    registry = read_json(REGISTRY_PATH); runner = UnifiedRunner(base_args(args), registry)
    command: list[object] = [runner.python, ROOT / "tools" / "prepare_unified_protocols.py", "--dataset-root", runner.dataset, "--output", runner.splits, "--reserved-final-subject", "N"]
    if args.overwrite: command.append("--overwrite")
    runner.run(command)


def validate(args: argparse.Namespace) -> None:
    registry = read_json(REGISTRY_PATH); runner = UnifiedRunner(base_args(args), registry); issues: list[str] = []
    resolved = {name: resolve_loss(runner, registry, name, 2) for name in registry["loss_configs"]}
    for pair in registry["pairs"]:
        left, right = normalized_loss(resolved[pair["control"]]), normalized_loss(resolved[pair["active"]])
        differences = sorted(key for key in left if left[key] != right[key])
        if differences != sorted(pair["expected_parameter_differences"]): issues.append(f"{pair['pair_id']}: {differences}")
    if int(runner.plan["pretrain_common"]["batch_size"]) != 32: issues.append("pretraining batch size is not 32")
    for name, profile in registry["sampling_profiles"].items():
        if profile["pretrain_sampler"] == "balanced_batch":
            product = int(profile["balanced_classes_per_batch"]) * int(profile["balanced_samples_per_class"])
            if product != 32: issues.append(f"{name}: balanced batch product={product}")
    if issues: raise RuntimeError("Validation failed:\n" + "\n".join(issues))
    print(json.dumps({
        "status": "OK", "loss_configs": len(resolved), "augmentations": len(registry["augmentations"]),
        "sampling_profiles": len(registry["sampling_profiles"]), "strict_pairs": len(registry["pairs"]),
        "finetune_epochs": registry["training"]["finetune_epochs"], "test_checkpoint_epoch": registry["training"]["test_checkpoint_epoch"],
        "splits_ready": (runner.splits / "protocol_audit.json").is_file(), "results": str(runner.results),
    }, indent=2, ensure_ascii=False))


def add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--platform", choices=["auto", "windows", "hpc"], default="auto")
    parser.add_argument("--project-root"); parser.add_argument("--dataset-root"); parser.add_argument("--python-bin")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--validate-command", action="store_true")


def add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset"); parser.add_argument("--task", choices=["t15", "t17"])
    parser.add_argument("--protocol", choices=["subject_dev", "final_refit"])
    parser.add_argument("--configs"); parser.add_argument("--augmentations"); parser.add_argument("--samplings")
    parser.add_argument("--seeds"); parser.add_argument("--subjects"); parser.add_argument("--name")
    parser.add_argument("--manifest-path"); parser.add_argument("--overwrite-manifest", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("list-configs"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("validate"); add_runtime(p)
    p = sub.add_parser("prepare"); p.add_argument("--overwrite", action="store_true"); add_runtime(p)
    p = sub.add_parser("build-manifest"); add_selection(p); add_runtime(p)
    p = sub.add_parser("count-manifest"); p.add_argument("--manifest", required=True)
    p = sub.add_parser("run-task"); p.add_argument("--manifest", required=True); p.add_argument("--index", required=True, type=int)
    p.add_argument("--phase", required=True, choices=["pretrain", "proto-env", "finetune", "evaluate", "test"])
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True); add_runtime(p)
    p = sub.add_parser("run-all"); p.add_argument("--manifest", required=True); p.add_argument("--phases", required=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True); add_runtime(p)
    p = sub.add_parser("run-local"); add_selection(p); p.add_argument("--phases", required=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True); add_runtime(p)
    args = parser.parse_args()
    if args.action == "list-configs":
        registry = read_json(REGISTRY_PATH)
        if args.json: print(json.dumps(registry, indent=2, ensure_ascii=False))
        else:
            for section in ("presets", "loss_groups", "augmentation_groups", "sampling_groups"):
                print(section + ":")
                for name, value in registry[section].items(): print(f"  {name}: {value}")
    elif args.action == "validate": validate(args)
    elif args.action == "prepare": prepare(args)
    elif args.action == "build-manifest": build_manifest(args)
    elif args.action == "count-manifest": print(manifest_count(Path(args.manifest)))
    elif args.action == "run-task": run_task(args)
    elif args.action == "run-all": run_all(args)
    elif args.action == "run-local":
        manifest = build_manifest(args); args.manifest = str(manifest); run_all(args)


if __name__ == "__main__":
    main()
