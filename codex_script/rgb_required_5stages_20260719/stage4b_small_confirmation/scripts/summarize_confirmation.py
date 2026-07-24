#!/usr/bin/env python3
"""Create a validation-only Stage 4B confirmation report.

The report combines existing seed-1 runs with new Stage 4B runs and reports
fine-tuning stability, strict within-seed deltas, per-class recall, pretraining
loss diagnostics, and final prototype assignment geometry. It never discovers
or reads test results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NEW_CL_ROOT = "results/cl_rgb_req_s4b_confirm_20260724"
NEW_FT_ROOT = "results/ft_rgb_req_s4b_confirm_20260724/weights"
OUTPUT_ROOT = "results/ft_rgb_req_s4b_confirm_20260724/analysis"


def run(family, seed, experiment_id, source, finetune_root, pretrain_root):
    return {
        "family": family,
        "seed": seed,
        "experiment_id": experiment_id,
        "source": source,
        "finetune_root": finetune_root,
        "pretrain_root": pretrain_root,
    }


RUNS: List[Dict[str, object]] = [
    run("rel_r0", 1, "r0_sup_ft", "existing_stage4",
        "results/ft_rgb_req_s4_20260719/weights/r0_sup_ft",
        "results/cl_rgb_req_s4_20260719/r0_sup"),
    run("rel_r0", 2, "rel_r0_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r0_s2_ft", f"{NEW_CL_ROOT}/rel_r0_s2"),
    run("rel_r0", 3, "rel_r0_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r0_s3_ft", f"{NEW_CL_ROOT}/rel_r0_s3"),
    run("rel_r9", 1, "r9_diff_k3_s125_ft", "existing_stage4",
        "results/ft_rgb_req_s4_20260719/weights/r9_diff_k3_s125_ft",
        "results/cl_rgb_req_s4_20260719/r9_diff_k3_s125"),
    run("rel_r9", 2, "rel_r9_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r9_s2_ft", f"{NEW_CL_ROOT}/rel_r9_s2"),
    run("rel_r9", 3, "rel_r9_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r9_s3_ft", f"{NEW_CL_ROOT}/rel_r9_s3"),
    run("rel_r12", 1, "r12_diff_k3_s125_pm09_ft", "existing_stage4",
        "results/ft_rgb_req_s4_20260719/weights/r12_diff_k3_s125_pm09_ft",
        "results/cl_rgb_req_s4_20260719/r12_diff_k3_s125_pm09"),
    run("rel_r12", 2, "rel_r12_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r12_s2_ft", f"{NEW_CL_ROOT}/rel_r12_s2"),
    run("rel_r12", 3, "rel_r12_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_r12_s3_ft", f"{NEW_CL_ROOT}/rel_r12_s3"),
    run("rel_null", 1, "rel_null_s1_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_null_s1_ft", f"{NEW_CL_ROOT}/rel_null_s1"),
    run("rel_null", 2, "rel_null_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_null_s2_ft", f"{NEW_CL_ROOT}/rel_null_s2"),
    run("rel_null", 3, "rel_null_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/rel_null_s3_ft", f"{NEW_CL_ROOT}/rel_null_s3"),
    run("proto_p0", 1, "p0_sup_ft", "existing_stage1",
        "results/ft_rgb_req_s1_20260719/weights/p0_sup_ft",
        "results/cl_rgb_req_s1_20260719/p0_sup"),
    run("proto_p0", 2, "proto_p0_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p0_s2_ft", f"{NEW_CL_ROOT}/proto_p0_s2"),
    run("proto_p0", 3, "proto_p0_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p0_s3_ft", f"{NEW_CL_ROOT}/proto_p0_s3"),
    run("proto_null_p2", 1, "proto_null_p2_s1_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p2_s1_ft", f"{NEW_CL_ROOT}/proto_null_p2_s1"),
    run("proto_null_p2", 2, "proto_null_p2_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p2_s2_ft", f"{NEW_CL_ROOT}/proto_null_p2_s2"),
    run("proto_null_p2", 3, "proto_null_p2_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p2_s3_ft", f"{NEW_CL_ROOT}/proto_null_p2_s3"),
    run("proto_p4", 1, "p4_soft_p2_ft", "existing_stage1",
        "results/ft_rgb_req_s1_20260719/weights/p4_soft_p2_ft",
        "results/cl_rgb_req_s1_20260719/p4_soft_p2"),
    run("proto_p4", 2, "proto_p4_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p4_s2_ft", f"{NEW_CL_ROOT}/proto_p4_s2"),
    run("proto_p4", 3, "proto_p4_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p4_s3_ft", f"{NEW_CL_ROOT}/proto_p4_s3"),
    run("proto_null_p3", 1, "proto_null_p3_s1_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p3_s1_ft", f"{NEW_CL_ROOT}/proto_null_p3_s1"),
    run("proto_null_p3", 2, "proto_null_p3_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p3_s2_ft", f"{NEW_CL_ROOT}/proto_null_p3_s2"),
    run("proto_null_p3", 3, "proto_null_p3_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_null_p3_s3_ft", f"{NEW_CL_ROOT}/proto_null_p3_s3"),
    run("proto_p6", 1, "p6_all_p3_ft", "existing_stage1",
        "results/ft_rgb_req_s1_20260719/weights/p6_all_p3_ft",
        "results/cl_rgb_req_s1_20260719/p6_all_p3"),
    run("proto_p6", 2, "proto_p6_s2_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p6_s2_ft", f"{NEW_CL_ROOT}/proto_p6_s2"),
    run("proto_p6", 3, "proto_p6_s3_ft", "new_stage4b",
        f"{NEW_FT_ROOT}/proto_p6_s3_ft", f"{NEW_CL_ROOT}/proto_p6_s3"),
    run("rel_r7_optional", 1, "r7_diff_k10_ft", "existing_stage4",
        "results/ft_rgb_req_s4_20260719/weights/r7_diff_k10_ft",
        "results/cl_rgb_req_s4_20260719/r7_diff_k10"),
    run("rel_r7_optional", 2, "rel_r7_s2_ft", "new_stage4b_optional",
        f"{NEW_FT_ROOT}/rel_r7_s2_ft", f"{NEW_CL_ROOT}/rel_r7_s2"),
    run("rel_r7_optional", 3, "rel_r7_s3_ft", "new_stage4b_optional",
        f"{NEW_FT_ROOT}/rel_r7_s3_ft", f"{NEW_CL_ROOT}/rel_r7_s3"),
]

FAMILY_ORDER = [
    "rel_r0", "rel_null", "rel_r9", "rel_r12", "rel_r7_optional",
    "proto_p0", "proto_null_p2", "proto_p4", "proto_null_p3", "proto_p6",
]

BASELINE = {
    "rel_r0": "rel_r0",
    "rel_null": "rel_r0",
    "rel_r9": "rel_r0",
    "rel_r12": "rel_r0",
    "rel_r7_optional": "rel_r0",
    "proto_p0": "proto_p0",
    "proto_null_p2": "proto_p0",
    "proto_p4": "proto_p0",
    "proto_null_p3": "proto_p0",
    "proto_p6": "proto_p0",
}

# comparison id, treatment family, reference family, required
COMPARISONS: List[Tuple[str, str, str, bool]] = [
    ("rel_r9_minus_r0", "rel_r9", "rel_r0", True),
    ("rel_r9_minus_null", "rel_r9", "rel_null", True),
    ("rel_r12_minus_r0", "rel_r12", "rel_r0", True),
    ("rel_r12_minus_null", "rel_r12", "rel_null", True),
    ("rel_r7_minus_r0", "rel_r7_optional", "rel_r0", False),
    ("null_p2_minus_p0", "proto_null_p2", "proto_p0", True),
    ("p4_minus_p0", "proto_p4", "proto_p0", True),
    ("p4_minus_null_p2", "proto_p4", "proto_null_p2", True),
    ("null_p3_minus_p0", "proto_null_p3", "proto_p0", True),
    ("p6_minus_p0", "proto_p6", "proto_p0", True),
    ("p6_minus_null_p3", "proto_p6", "proto_null_p3", True),
    ("baseline_repeat_r0_minus_p0", "rel_r0", "proto_p0", True),
]

EPOCH_RE = re.compile(
    r"^\[(?P<epoch>\d+)\].*?\|\s+val loss:\s*(?P<loss>[-+0-9.eE]+),\s*"
    r"val_acc:\s*(?P<acc>[-+0-9.eE]+),\s*"
    r"val_balanced_acc:\s*(?P<ba>[-+0-9.eE]+),\s*"
    r"val_macro_f1:\s*(?P<f1>[-+0-9.eE]+)"
)


def safe_mean(values: Iterable[object]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    return statistics.mean(clean) if clean else None


def safe_stdev(values: Iterable[object]) -> Optional[float]:
    clean = [
        float(value) for value in values
        if value not in (None, "") and math.isfinite(float(value))
    ]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def pct(value: object) -> float:
    return 100.0 * float(value)


def format_number(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def find_single(root: Path, pattern: str) -> Tuple[str, Optional[Path]]:
    matches = sorted(root.rglob(pattern)) if root.is_dir() else []
    if not matches:
        return "missing", None
    if len(matches) > 1:
        return "multiple", None
    return "complete", matches[0]


def parse_finetune_log(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(raw_line)
        if match:
            records.append({
                "epoch0": int(match.group("epoch")),
                "val_loss": float(match.group("loss")),
                "val_acc": float(match.group("acc")),
                "val_ba": float(match.group("ba")),
                "val_f1": float(match.group("f1")),
                "per_class": {},
                "support": {},
            })
            continue
        stripped = raw_line.strip()
        if not records:
            continue
        if stripped.startswith("val_per_class_acc:"):
            records[-1]["per_class"] = json.loads(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("val_per_class_support:"):
            records[-1]["support"] = json.loads(stripped.split(":", 1)[1].strip())
    return records


def choose_best_record(records, summary):
    if not records:
        return None
    best_value = float(summary["best_val_balanced_acc"])
    best_epoch = int(summary["best_val_balanced_epoch"])
    candidates = [
        row for row in records
        if int(row["epoch0"]) in {best_epoch, best_epoch - 1}
        and abs(float(row["val_ba"]) - best_value) <= 5e-4
    ]
    if candidates:
        return min(candidates, key=lambda row: abs(int(row["epoch0"]) - best_epoch))
    return min(records, key=lambda row: abs(float(row["val_ba"]) - best_value))


def parse_debug_log(path: Path, args_path: Optional[Path]) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    nonfinite_records = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(row)
            if bool(row.get("nonfinite_check", {}).get("has_nonfinite", False)):
                nonfinite_records += 1
    args = {}
    if args_path and args_path.is_file():
        args = json.loads(args_path.read_text(encoding="utf-8"))
    proto_start = int(args.get("proto_loss_start_epoch", 50))
    rel_start = int(args.get("rel_loss_start_epoch", 50))
    proto_active = [row for row in records if int(row.get("epoch", 0)) > proto_start]
    rel_active = [row for row in records if int(row.get("epoch", 0)) > rel_start]
    last = max(records, key=lambda row: int(row.get("epoch", 0))) if records else {}
    last_proto = last.get("proto_batch_stats", {}) or {}
    feature_stats = last.get("feature_stats", {}) or {}
    return {
        "debug_records": len(records),
        "nonfinite_records": nonfinite_records,
        "proto_start_epoch": proto_start,
        "rel_start_epoch": rel_start,
        "mean_supcon_e151_200": safe_mean(
            row.get("loss_supcon") for row in records
            if 151 <= int(row.get("epoch", 0)) <= 200
        ),
        "mean_proto_loss_active": safe_mean(row.get("loss_proto") for row in proto_active),
        "mean_weighted_proto_active": safe_mean(
            row.get("weighted_proto_contrib") for row in proto_active
        ),
        "proto_nonzero_fraction_active": safe_mean(
            float(abs(float(row.get("loss_proto", 0.0))) > 1e-12) for row in proto_active
        ),
        "mean_rel_loss_active": safe_mean(row.get("loss_rel") for row in rel_active),
        "mean_weighted_rel_active": safe_mean(
            row.get("weighted_rel_contrib") for row in rel_active
        ),
        "rel_nonzero_fraction_active": safe_mean(
            float(abs(float(row.get("loss_rel", 0.0))) > 1e-12) for row in rel_active
        ),
        "final_feature_std": feature_stats.get("q_feature_std_mean"),
        "final_valid_proto_ratio": last_proto.get("valid_proto_assign_ratio"),
        "final_unique_proto_ids_batch": last_proto.get("num_unique_proto_ids_in_batch"),
    }


def checkpoint_geometry(path: Path) -> Dict[str, object]:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        return {"checkpoint_status": "torch_unavailable"}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    bank = checkpoint.get("prototype_bank")
    class_counts = checkpoint.get("class_num_prototypes")
    sample_to_proto = checkpoint.get("sample_to_proto")
    sample_to_class = checkpoint.get("sample_to_class")
    valid_mask = checkpoint.get("valid_sample_mask")
    if bank is None or class_counts is None or sample_to_proto is None or sample_to_class is None:
        return {"checkpoint_status": "no_prototype_state"}
    bank = bank.float()
    class_counts = class_counts.long()
    sample_to_proto = sample_to_proto.long()
    sample_to_class = sample_to_class.long()
    valid_mask = sample_to_proto >= 0 if valid_mask is None else valid_mask.bool() & (sample_to_proto >= 0)
    valid_samples = int(valid_mask.sum().item())
    dead = 0
    near_dead = 0
    cvs: List[float] = []
    entropies: List[float] = []
    active_vectors = []
    active_classes = []
    for class_id in range(int(class_counts.numel())):
        k = int(class_counts[class_id].item())
        if k <= 0:
            continue
        class_mask = valid_mask & (sample_to_class == class_id)
        assignments = sample_to_proto[class_mask]
        counts = torch.bincount(assignments.clamp_min(0), minlength=k)[:k].float()
        dead += int((counts == 0).sum().item())
        n_class = int(class_mask.sum().item())
        near_dead += int((counts <= max(2, int(math.ceil(0.05 * n_class)))).sum().item())
        mean_count = float(counts.mean().item())
        cvs.append(float(counts.std(unbiased=False).item()) / mean_count if mean_count else 0.0)
        probabilities = counts / counts.sum().clamp_min(1.0)
        entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
        entropies.append(entropy / math.log(k) if k > 1 else 1.0)
        active_vectors.append(functional.normalize(bank[class_id, :k], dim=1))
        active_classes.extend([class_id] * k)
    if not active_vectors:
        return {"checkpoint_status": "empty_prototype_state"}
    vectors = torch.cat(active_vectors, dim=0)
    class_tensor = torch.tensor(active_classes, dtype=torch.long)
    cosine = vectors @ vectors.t()
    eye = torch.eye(cosine.shape[0], dtype=torch.bool)
    same_mask = (class_tensor[:, None] == class_tensor[None, :]) & ~eye
    diff_mask = class_tensor[:, None] != class_tensor[None, :]
    same_values = cosine[same_mask]
    nearest_diff = cosine.masked_fill(~diff_mask, -float("inf")).max(dim=1).values
    return {
        "checkpoint_status": "complete",
        "valid_samples": valid_samples,
        "invalid_samples": int(valid_mask.numel() - valid_samples),
        "active_prototypes": int(vectors.shape[0]),
        "dead_prototypes": dead,
        "near_dead_prototypes": near_dead,
        "assignment_cv_mean": safe_mean(cvs),
        "assignment_entropy_norm_mean": safe_mean(entropies),
        "same_class_cos_mean": float(same_values.mean().item()) if same_values.numel() else None,
        "nearest_diff_cos_mean": float(nearest_diff.mean().item()),
    }


def build_finetune_rows(project):
    run_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []
    for spec in RUNS:
        root = project / str(spec["finetune_root"])
        status, summary_path = find_single(root, "summary.json")
        row: Dict[str, object] = {
            "family": spec["family"], "seed": spec["seed"],
            "experiment_id": spec["experiment_id"], "source": spec["source"],
            "status": status, "best_val_acc_pct": "",
            "best_val_balanced_acc_pct": "", "best_val_macro_f1_pct": "",
            "best_val_balanced_epoch": "", "final_val_balanced_acc_pct": "",
            "last10_val_balanced_acc_mean_pct": "",
            "last10_val_balanced_acc_std_pp": "",
            "summary_path": str(summary_path) if summary_path else "",
        }
        if summary_path:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            log_path = summary_path.parent / "train_logs.txt"
            records = parse_finetune_log(log_path) if log_path.is_file() else []
            last10 = sorted(records, key=lambda item: int(item["epoch0"]))[-10:]
            best_record = choose_best_record(records, summary)
            last10_mean = safe_mean(item["val_ba"] for item in last10)
            last10_std = safe_stdev(item["val_ba"] for item in last10)
            row.update({
                "best_val_acc_pct": pct(summary["best_val_acc"]),
                "best_val_balanced_acc_pct": pct(summary["best_val_balanced_acc"]),
                "best_val_macro_f1_pct": pct(summary["best_val_macro_f1"]),
                "best_val_balanced_epoch": int(summary["best_val_balanced_epoch"]),
                "final_val_balanced_acc_pct": pct(summary["final_val_balanced_acc"]),
                "last10_val_balanced_acc_mean_pct": 100.0 * last10_mean if last10_mean is not None else "",
                "last10_val_balanced_acc_std_pp": 100.0 * last10_std if last10_std is not None else "",
            })
            if best_record:
                support = best_record.get("support", {}) or {}
                for class_name, recall in (best_record.get("per_class", {}) or {}).items():
                    per_class_rows.append({
                        "family": spec["family"], "seed": spec["seed"],
                        "experiment_id": spec["experiment_id"],
                        "class_name": class_name, "support": support.get(class_name, ""),
                        "recall_pct": 100.0 * float(recall), "epoch0": best_record["epoch0"],
                    })
        run_rows.append(row)
    return run_rows, per_class_rows


def build_family_rows(run_rows):
    grouped = {
        family: [row for row in run_rows if row["family"] == family and row["status"] == "complete"]
        for family in FAMILY_ORDER
    }
    ba_means = {
        family: safe_mean(row["best_val_balanced_acc_pct"] for row in rows)
        for family, rows in grouped.items()
    }
    output = []
    for family in FAMILY_ORDER:
        rows = grouped[family]
        seeds = sorted(int(row["seed"]) for row in rows)
        ba_mean = ba_means[family]
        baseline_mean = ba_means.get(BASELINE[family])
        output.append({
            "family": family, "baseline_family": BASELINE[family],
            "completed_seeds": len(set(seeds)), "expected_seeds": 3,
            "seeds": ",".join(map(str, seeds)),
            "mean_val_acc_pct": safe_mean(row["best_val_acc_pct"] for row in rows),
            "mean_val_balanced_acc_pct": ba_mean,
            "std_val_balanced_acc_pp": safe_stdev(row["best_val_balanced_acc_pct"] for row in rows),
            "mean_val_macro_f1_pct": safe_mean(row["best_val_macro_f1_pct"] for row in rows),
            "mean_last10_val_balanced_acc_pct": safe_mean(
                row["last10_val_balanced_acc_mean_pct"] for row in rows
            ),
            "std_across_runs_last10_mean_pp": safe_stdev(
                row["last10_val_balanced_acc_mean_pct"] for row in rows
            ),
            "delta_ba_vs_baseline_pp": (
                ba_mean - baseline_mean
                if ba_mean is not None and baseline_mean is not None else None
            ),
            "status": "complete" if set(seeds) == {1, 2, 3} else "incomplete",
        })
    return output


def build_paired_rows(run_rows):
    lookup = {
        (str(row["family"]), int(row["seed"])): row
        for row in run_rows if row["status"] == "complete"
    }
    pair_rows = []
    pair_summary = []
    for comparison, treatment, reference, required in COMPARISONS:
        rows = []
        for seed in (1, 2, 3):
            left = lookup.get((treatment, seed))
            right = lookup.get((reference, seed))
            if not left or not right:
                continue
            late_left = left["last10_val_balanced_acc_mean_pct"]
            late_right = right["last10_val_balanced_acc_mean_pct"]
            pair = {
                "comparison": comparison, "treatment_family": treatment,
                "reference_family": reference, "required": required, "seed": seed,
                "delta_best_ba_pp": float(left["best_val_balanced_acc_pct"]) - float(right["best_val_balanced_acc_pct"]),
                "delta_last10_ba_pp": (
                    float(late_left) - float(late_right)
                    if late_left not in ("", None) and late_right not in ("", None) else None
                ),
                "delta_best_acc_pp": float(left["best_val_acc_pct"]) - float(right["best_val_acc_pct"]),
                "delta_best_macro_f1_pp": float(left["best_val_macro_f1_pct"]) - float(right["best_val_macro_f1_pct"]),
            }
            rows.append(pair)
            pair_rows.append(pair)
        best_deltas = [float(row["delta_best_ba_pp"]) for row in rows]
        late_deltas = [
            float(row["delta_last10_ba_pp"]) for row in rows
            if row["delta_last10_ba_pp"] is not None
        ]
        pair_summary.append({
            "comparison": comparison, "treatment_family": treatment,
            "reference_family": reference, "required": required,
            "completed_pairs": len(rows), "expected_pairs": 3,
            "mean_delta_best_ba_pp": safe_mean(best_deltas),
            "std_delta_best_ba_pp": safe_stdev(best_deltas),
            "positive_seed_count": sum(delta > 0 for delta in best_deltas),
            "mean_delta_last10_ba_pp": safe_mean(late_deltas),
            "status": "complete" if len(rows) == 3 else "incomplete",
        })
    return pair_rows, pair_summary


def build_per_class_pair_rows(per_class_rows):
    lookup = {
        (str(row["family"]), int(row["seed"]), str(row["class_name"])): row
        for row in per_class_rows
    }
    classes = sorted({str(row["class_name"]) for row in per_class_rows})
    output = []
    for comparison, treatment, reference, required in COMPARISONS:
        for class_name in classes:
            deltas = []
            supports = []
            for seed in (1, 2, 3):
                left = lookup.get((treatment, seed, class_name))
                right = lookup.get((reference, seed, class_name))
                if not left or not right:
                    continue
                deltas.append(float(left["recall_pct"]) - float(right["recall_pct"]))
                if left.get("support") not in (None, ""):
                    supports.append(int(left["support"]))
            if deltas:
                output.append({
                    "comparison": comparison, "treatment_family": treatment,
                    "reference_family": reference, "required": required,
                    "class_name": class_name, "support": supports[0] if supports else "",
                    "completed_pairs": len(deltas),
                    "mean_recall_delta_pp": safe_mean(deltas),
                    "std_recall_delta_pp": safe_stdev(deltas),
                    "positive_seed_count": sum(delta > 0 for delta in deltas),
                })
    return output


def build_pretrain_rows(project, skip_checkpoints):
    rows = []
    for spec in RUNS:
        root = project / str(spec["pretrain_root"])
        debug_path = root / "debug_train_log.jsonl"
        args_path = root / "args.json"
        checkpoint_path = root / "checkpoint_0200.pth"
        row = {
            "family": spec["family"], "seed": spec["seed"],
            "experiment_id": str(spec["experiment_id"]).removesuffix("_ft"),
            "source": spec["source"],
            "pretrain_status": "complete" if debug_path.is_file() else "missing",
            "debug_path": str(debug_path) if debug_path.is_file() else "",
            "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else "",
        }
        if debug_path.is_file():
            row.update(parse_debug_log(debug_path, args_path))
        if checkpoint_path.is_file() and not skip_checkpoints:
            row.update(checkpoint_geometry(checkpoint_path))
        elif skip_checkpoints:
            row["checkpoint_status"] = "skipped"
        else:
            row["checkpoint_status"] = "missing"
        rows.append(row)
    return rows


def render_markdown(family_rows, pair_summary, per_class_pair_rows, pretrain_rows):
    lines = [
        "# Stage 4B 完整确认实验：validation 汇总", "",
        "本报告只读取对比预训练和验证集微调结果，不发现、不读取测试集结果。", "",
        "## 1. 三 seed family 汇总", "",
        "| Family | Seeds | Best BA mean (%) | BA std (pp) | Last-10 BA mean (%) | Accuracy (%) | Macro-F1 (%) | Delta vs base (pp) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in family_rows:
        lines.append(
            "| {family} | {done}/{expected} | {ba} | {std} | {late} | {acc} | {f1} | {delta} | {status} |".format(
                family=row["family"], done=row["completed_seeds"],
                expected=row["expected_seeds"], ba=format_number(row["mean_val_balanced_acc_pct"]),
                std=format_number(row["std_val_balanced_acc_pp"]),
                late=format_number(row["mean_last10_val_balanced_acc_pct"]),
                acc=format_number(row["mean_val_acc_pct"]),
                f1=format_number(row["mean_val_macro_f1_pct"]),
                delta=format_number(row["delta_ba_vs_baseline_pp"]), status=row["status"],
            )
        )
    lines.extend([
        "", "## 2. 逐 seed 配对比较", "",
        "| Comparison | Pairs | Mean best-BA delta (pp) | Delta std | Positive seeds | Mean last-10 delta (pp) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in pair_summary:
        lines.append(
            "| {name} | {done}/{expected} | {mean} | {std} | {wins}/{done} | {late} | {status} |".format(
                name=row["comparison"], done=row["completed_pairs"],
                expected=row["expected_pairs"], mean=format_number(row["mean_delta_best_ba_pp"]),
                std=format_number(row["std_delta_best_ba_pp"]),
                wins=row["positive_seed_count"], late=format_number(row["mean_delta_last10_ba_pp"]),
                status=row["status"],
            )
        )
    lines.extend([
        "", "## 3. 关键逐类别变化", "",
        "每个 comparison 显示绝对平均变化最大的 5 个类别。小 support 类必须与 Accuracy、Macro-F1 和 last-10 BA 一起判断。", "",
        "| Comparison | Class | Support | Mean recall delta (pp) | Positive seeds |",
        "|---|---|---:|---:|---:|",
    ])
    important = {
        "rel_r9_minus_null", "p4_minus_null_p2",
        "p6_minus_null_p3", "baseline_repeat_r0_minus_p0",
    }
    for comparison in sorted(important):
        candidates = [row for row in per_class_pair_rows if row["comparison"] == comparison]
        candidates.sort(key=lambda row: abs(float(row["mean_recall_delta_pp"])), reverse=True)
        for row in candidates[:5]:
            lines.append(
                "| {comparison} | {class_name} | {support} | {delta} | {wins}/{pairs} |".format(
                    comparison=comparison, class_name=row["class_name"], support=row["support"],
                    delta=format_number(row["mean_recall_delta_pp"]),
                    wins=row["positive_seed_count"], pairs=row["completed_pairs"],
                )
            )
    lines.extend([
        "", "## 4. 对比预训练诊断", "",
        "| Family | Seed | SupLoss e151-200 | ProtoLoss active | RelLoss active | Proto nonzero | Rel nonzero | Valid proto | Dead | Near-dead | H(assign) | Same-class cos | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in pretrain_rows:
        lines.append(
            "| {family} | {seed} | {sup} | {proto} | {rel} | {pnz} | {rnz} | {valid} | {dead} | {near} | {entropy} | {samecos} | {status} |".format(
                family=row["family"], seed=row["seed"],
                sup=format_number(row.get("mean_supcon_e151_200"), 6),
                proto=format_number(row.get("mean_proto_loss_active"), 6),
                rel=format_number(row.get("mean_rel_loss_active"), 8),
                pnz=format_number(row.get("proto_nonzero_fraction_active"), 3),
                rnz=format_number(row.get("rel_nonzero_fraction_active"), 3),
                valid=format_number(row.get("final_valid_proto_ratio"), 3),
                dead=format_number(row.get("dead_prototypes"), 0),
                near=format_number(row.get("near_dead_prototypes"), 0),
                entropy=format_number(row.get("assignment_entropy_norm_mean"), 4),
                samecos=format_number(row.get("same_class_cos_mean"), 4),
                status=row.get("pretrain_status", "missing"),
            )
        )
    lines.extend([
        "", "## 5. 判定门槛", "", "### RelLoss", "",
        "- `rel_r9_minus_r0` 和 `rel_r9_minus_null` 都必须完成 3/3 seed。",
        "- R9 应在至少 2/3 seed 为正，三 seed平均 best BA 与 last-10 BA 均高于 Null-rel。",
        "- 若 R9≈Null-rel>R0，收益主要来自 prototype refresh/计算路径，不能归因于 relation 梯度。",
        "- R12 若仍有高 BA、但 RelLoss 非零率和量级接近零，应视为训练轨迹证据而非有效 RelLoss 设置。",
        "", "### ProtoLoss", "",
        "- `p4_minus_null_p2` 与 `p6_minus_null_p3` 是判断 ProtoLoss 梯度贡献的主要比较。",
        "- `null_p2_minus_p0` 与 `null_p3_minus_p0` 测量 prototype state/refresh 路径本身的贡献。",
        "- P4/P6 应在至少 2/3 seed 高于各自 Null-proto，且 mean best BA、mean last-10 BA 同向改善。",
        "- 若 P4≈Null-P2>P0 或 P6≈Null-P3>P0，不能把提升归因于 ProtoLoss。",
        "", "### 稳定性", "",
        "- `baseline_repeat_r0_minus_p0` 是同配置技术重复审计；若其波动与候选增益同量级，应继续增加重复或先修复确定性。",
        "- 逐类提升不能只由 support 很小的 `close` 等类别驱动。",
        "- 所有必做 family 和 comparison 完成前，不选择 Stage 5 配置，不运行测试集。",
        "", "## 6. 输出文件", "",
        "- `confirmation_validation_runs.csv`：每个微调运行的最佳、最终和末 10 epoch 指标；",
        "- `confirmation_family_summary.csv`：每个 family 的三 seed 汇总；",
        "- `confirmation_paired_deltas.csv`：每个 seed 的严格配对差；",
        "- `confirmation_pair_summary.csv`：配对差的均值、标准差和正向 seed 数；",
        "- `confirmation_per_class_runs.csv`：最佳 BA epoch 的逐类 recall；",
        "- `confirmation_per_class_pair_summary.csv`：逐类配对变化；",
        "- `confirmation_pretrain_diagnostics.csv`：预训练 loss、非零比例、assignment 和 prototype 几何。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=os.environ.get(
            "PROJECT_ROOT",
            "/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623",
        ),
    )
    parser.add_argument(
        "--skip-checkpoints", action="store_true",
        help="skip checkpoint geometry for a faster summary",
    )
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    output = project / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)

    run_rows, per_class_rows = build_finetune_rows(project)
    family_rows = build_family_rows(run_rows)
    paired_rows, pair_summary = build_paired_rows(run_rows)
    per_class_pair_rows = build_per_class_pair_rows(per_class_rows)
    pretrain_rows = build_pretrain_rows(project, args.skip_checkpoints)

    write_csv(output / "confirmation_validation_runs.csv", run_rows, [
        "family", "seed", "experiment_id", "source", "status",
        "best_val_acc_pct", "best_val_balanced_acc_pct", "best_val_macro_f1_pct",
        "best_val_balanced_epoch", "final_val_balanced_acc_pct",
        "last10_val_balanced_acc_mean_pct", "last10_val_balanced_acc_std_pp",
        "summary_path",
    ])
    write_csv(output / "confirmation_family_summary.csv", family_rows, [
        "family", "baseline_family", "completed_seeds", "expected_seeds", "seeds",
        "mean_val_acc_pct", "mean_val_balanced_acc_pct", "std_val_balanced_acc_pp",
        "mean_val_macro_f1_pct", "mean_last10_val_balanced_acc_pct",
        "std_across_runs_last10_mean_pp", "delta_ba_vs_baseline_pp", "status",
    ])
    write_csv(output / "confirmation_paired_deltas.csv", paired_rows, [
        "comparison", "treatment_family", "reference_family", "required", "seed",
        "delta_best_ba_pp", "delta_last10_ba_pp", "delta_best_acc_pp",
        "delta_best_macro_f1_pp",
    ])
    write_csv(output / "confirmation_pair_summary.csv", pair_summary, [
        "comparison", "treatment_family", "reference_family", "required",
        "completed_pairs", "expected_pairs", "mean_delta_best_ba_pp",
        "std_delta_best_ba_pp", "positive_seed_count",
        "mean_delta_last10_ba_pp", "status",
    ])
    write_csv(output / "confirmation_per_class_runs.csv", per_class_rows, [
        "family", "seed", "experiment_id", "class_name", "support", "recall_pct", "epoch0",
    ])
    write_csv(output / "confirmation_per_class_pair_summary.csv", per_class_pair_rows, [
        "comparison", "treatment_family", "reference_family", "required", "class_name",
        "support", "completed_pairs", "mean_recall_delta_pp", "std_recall_delta_pp",
        "positive_seed_count",
    ])
    pretrain_columns = [
        "family", "seed", "experiment_id", "source", "pretrain_status", "debug_records",
        "nonfinite_records", "proto_start_epoch", "rel_start_epoch",
        "mean_supcon_e151_200", "mean_proto_loss_active", "mean_weighted_proto_active",
        "proto_nonzero_fraction_active", "mean_rel_loss_active", "mean_weighted_rel_active",
        "rel_nonzero_fraction_active", "final_feature_std", "final_valid_proto_ratio",
        "final_unique_proto_ids_batch", "checkpoint_status", "valid_samples",
        "invalid_samples", "active_prototypes", "dead_prototypes", "near_dead_prototypes",
        "assignment_cv_mean", "assignment_entropy_norm_mean", "same_class_cos_mean",
        "nearest_diff_cos_mean", "debug_path", "checkpoint_path",
    ]
    write_csv(
        output / "confirmation_pretrain_diagnostics.csv",
        [{column: row.get(column, "") for column in pretrain_columns} for row in pretrain_rows],
        pretrain_columns,
    )
    (output / "confirmation_summary.md").write_text(
        render_markdown(family_rows, pair_summary, per_class_pair_rows, pretrain_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "validation_runs": len(run_rows),
        "complete_validation_runs": sum(row["status"] == "complete" for row in run_rows),
        "families": len(family_rows),
        "complete_families": sum(row["status"] == "complete" for row in family_rows),
        "comparisons": len(pair_summary),
        "complete_comparisons": sum(row["status"] == "complete" for row in pair_summary),
        "output": str(output),
        "test_data_read": False,
    }, indent=2, ensure_ascii=False))
    print(f"Markdown: {output / 'confirmation_summary.md'}")


if __name__ == "__main__":
    main()
