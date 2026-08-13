#!/usr/bin/env python3
"""Reproducible extraction of the completed MR-fold screening evidence."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def experiment_from(path: Path, marker: str) -> tuple[str, str]:
    parts = path.parts
    positions = [i for i, value in enumerate(parts) if value == marker]
    if not positions:
        raise ValueError(path)
    i = positions[-1]
    if marker == "classifier":
        return parts[i + 2], parts[i + 3]
    return parts[i + 2], parts[i + 3]


def classifier_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("classifier/fold_MR/*/*/*/summary.json"):
        stage, exp = experiment_from(path, "classifier")
        obj = read_json(path)
        rows.append({
            "stage": stage, "experiment": exp,
            "best_val_balanced_acc": obj.get("best_val_balanced_acc"),
            "best_val_macro_f1": obj.get("best_val_macro_f1"),
            "best_val_acc": obj.get("best_val_acc"),
            "best_epoch": obj.get("best_val_balanced_epoch"),
            "final_val_balanced_acc": obj.get("final_val_balanced_acc"),
            "final_val_macro_f1": obj.get("final_val_macro_f1"),
            "final_val_acc": obj.get("final_val_acc"),
            "final_train_balanced_acc": obj.get("final_train_balanced_acc"),
            "final_train_acc": obj.get("final_train_acc"),
            "best_to_final_ba_drop": (
                obj["best_val_balanced_acc"] - obj["final_val_balanced_acc"]
                if obj.get("best_val_balanced_acc") is not None and obj.get("final_val_balanced_acc") is not None else None
            ),
            "summary_path": str(path),
        })
    baseline = next(row for row in rows if row["experiment"] == "s0_sup")
    direct = next(row for row in rows if row["experiment"] == "d0_k400_direct")
    for row in rows:
        row["delta_ba_vs_sup"] = row["best_val_balanced_acc"] - baseline["best_val_balanced_acc"]
        row["delta_f1_vs_sup"] = row["best_val_macro_f1"] - baseline["best_val_macro_f1"]
        row["delta_ba_vs_direct"] = row["best_val_balanced_acc"] - direct["best_val_balanced_acc"]
    return sorted(rows, key=lambda row: row["best_val_balanced_acc"], reverse=True)


def parse_epoch_per_class(summary_path: Path, epoch: int) -> dict[str, float]:
    log = summary_path.with_name("train_logs.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    marker = re.compile(rf"^\[{epoch - 1}\]\s+\|")
    for i, line in enumerate(log):
        if marker.search(line):
            for following in log[i + 1:i + 5]:
                if "val_per_class_acc:" in following:
                    return json.loads(following.split("val_per_class_acc:", 1)[1].strip())
    return {}


def per_class_rows(classifiers: list[dict]) -> list[dict]:
    selected = ["d0_k400_direct", "s0_sup", "ph2_l010", "re2_k10_s50", "h2_emg_both_p1_k10", "hn1_null_p1"]
    by_exp = {row["experiment"]: row for row in classifiers}
    values = {}
    for exp in selected:
        row = by_exp[exp]
        values[exp] = parse_epoch_per_class(Path(row["summary_path"]), int(row["best_epoch"]))
    classes = sorted(set().union(*(value.keys() for value in values.values())))
    rows = []
    for cls in classes:
        row = {"class": cls}
        for exp in selected:
            row[exp] = values[exp].get(cls)
        row["sup_minus_direct"] = (row["s0_sup"] - row["d0_k400_direct"]) if row["s0_sup"] is not None else None
        row["re2_minus_sup"] = (row["re2_k10_s50"] - row["s0_sup"]) if row["re2_k10_s50"] is not None else None
        rows.append(row)
    return rows


def load_debug(path: Path) -> dict[int, list[dict]]:
    epochs = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        epochs[int(obj["epoch"])].append(obj)
    return epochs


def epoch_value(items: list[dict], key: str):
    return mean([item.get(key) for item in items])


def pretrain_rows(root: Path) -> list[dict]:
    rows = []
    for args_path in root.glob("pretrain/fold_MR/*/*/args.json"):
        stage, exp = experiment_from(args_path, "pretrain")
        args = read_json(args_path)
        log_path = args_path.with_name("debug_train_log.jsonl")
        if not log_path.is_file():
            continue
        epochs = load_debug(log_path)
        final_epochs = [e for e in sorted(epochs) if e >= max(epochs) - 4]
        active_start = min(
            int(args.get("proto_loss_start_epoch", 9999)) if float(args.get("lambda_proto", 0)) > 0 else 9999,
            int(args.get("rel_loss_start_epoch", 9999)) if float(args.get("lambda_rel", 0)) > 0 else 9999,
        )
        active_epochs = [e for e in sorted(epochs) if e >= active_start]
        def avg_epochs(key, selected):
            return mean([epoch_value(epochs[e], key) for e in selected])
        feature_final = []
        grad_final = []
        queue_pos_final = []
        nonfinite = 0
        for e in final_epochs:
            for item in epochs[e]:
                feature_final.append((item.get("feature_stats") or {}).get("q_feature_std_mean"))
                grad_final.append((item.get("grad_stats") or {}).get("total_grad_norm"))
                queue_pos_final.append((((item.get("supcon_queue_anchor_stats") or {}).get("pos_per_anchor_queue") or {}).get("mean")))
                nonfinite += int(bool((item.get("nonfinite_check") or {}).get("has_nonfinite")))
        boundary = {}
        for name, point in (("proto", int(args.get("proto_loss_start_epoch", 9999))), ("rel", int(args.get("rel_loss_start_epoch", 9999)))):
            for offset in (-1, 0, 1, 10):
                e = point + offset
                boundary[f"{name}_weighted_e{offset:+d}"] = epoch_value(epochs[e], f"weighted_{name}_contrib") if e in epochs else None
        rows.append({
            "stage": stage, "experiment": exp, "ablation_mode": args.get("ablation_mode"),
            "P": args.get("default_num_prototypes"), "lambda_proto": args.get("lambda_proto"),
            "lambda_rel": args.get("lambda_rel"), "proto_start": args.get("proto_loss_start_epoch"),
            "rel_start": args.get("rel_loss_start_epoch"), "rel_topk": args.get("rel_topk_diff_classes"),
            "rel_same_weight": args.get("rel_same_weight"), "rel_diff_weight": args.get("rel_diff_weight"),
            "rel_schedule": args.get("rel_lambda_schedule"),
            "final5_loss": avg_epochs("loss", final_epochs),
            "final5_supcon": avg_epochs("loss_supcon", final_epochs),
            "final5_proto": avg_epochs("loss_proto", final_epochs),
            "final5_rel": avg_epochs("loss_rel", final_epochs),
            "final5_weighted_proto": avg_epochs("weighted_proto_contrib", final_epochs),
            "final5_weighted_rel": avg_epochs("weighted_rel_contrib", final_epochs),
            "active_mean_weighted_proto": avg_epochs("weighted_proto_contrib", active_epochs),
            "active_mean_weighted_rel": avg_epochs("weighted_rel_contrib", active_epochs),
            "final5_q_feature_std": mean(feature_final), "final5_grad_norm": mean(grad_final),
            "final5_queue_positive_mean": mean(queue_pos_final), "nonfinite_records": nonfinite,
            **boundary,
        })
    return sorted(rows, key=lambda row: (row["stage"], row["experiment"]))


def diagnostic_rows(root: Path) -> list[dict]:
    grouped = defaultdict(list)
    for path in root.rglob("proto_diag_epoch_*.json"):
        stage, exp = experiment_from(path, "pretrain")
        grouped[(stage, exp)].append(path)
    rows = []
    for (stage, exp), paths in sorted(grouped.items()):
        objects = [read_json(path) for path in sorted(paths)]
        final = max(objects, key=lambda obj: int(obj["epoch"]))
        rows.append({
            "stage": stage, "experiment": exp, "first_diag_epoch": min(int(obj["epoch"]) for obj in objects),
            "final_diag_epoch": int(final["epoch"]), "num_diag_files": len(objects),
            "active_prototypes": final.get("active_prototypes"),
            "strict_dead": final.get("strict_dead_prototypes"), "near_dead": final.get("near_dead_prototypes"),
            "assignment_cv_mean": final.get("assignment_cv_mean"),
            "assignment_entropy_norm": final.get("assignment_entropy_normalized_mean"),
            "same_class_cos_mean": final.get("same_class_cos_mean"),
            "nearest_diff_cos_mean": final.get("nearest_different_class_cos_mean"),
            "nearest_diff_cos_max": final.get("nearest_different_class_cos_max"),
            "valid_samples": final.get("valid_samples"), "invalid_samples": final.get("invalid_samples"),
            "max_near_dead_over_time": max(int(obj.get("near_dead_prototypes", 0) or 0) for obj in objects),
            "min_entropy_over_time": min(float(obj.get("assignment_entropy_normalized_mean", 1) or 1) for obj in objects),
            "mean_same_class_cos_over_time": mean([obj.get("same_class_cos_mean") for obj in objects]),
        })
    return rows


def environment_rows(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("prototype_environment/fold_MR/*/*/environment_prototype_summary.json"):
        parts = path.parts
        i = parts.index("prototype_environment")
        stage, exp = parts[i + 2], parts[i + 3]
        series = read_json(path)
        if not series:
            continue
        final = max(series, key=lambda obj: int(obj["epoch"]))
        rows.append({
            "stage": stage, "experiment": exp, "final_epoch": final["epoch"],
            "lighting_nmi": final.get("mean_class_nmi"), "lighting_ari": final.get("mean_class_ari"),
            "lighting_purity": final.get("mean_class_purity"),
            "max_lighting_nmi": max(float(obj.get("mean_class_nmi", 0)) for obj in series),
            "assignment_ids_used": final.get("assignment_ids_used"),
        })
    return sorted(rows, key=lambda row: (row["stage"], row["experiment"]))


def rank_map(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: values[key], reverse=True)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for key in ordered[i:j]:
            ranks[key] = rank
        i = j
    return ranks


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return numerator / denominator if denominator else None


def test_rows(root: Path, classifiers: list[dict]) -> tuple[list[dict], list[dict], dict]:
    validation = {row["experiment"]: row for row in classifiers}
    rows = []
    per_class = []
    for path in root.glob("test/fold_MR/*/*/test_results.csv"):
        rel = path.relative_to(root).parts
        stage, exp = rel[2], rel[3]
        records = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
        if not records:
            continue
        obj = records[-1]
        val = validation[exp]
        row = {
            "stage": stage, "experiment": exp, "samples": int(obj["num_samples"]),
            "test_balanced_acc": float(obj["test_balanced_acc"]),
            "test_macro_f1": float(obj["test_macro_f1"]), "test_accuracy": float(obj["test_acc"]),
            "test_loss": float(obj["test_loss"]),
            "val_balanced_acc": val["best_val_balanced_acc"], "val_macro_f1": val["best_val_macro_f1"],
            "val_accuracy": val["best_val_acc"],
            "generalization_gap_ba": val["best_val_balanced_acc"] - float(obj["test_balanced_acc"]),
            "test_results_path": str(path),
        }
        rows.append(row)
        classes = json.loads(obj["test_per_class_acc_json"])
        for cls, recall in classes.items():
            per_class.append({"stage": stage, "experiment": exp, "class": cls, "recall": recall})
    by_exp = {row["experiment"]: row for row in rows}
    direct, sup = by_exp["d0_k400_direct"], by_exp["s0_sup"]
    for row in rows:
        row["delta_ba_vs_direct"] = row["test_balanced_acc"] - direct["test_balanced_acc"]
        row["delta_ba_vs_sup"] = row["test_balanced_acc"] - sup["test_balanced_acc"]
        row["delta_f1_vs_sup"] = row["test_macro_f1"] - sup["test_macro_f1"]
        row["delta_accuracy_vs_sup"] = row["test_accuracy"] - sup["test_accuracy"]
    val_values = {row["experiment"]: row["val_balanced_acc"] for row in rows}
    test_values = {row["experiment"]: row["test_balanced_acc"] for row in rows}
    val_ranks, test_ranks = rank_map(val_values), rank_map(test_values)
    for row in rows:
        row["validation_rank"] = val_ranks[row["experiment"]]
        row["test_rank"] = test_ranks[row["experiment"]]
        row["rank_change_test_minus_val"] = row["test_rank"] - row["validation_rank"]
    keys = sorted(by_exp)
    meta = {
        "experiments": len(rows),
        "pearson_val_test_ba": correlation([val_values[k] for k in keys], [test_values[k] for k in keys]),
        "spearman_val_test_ba": correlation([val_ranks[k] for k in keys], [test_ranks[k] for k in keys]),
        "mean_generalization_gap_ba": mean([row["generalization_gap_ba"] for row in rows]),
        "median_generalization_gap_ba": statistics.median(row["generalization_gap_ba"] for row in rows),
    }
    return sorted(rows, key=lambda row: row["test_balanced_acc"], reverse=True), per_class, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    classifiers = classifier_rows(args.results_root)
    tests, test_per_class, test_meta = test_rows(args.results_root, classifiers)
    outputs = {
        "classification_ranking": classifiers,
        "selected_per_class": per_class_rows(classifiers),
        "pretrain_dynamics": pretrain_rows(args.results_root),
        "prototype_diagnostics": diagnostic_rows(args.results_root),
        "prototype_lighting": environment_rows(args.results_root),
        "outer_test_analysis": tests,
        "outer_test_per_class": test_per_class,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv(args.output / f"{name}.csv", rows)
    (args.output / "analysis_tables.json").write_text(json.dumps(outputs | {"outer_test_meta": test_meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "outer_test_meta.json").write_text(json.dumps(test_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
