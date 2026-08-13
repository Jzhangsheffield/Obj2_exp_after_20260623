#!/usr/bin/env python3
"""Paired, multi-fold and multi-seed analysis for confirmation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("")
            return
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def t95(sample_count: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045}
    if sample_count <= 1:
        return 0.0
    return table.get(sample_count, 1.96)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    true = [int(row["true_label_id"]) for row in rows]
    pred = [int(row["pred_label_id"]) for row in rows]
    classes = sorted(set(true))
    accuracy = mean([float(a == b) for a, b in zip(true, pred)])
    recalls, f1s = [], []
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(true, pred))
        fn = sum(a == cls and b != cls for a, b in zip(true, pred))
        fp = sum(a != cls and b == cls for a, b in zip(true, pred))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {"accuracy": accuracy, "balanced_accuracy": mean(recalls), "macro_f1": mean(f1s), "num_samples": len(rows), "num_present_classes": len(classes)}


def per_class(rows: list[dict[str, str]], identity: dict) -> list[dict]:
    result = []
    for cls in sorted(set(int(row["true_label_id"]) for row in rows)):
        selected = [row for row in rows if int(row["true_label_id"]) == cls]
        tp = sum(int(row["pred_label_id"]) == cls for row in selected)
        predicted = sum(int(row["pred_label_id"]) == cls for row in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / len(selected) if selected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({**identity, "class_id": cls, "class_name": selected[0].get("true_label_name", str(cls)), "support": len(selected), "precision": precision, "recall": recall, "f1": f1})
    return result


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(control: list[dict], active: list[dict], iterations: int, seed: int) -> dict:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(control):
        by_class[int(row["true_label_id"])].append(index)
    distributions = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1")}
    for _ in range(iterations):
        indices = []
        for class_indices in by_class.values():
            indices.extend(rng.choice(class_indices) for _ in class_indices)
        c_metrics = metrics([control[index] for index in indices])
        a_metrics = metrics([active[index] for index in indices])
        for name in distributions:
            distributions[name].append(a_metrics[name] - c_metrics[name])
    output = {}
    for name, values in distributions.items():
        output[f"{name}_delta_mean"] = mean(values)
        output[f"{name}_ci_low"] = percentile(values, 0.025)
        output[f"{name}_ci_high"] = percentile(values, 0.975)
        output[f"{name}_prob_gt_zero"] = mean([float(value > 0) for value in values])
    return output


def align_pair(control: list[dict], active: list[dict]) -> tuple[list[dict], list[dict]]:
    left = {row["original_key"]: row for row in control}
    right = {row["original_key"]: row for row in active}
    if set(left) != set(right):
        raise RuntimeError(f"Paired sample IDs differ: control={len(left)}, active={len(right)}, overlap={len(set(left) & set(right))}")
    keys = sorted(left)
    for key in keys:
        if left[key]["true_label_id"] != right[key]["true_label_id"]:
            raise RuntimeError(f"True label mismatch for {key}")
    return [left[key] for key in keys], [right[key] for key in keys]


def summary_rows(run_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        groups[row["config_id"]].append(row)
    result = []
    for config_id, rows in sorted(groups.items()):
        item = {"config_id": config_id, "num_runs": len(rows), "num_folds": len(set(row["fold"] for row in rows)), "num_seeds": len(set(row["seed"] for row in rows))}
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = sd(values)
            margin = t95(len(values)) * sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            item[f"{metric}_ci_low"] = mean(values) - margin
            item[f"{metric}_ci_high"] = mean(values) + margin
        result.append(item)
    return result


def fold_level_config_rows(run_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["config_id"], row["fold"])].append(row)
    output = []
    for (config_id, fold), rows in sorted(grouped.items()):
        item = {"config_id": config_id, "fold": fold, "num_seeds": len(rows)}
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_seed_mean"] = mean(values)
            item[f"{metric}_seed_std"] = sd(values)
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int)
    args = parser.parse_args()
    registry = json.loads((ROOT / "config" / "locked_config_registry.json").read_text(encoding="utf-8"))
    stat_plan = json.loads((ROOT / "config" / "statistics_plan.json").read_text(encoding="utf-8"))
    iterations = args.bootstrap_iterations or int(stat_plan["bootstrap_iterations"])
    output = args.output or args.results_root / "analysis_confirmation"
    output.mkdir(parents=True, exist_ok=True)

    expected_ids = None
    if args.manifest:
        expected_ids = {row["run_id"] for row in read_csv(args.manifest)}
    runs: dict[tuple[str, str, int], list[dict]] = {}
    run_metrics, class_rows, environment_rows = [], [], []
    duplicate_keys = []
    for path in sorted(args.results_root.glob("test/fold_*/*/s*/predictions.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        config_id, fold, seed = rows[0]["config_id"], rows[0]["fold"], int(rows[0]["seed"])
        run_id = f"{config_id}_f{fold}_s{seed}"
        if expected_ids is not None and run_id not in expected_ids:
            continue
        keys = [row["original_key"] for row in rows]
        if len(keys) != len(set(keys)):
            duplicate_keys.append(run_id)
        identity = {"run_id": run_id, "config_id": config_id, "fold": fold, "seed": seed}
        item = {**identity, **metrics(rows), "predictions_path": str(path)}
        run_metrics.append(item)
        class_rows.extend(per_class(rows, identity))
        for environment in sorted(set(row.get("lighting", "") for row in rows)):
            subset = [row for row in rows if row.get("lighting", "") == environment]
            environment_rows.append({**identity, "lighting": environment, **metrics(subset)})
        runs[(config_id, fold, seed)] = rows

    pair_rows, bootstrap_rows, mcnemar_rows, flip_rows = [], [], [], []
    comparisons = [{**pair, "comparison_type": "strict_active_null"} for pair in registry["pairs"]]
    observed_configs = sorted(set(config for config, _, _ in runs))
    if "s0" in observed_configs:
        comparisons.extend({"pair_id": f"benchmark_{active}_vs_s0", "control": "s0", "active": active, "comparison_type": "benchmark_vs_s0"} for active in observed_configs if active != "s0")
    if "d0" in observed_configs:
        comparisons.extend({"pair_id": f"benchmark_{active}_vs_d0", "control": "d0", "active": active, "comparison_type": "benchmark_vs_direct"} for active in observed_configs if active not in {"d0", "s0"})
    for pair_index, pair in enumerate(comparisons):
        control, active = pair["control"], pair["active"]
        keys = sorted(set((fold, seed) for config, fold, seed in runs if config == control) & set((fold, seed) for config, fold, seed in runs if config == active))
        for fold, seed in keys:
            c_rows, a_rows = align_pair(runs[(control, fold, seed)], runs[(active, fold, seed)])
            cm, am = metrics(c_rows), metrics(a_rows)
            identity = {"pair_id": pair["pair_id"], "comparison_type": pair["comparison_type"], "control": control, "active": active, "fold": fold, "seed": seed, "num_samples": len(c_rows)}
            paired = {**identity}
            for name in ("balanced_accuracy", "macro_f1", "accuracy"):
                paired[f"control_{name}"] = cm[name]
                paired[f"active_{name}"] = am[name]
                paired[f"delta_{name}"] = am[name] - cm[name]
            pair_rows.append(paired)
            boot = paired_bootstrap(c_rows, a_rows, iterations, int(stat_plan["bootstrap_seed"]) + pair_index * 1000 + seed)
            bootstrap_rows.append({**identity, "iterations": iterations, **boot})
            b = sum(int(c["correct"]) == 0 and int(a["correct"]) == 1 for c, a in zip(c_rows, a_rows))
            c = sum(int(crow["correct"]) == 1 and int(arow["correct"]) == 0 for crow, arow in zip(c_rows, a_rows))
            mcnemar_rows.append({**identity, "active_correct_control_wrong": b, "control_correct_active_wrong": c, "discordant": b + c, "exact_p_value": exact_mcnemar(b, c)})
            class_ids = sorted(set(int(row["true_label_id"]) for row in c_rows))
            for cls in class_ids:
                selected = [(crow, arow) for crow, arow in zip(c_rows, a_rows) if int(crow["true_label_id"]) == cls]
                flip_rows.append({**identity, "class_id": cls, "class_name": selected[0][0]["true_label_name"], "support": len(selected), "active_fixed": sum(int(crow["correct"]) == 0 and int(arow["correct"]) == 1 for crow, arow in selected), "active_broke": sum(int(crow["correct"]) == 1 and int(arow["correct"]) == 0 for crow, arow in selected)})

    fold_pair_rows = []
    by_pair_fold: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pair_rows:
        by_pair_fold[(row["pair_id"], row["fold"])].append(row)
    for (pair_id, fold), rows in sorted(by_pair_fold.items()):
        item = {"pair_id": pair_id, "comparison_type": rows[0]["comparison_type"], "control": rows[0]["control"], "active": rows[0]["active"], "fold": fold, "num_seeds": len(rows)}
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            values = [float(row[f"delta_{metric}"]) for row in rows]
            item[f"delta_{metric}_seed_mean"] = mean(values)
            item[f"delta_{metric}_seed_std"] = sd(values)
            item[f"positive_seeds_{metric}"] = sum(value > 0 for value in values)
        fold_pair_rows.append(item)

    pair_summaries = []
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in fold_pair_rows:
        by_pair[row["pair_id"]].append(row)
    for pair_id, folds in sorted(by_pair.items()):
        raw_runs = [row for row in pair_rows if row["pair_id"] == pair_id]
        item = {"pair_id": pair_id, "comparison_type": folds[0]["comparison_type"], "control": folds[0]["control"], "active": folds[0]["active"], "num_paired_runs": len(raw_runs), "num_folds": len(folds), "max_num_seeds_per_fold": max(row["num_seeds"] for row in folds)}
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            values = [float(row[f"delta_{metric}_seed_mean"]) for row in folds]
            margin = t95(len(values)) * sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            item[f"delta_{metric}_fold_mean"] = mean(values)
            item[f"delta_{metric}_fold_std"] = sd(values)
            item[f"delta_{metric}_fold_ci_low"] = mean(values) - margin
            item[f"delta_{metric}_fold_ci_high"] = mean(values) + margin
            item[f"positive_folds_{metric}"] = sum(value > 0 for value in values)
        pair_summaries.append(item)

    expected_missing = []
    if expected_ids is not None:
        observed_ids = {row["run_id"] for row in run_metrics}
        expected_missing = sorted(expected_ids - observed_ids)

    write_csv(output / "overall_runs.csv", run_metrics)
    write_csv(output / "config_mean_std_ci.csv", summary_rows(run_metrics))
    write_csv(output / "config_by_fold_seed_summary.csv", fold_level_config_rows(run_metrics))
    write_csv(output / "paired_run_differences.csv", pair_rows)
    write_csv(output / "paired_fold_seed_summary.csv", fold_pair_rows)
    write_csv(output / "paired_difference_summary.csv", pair_summaries)
    write_csv(output / "paired_bootstrap.csv", bootstrap_rows)
    write_csv(output / "mcnemar_exact.csv", mcnemar_rows)
    write_csv(output / "per_class_metrics.csv", class_rows)
    write_csv(output / "per_environment_metrics.csv", environment_rows)
    write_csv(output / "prediction_flips_by_class.csv", flip_rows)
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    strict_paired_runs = sum(row["comparison_type"] == "strict_active_null" for row in pair_rows)
    write_json(output / "analysis_audit.json", {"prediction_runs": len(run_metrics), "all_paired_comparisons": len(pair_rows), "strict_active_null_paired_runs": strict_paired_runs, "expected_missing": expected_missing, "duplicate_sample_id_runs": duplicate_keys, "bootstrap_iterations": iterations})

    report = [
        "# Confirmation experiment statistical summary", "",
        f"- Prediction runs found: {len(run_metrics)}",
        f"- Strict Active–Null paired runs: {strict_paired_runs}",
        f"- All paired comparisons including S0/direct benchmarks: {len(pair_rows)}",
        f"- Missing expected runs: {len(expected_missing)}",
        f"- Bootstrap: {iterations} stratified paired resamples per fold/seed pair",
        "- McNemar: exact two-sided test on paired correctness",
        "",
        "## Configuration-level mean ± standard deviation", "",
        "| Config | Runs | Folds | BA | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows(run_metrics), key=lambda item: item["balanced_accuracy_mean"], reverse=True):
        report.append(f"| {row['config_id']} | {row['num_runs']} | {row['num_folds']} | {100*row['balanced_accuracy_mean']:.2f} ± {100*row['balanced_accuracy_std']:.2f} | {100*row['macro_f1_mean']:.2f} ± {100*row['macro_f1_std']:.2f} | {100*row['accuracy_mean']:.2f} ± {100*row['accuracy_std']:.2f} |")
    report.extend(["", "## Paired differences", "", "| Type | Pair | Paired runs | ΔBA | ΔMacro-F1 | ΔAccuracy | BA positive |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in pair_summaries:
        report.append(f"| {row['comparison_type']} | {row['active']} − {row['control']} | {row['num_paired_runs']} | {100*row['delta_balanced_accuracy_fold_mean']:+.2f} | {100*row['delta_macro_f1_fold_mean']:+.2f} | {100*row['delta_accuracy_fold_mean']:+.2f} | {row['positive_folds_balanced_accuracy']}/{row['num_folds']} folds |")
    if expected_missing:
        report.extend(["", "## Missing expected runs", "", *[f"- {run_id}" for run_id in expected_missing]])
    (output / "CONFIRMATION_STATISTICAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(run_metrics), "strict_paired_runs": strict_paired_runs, "all_paired_comparisons": len(pair_rows), "missing": len(expected_missing)}, indent=2))


if __name__ == "__main__":
    main()
