#!/usr/bin/env python3
"""Summarize fixed-epoch development/final results for the unified experiment package."""
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
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def metric_bundle(rows: list[dict], expected_classes: int) -> dict:
    if not rows:
        return {"samples": 0, "accuracy": float("nan"), "balanced_accuracy_present": float("nan"),
                "macro_f1_present": float("nan"), "present_classes": 0, "expected_classes": expected_classes}
    true = [int(row["true_label_id"]) for row in rows]; pred = [int(row["pred_label_id"]) for row in rows]
    classes = sorted(set(true)); recalls = []; f1s = []
    for cls in classes:
        tp = sum(a == cls and b == cls for a, b in zip(true, pred))
        fn = sum(a == cls and b != cls for a, b in zip(true, pred))
        fp = sum(a != cls and b == cls for a, b in zip(true, pred))
        recall = tp / (tp + fn) if tp + fn else 0.0; precision = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(recall); f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "samples": len(rows), "accuracy": mean([float(a == b) for a, b in zip(true, pred)]),
        "balanced_accuracy_present": mean(recalls), "macro_f1_present": mean(f1s),
        "present_classes": len(classes), "expected_classes": expected_classes,
    }


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0: return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values); pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def align(left: list[dict], right: list[dict]) -> tuple[list[dict], list[dict]]:
    lmap = {row["original_key"]: row for row in left}; rmap = {row["original_key"]: row for row in right}
    if set(lmap) != set(rmap): raise RuntimeError(f"Paired sample mismatch: {len(lmap)} vs {len(rmap)}")
    keys = sorted(lmap)
    return [lmap[key] for key in keys], [rmap[key] for key in keys]


def paired_bootstrap(left: list[dict], right: list[dict], expected_classes: int, iterations: int, seed: int) -> dict:
    rng = random.Random(seed); by_class: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(left): by_class[int(row["true_label_id"])].append(index)
    dist = {name: [] for name in ("balanced_accuracy_present", "macro_f1_present", "accuracy")}
    for _ in range(iterations):
        indices = [rng.choice(indices) for indices in by_class.values() for _ in indices]
        lm = metric_bundle([left[index] for index in indices], expected_classes)
        rm = metric_bundle([right[index] for index in indices], expected_classes)
        for name in dist: dist[name].append(float(rm[name]) - float(lm[name]))
    output = {}
    for name, values in dist.items():
        output[f"delta_{name}_mean"] = mean(values); output[f"delta_{name}_ci_low"] = percentile(values, 0.025)
        output[f"delta_{name}_ci_high"] = percentile(values, 0.975); output[f"delta_{name}_prob_gt_zero"] = mean([float(value > 0) for value in values])
    return output


def identity_from_path(results: Path, path: Path) -> dict:
    parts = path.relative_to(results).parts
    if len(parts) < 9:
        raise ValueError(f"Unexpected result path: {path}")
    kind, task, protocol, subject_tag, config, aug, sampling, seed = parts[:8]
    return {
        "kind": kind, "task": task, "protocol": protocol, "subject": subject_tag[1:],
        "config_id": config, "augmentation_id": aug, "sampling_id": sampling,
        "seed": int(seed.removeprefix("s")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True); parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = json.loads((ROOT / "config" / "unified_experiment_registry.json").read_text(encoding="utf-8"))
    output = args.output or args.results_root / "analysis_unified"; output.mkdir(parents=True, exist_ok=True)
    expected = {row["run_id"] for row in read_csv(args.manifest)} if args.manifest else None
    runs = []; class_rows = []; lighting_rows = []; confusion_rows = []; prediction_runs = {}
    observed: set[str] = set()
    for path in sorted(args.results_root.glob("*/t*/**/predictions.csv")):
        identity = identity_from_path(args.results_root, path)
        run_id = f"{identity['task']}_{identity['protocol']}_{identity['subject']}_{identity['config_id']}_{identity['augmentation_id']}_{identity['sampling_id']}_s{identity['seed']}"
        if expected is not None and run_id not in expected: continue
        observed.add(run_id); rows = read_csv(path); expected_classes = int(registry["tasks"][identity["task"]]["num_classes"])
        prediction_runs[(identity["task"], identity["protocol"], identity["subject"], identity["seed"], identity["augmentation_id"], identity["sampling_id"], identity["config_id"])] = rows
        overall = {**identity, "run_id": run_id, **metric_bundle(rows, expected_classes), "predictions_path": str(path)}
        if identity["task"] == "t17":
            legacy = [row for row in rows if row.get("true_label_name") not in {"take", "put"}]
            take_put = [row for row in rows if row.get("true_label_name") in {"take", "put"}]
            for prefix, subset, count in (("legacy15", legacy, 15), ("take_put", take_put, 2)):
                for key, value in metric_bundle(subset, count).items(): overall[f"{prefix}_{key}"] = value
        runs.append(overall)
        for cls in sorted(set(int(row["true_label_id"]) for row in rows)):
            selected = [row for row in rows if int(row["true_label_id"]) == cls]
            bundle = metric_bundle(selected, 1)
            predicted = sum(int(row["pred_label_id"]) == cls for row in rows)
            tp = sum(int(row["pred_label_id"]) == cls for row in selected)
            precision = tp / predicted if predicted else 0.0
            recall = bundle["balanced_accuracy_present"]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_rows.append({**identity, "run_id": run_id, "class_id": cls,
                               "class_name": selected[0].get("true_label_name", str(cls)),
                               "support": len(selected), "precision": precision, "recall": recall, "f1": f1})
        for lighting in sorted(set(row.get("lighting", "") for row in rows)):
            subset = [row for row in rows if row.get("lighting", "") == lighting]
            lighting_rows.append({**identity, "run_id": run_id, "lighting": lighting,
                                  **metric_bundle(subset, expected_classes)})
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows: counts[(row.get("true_label_name", row["true_label_id"]), row.get("pred_label_name", row["pred_label_id"]))] += 1
        for (true_name, pred_name), count in sorted(counts.items()):
            confusion_rows.append({**identity, "run_id": run_id, "true_class": true_name, "pred_class": pred_name, "count": count})

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[(row["task"], row["protocol"], row["config_id"], row["augmentation_id"], row["sampling_id"])].append(row)
    summaries = []
    metric_names = ["balanced_accuracy_present", "macro_f1_present", "accuracy"]
    for key, rows in sorted(grouped.items()):
        item = dict(zip(("task", "protocol", "config_id", "augmentation_id", "sampling_id"), key))
        item.update({"runs": len(rows), "subjects": len(set(row["subject"] for row in rows)), "seeds": len(set(row["seed"] for row in rows))})
        extra = [name for name in ("legacy15_balanced_accuracy_present", "legacy15_macro_f1_present", "take_put_balanced_accuracy_present", "take_put_macro_f1_present") if name in rows[0]]
        for metric in metric_names + extra:
            values = [float(row[metric]) for row in rows]; item[f"{metric}_mean"] = mean(values); item[f"{metric}_std"] = sd(values)
        summaries.append(item)

    pair_rows = []; mcnemar_rows = []; bootstrap_rows = []
    run_index = {(row["task"], row["protocol"], row["subject"], row["seed"], row["augmentation_id"], row["sampling_id"], row["config_id"]): row for row in runs}
    for pair in registry["pairs"]:
        for key, control in sorted(run_index.items()):
            task, protocol, subject, seed, aug, sampling, config = key
            if config != pair["control"]: continue
            active = run_index.get((task, protocol, subject, seed, aug, sampling, pair["active"]))
            if not active: continue
            item = {"pair_id": pair["pair_id"], "task": task, "protocol": protocol, "subject": subject,
                    "seed": seed, "augmentation_id": aug, "sampling_id": sampling,
                    "control": pair["control"], "active": pair["active"]}
            for metric in metric_names:
                item[f"control_{metric}"] = control[metric]; item[f"active_{metric}"] = active[metric]
                item[f"delta_{metric}"] = float(active[metric]) - float(control[metric])
            pair_rows.append(item)
            left, right = align(prediction_runs[key], prediction_runs[(task, protocol, subject, seed, aug, sampling, pair["active"])])
            b = sum(int(lrow["correct"]) == 0 and int(rrow["correct"]) == 1 for lrow, rrow in zip(left, right))
            c = sum(int(lrow["correct"]) == 1 and int(rrow["correct"]) == 0 for lrow, rrow in zip(left, right))
            mcnemar_rows.append({**item, "active_fixed": b, "active_broke": c, "discordant": b + c, "exact_p_value": exact_mcnemar(b, c)})
            bootstrap_rows.append({**item, "iterations": 2000, **paired_bootstrap(left, right, int(registry["tasks"][task]["num_classes"]), 2000, 20260813 + int(seed))})

    missing = sorted(expected - observed) if expected is not None else []
    write_csv(output / "overall_runs.csv", runs); write_csv(output / "configuration_mean_std.csv", summaries)
    write_csv(output / "per_class_metrics.csv", class_rows); write_csv(output / "per_lighting_metrics.csv", lighting_rows)
    write_csv(output / "confusion_long.csv", confusion_rows); write_csv(output / "strict_pair_differences.csv", pair_rows)
    write_csv(output / "mcnemar_exact.csv", mcnemar_rows); write_csv(output / "paired_bootstrap.csv", bootstrap_rows)
    (output / "analysis_audit.json").write_text(json.dumps({"runs": len(runs), "expected_missing": missing}, indent=2), encoding="utf-8")
    lines = ["# Unified fixed-epoch experiment summary", "", f"- Runs found: {len(runs)}", f"- Missing expected runs: {len(missing)}", "- Every reported model is evaluated from epoch 50.", "", "| Task | Protocol | Loss | Aug | Sampling | Runs | BA | Macro-F1 | Accuracy |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in sorted(summaries, key=lambda item: item["balanced_accuracy_present_mean"], reverse=True):
        lines.append(f"| {row['task']} | {row['protocol']} | {row['config_id']} | {row['augmentation_id']} | {row['sampling_id']} | {row['runs']} | {100*row['balanced_accuracy_present_mean']:.2f} ± {100*row['balanced_accuracy_present_std']:.2f} | {100*row['macro_f1_present_mean']:.2f} ± {100*row['macro_f1_present_std']:.2f} | {100*row['accuracy_mean']:.2f} ± {100*row['accuracy_std']:.2f} |")
    if missing: lines.extend(["", "## Missing runs", "", *[f"- {run_id}" for run_id in missing]])
    (output / "UNIFIED_STATISTICAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(runs), "summaries": len(summaries), "paired": len(pair_rows), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
