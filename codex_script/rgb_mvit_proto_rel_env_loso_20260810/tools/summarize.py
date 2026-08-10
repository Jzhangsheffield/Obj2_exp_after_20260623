#!/usr/bin/env python3
"""Aggregate inner-validation and outer-person metrics."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.results_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    validation: list[dict] = []
    for path in args.results_root.glob("classifier/fold_*/*/*/**/summary.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(args.results_root).parts
        validation.append({
            "fold": rel[1].replace("fold_", ""), "stage": rel[2], "experiment": rel[3],
            "best_val_balanced_acc": obj.get("best_val_balanced_acc"),
            "best_val_macro_f1": obj.get("best_val_macro_f1"),
            "best_val_acc": obj.get("best_val_acc"),
            "best_val_balanced_epoch": obj.get("best_val_balanced_epoch"),
            "path": str(path),
        })
    tests: list[dict] = []
    for path in args.results_root.glob("test/fold_*/*/*/**/*_test_metrics.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(args.results_root).parts
        tests.append({
            "fold": rel[1].replace("fold_", ""), "stage": rel[2], "experiment": rel[3],
            "balanced_acc": obj.get("balanced_acc"), "macro_f1": obj.get("macro_f1"),
            "accuracy": obj.get("acc"), "present_classes": obj.get("num_present_classes"), "path": str(path),
        })

    def write_csv(name: str, rows: list[dict]) -> None:
        with (out / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
            writer.writeheader()
            if rows:
                writer.writerows(rows)
    write_csv("validation_runs.csv", validation)
    write_csv("outer_test_runs.csv", tests)

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in tests:
        groups.setdefault((row["stage"], row["experiment"]), []).append(row)
    aggregate: list[dict] = []
    for (stage, experiment), rows in sorted(groups.items()):
        ba = [float(row["balanced_acc"]) for row in rows if row["balanced_acc"] is not None]
        f1 = [float(row["macro_f1"]) for row in rows if row["macro_f1"] is not None]
        aggregate.append({
            "stage": stage, "experiment": experiment, "folds": len(rows),
            "balanced_acc_mean": mean(ba), "balanced_acc_sample_std": std(ba),
            "macro_f1_mean": mean(f1), "macro_f1_sample_std": std(f1),
        })
    write_csv("outer_test_mean_std.csv", aggregate)
    lines = ["# MViT old Proto/Rel LOSO summary", "", f"Validation runs found: {len(validation)}", "", f"Outer-test runs found: {len(tests)}", "", "## Four-fold aggregates", "", "| Stage | Experiment | Folds | BA mean | BA std | Macro-F1 mean |", "|---|---|---:|---:|---:|---:|"]
    for row in aggregate:
        def fmt(value): return "" if value is None else f"{100 * value:.2f}%"
        lines.append(f"| {row['stage']} | {row['experiment']} | {row['folds']} | {fmt(row['balanced_acc_mean'])} | {fmt(row['balanced_acc_sample_std'])} | {fmt(row['macro_f1_mean'])} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summaries to {out}")


if __name__ == "__main__":
    main()
