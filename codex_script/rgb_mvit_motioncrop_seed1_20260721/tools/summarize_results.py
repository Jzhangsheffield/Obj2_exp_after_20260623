#!/usr/bin/env python3
"""Collect classifier-test and feature-separation outputs into two CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, type=Path)
    args = p.parse_args()
    out = args.results_root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    test_rows = []
    test_root = args.results_root / "test"
    if test_root.is_dir():
        for csv_path in sorted(test_root.glob("*/test_results.csv")):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row = {"experiment_id": csv_path.parent.name, **rows[-1]}
                test_rows.append(row)

    feature_rows = []
    feature_root = args.results_root / "features"
    if feature_root.is_dir():
        for path in sorted(feature_root.glob("*/feature_metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for feature_name, metrics in payload["metrics"].items():
                feature_rows.append({
                    "experiment_id": path.parent.name,
                    "backbone": payload["backbone"],
                    "feature": feature_name,
                    **metrics,
                })

    write_csv(out / "classifier_test_summary.csv", test_rows)
    write_csv(out / "feature_separation_summary.csv", feature_rows)
    payload = {"classifier_tests": test_rows, "feature_separation": feature_rows}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "test_rows": len(test_rows), "feature_rows": len(feature_rows)}, indent=2))


if __name__ == "__main__":
    main()
