#!/usr/bin/env python3
"""Aggregate per-run locked-test CSVs and rank by balanced accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from config_utils import load_stage, roots


def float_value(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    args = parser.parse_args()
    master, stage, _ = load_stage(Path(args.config).resolve())
    project, _ = roots(master, args.project_root, None)
    test_root = project / stage["finetune_output_rel"] / "test"
    rows: List[Dict[str, str]] = []
    for plan in stage.get("finetune_experiments", []):
        path = test_root / plan["id"] / "rgb_test_results.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["stage_id"] = stage["id"]
                row["experiment_id"] = plan["id"]
                rows.append(row)
    if not rows:
        raise FileNotFoundError("No per-run test CSVs under %s" % test_root)
    rows.sort(key=lambda row: (float_value(row, "test_balanced_acc"), float_value(row, "test_acc")), reverse=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    out = test_root / "rgb_test_results_ranked.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print("Ranked CSV: %s" % out)


if __name__ == "__main__":
    main()

