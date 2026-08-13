#!/usr/bin/env python3
"""Audit a run manifest and write a rerun manifest for incomplete phases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def one_checkpoint(root: Path, name: str) -> bool:
    return len(list(root.rglob(name))) == 1


def phase_status(row: dict[str, str]) -> dict[str, bool]:
    pretrain = Path(row["pretrain_dir"])
    finetune = Path(row["finetune_dir"])
    test = Path(row["test_dir"])
    return {
        "pretrain": one_checkpoint(pretrain, "checkpoint_0200.pth"),
        "finetune": one_checkpoint(finetune, "best_val_balanced.pth"),
        "test": (test / "test_results.csv").is_file() and (test / "predictions.csv").is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=["pretrain", "finetune", "test"], default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    audit = []
    missing = []
    for row in rows:
        status = phase_status(row)
        audit.append({"run_id": row["run_id"], **status})
        if not status[args.phase]:
            missing.append(row)
    output = args.output or args.manifest.with_name(args.manifest.stem + f"_missing_{args.phase}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["index"])
        writer.writeheader()
        for index, row in enumerate(missing):
            writer.writerow({**row, "index": index})
    print(json.dumps({"total": len(rows), "complete": len(rows) - len(missing), "missing": len(missing), "rerun_manifest": str(output), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
