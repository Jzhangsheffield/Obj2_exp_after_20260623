#!/usr/bin/env python3
"""Inspect completion state of a unified manifest without launching jobs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def base(results: Path, kind: str, row: dict) -> Path:
    prefix = "v" if row["protocol"] == "subject_dev" else "t"
    return results / kind / row["task"] / row["protocol"] / f"{prefix}{row['subject']}" / row["config_id"] / row["augmentation_id"] / row["sampling_id"] / f"s{row['seed']}"


def complete(results: Path, row: dict, phase: str) -> bool:
    if phase == "pretrain":
        return row["pretrain"].lower() not in {"1", "true", "yes"} or (base(results, "pretrain", row) / "checkpoint_0200.pth").is_file()
    if phase == "finetune": return len(list(base(results, "finetune", row).rglob("epoch_050.pth"))) == 1
    if phase == "evaluate": target = base(results, "dev_eval", row)
    elif phase == "test": target = base(results, "test", row)
    else: raise ValueError(phase)
    return (target / "test_results.csv").is_file() and (target / "predictions.csv").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["pretrain", "finetune", "evaluate", "test"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); rows = read_rows(args.manifest)
    missing = [row for row in rows if not complete(args.results_root, row, args.phase)]
    output = args.output or args.manifest.with_name(f"{args.manifest.stem}_missing_{args.phase}.csv")
    with output.open("w", newline="", encoding="utf-8") as stream:
        if missing:
            writer = csv.DictWriter(stream, fieldnames=list(missing[0])); writer.writeheader(); writer.writerows(missing)
    print(f"phase={args.phase} total={len(rows)} complete={len(rows)-len(missing)} missing={len(missing)} output={output}")


if __name__ == "__main__": main()
