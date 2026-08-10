#!/usr/bin/env python3
"""Build four strict person-held-out folds with grouped inner validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_key(row: dict) -> str:
    original = str(row.get("original_key", ""))
    return re.sub(r"_clip_.*$", "", original)


def stable_value(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


def inner_split(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    strata: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = (str(row.get("person")), str(row.get("tier1")), str(row.get("lighting")))
        strata[stratum][run_key(row)].append(row)
    val_runs: set[str] = set()
    for stratum, groups in strata.items():
        keys = sorted(groups, key=lambda key: stable_value("|".join(stratum) + "|" + key, seed))
        if len(keys) < 2:
            continue
        count = max(1, int(round(len(keys) * fraction)))
        count = min(count, len(keys) - 1)
        val_runs.update(keys[:count])
    train = [row for row in rows if run_key(row) not in val_runs]
    val = [row for row in rows if run_key(row) in val_runs]
    return train, val


def describe(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "runs": len({run_key(row) for row in rows}),
        "people": dict(Counter(str(row.get("person")) for row in rows)),
        "classes": dict(Counter(str(row.get("tier1")) for row in rows)),
        "lighting": dict(Counter(str(row.get("lighting")) for row in rows)),
        "class_lighting": dict(Counter(f"{row.get('tier1')}|{row.get('lighting')}" for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--people", nargs="+", required=True)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.inner_val_fraction < 0.5:
        raise ValueError("inner validation fraction must be between 0 and 0.5")

    combined: dict[str, dict] = {}
    for rel in (args.train, args.val, args.test):
        for row in read_jsonl(args.dataset_root / rel):
            combined[str(row["sample_name"])] = row
    rows = list(combined.values())
    actual_people = sorted({str(row.get("person")) for row in rows})
    if sorted(args.people) != actual_people:
        raise ValueError(f"Expected people {sorted(args.people)}, found {actual_people}")
    if any(str(row.get("lighting")) not in {"left", "normal", "right"} for row in rows):
        bad = sorted({str(row.get("lighting")) for row in rows} - {"left", "normal", "right"})
        raise ValueError(f"Unexpected lighting labels: {bad}")

    report = {"all": describe(rows), "folds": {}, "path_audit": {}}
    for person in args.people:
        outer_test = [row for row in rows if str(row.get("person")) == person]
        development = [row for row in rows if str(row.get("person")) != person]
        train, inner_val = inner_split(development, args.inner_val_fraction, args.seed)
        train_runs = {run_key(row) for row in train}
        val_runs = {run_key(row) for row in inner_val}
        test_runs = {run_key(row) for row in outer_test}
        if train_runs & val_runs or train_runs & test_runs or val_runs & test_runs:
            raise RuntimeError(f"Run leakage detected for fold {person}")
        fold_dir = args.output / f"fold_{person}"
        write_jsonl(fold_dir / "train.jsonl", train)
        write_jsonl(fold_dir / "inner_val.jsonl", inner_val)
        write_jsonl(fold_dir / "outer_test.jsonl", outer_test)
        report["folds"][person] = {
            "train": describe(train), "inner_val": describe(inner_val),
            "outer_test": describe(outer_test), "run_overlap": 0,
        }

    for key in ("rgb_cam_00143",):
        missing = [row["sample_name"] for row in rows if not row.get(key) or not (args.dataset_root / row[key]).is_file()]
        report["path_audit"][key] = {"missing": len(missing), "first_missing": missing[:20]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
