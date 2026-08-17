#!/usr/bin/env python3
"""Build audited train/test manifests for take_put, middle and full tasks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def counts(rows: list[dict], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset = Path(args.dataset_root).resolve()
    source = dataset / cfg["sources"]["merged_manifest"]
    output = Path(args.output).resolve()
    rows = read_jsonl(source)
    if not rows:
        raise ValueError(f"No rows in {source}")

    required = {"sample_name", "original_key", "person", "tier1"}
    missing = required - rows[0].keys()
    if missing:
        raise KeyError(f"Manifest misses required keys: {sorted(missing)}")
    for key in ("sample_name", "original_key"):
        values = [str(row[key]) for row in rows]
        if len(values) != len(set(values)):
            duplicates = [name for name, count in Counter(values).items() if count > 1][:10]
            raise ValueError(f"Duplicate {key}: {duplicates}")
    expected_people = set(cfg["data"]["people"])
    actual_people = {str(row["person"]) for row in rows}
    if actual_people != expected_people:
        raise ValueError(f"People mismatch: actual={sorted(actual_people)}, expected={sorted(expected_people)}")

    audit: dict = {
        "source": str(source),
        "source_rows": len(rows),
        "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "source_counts_by_person": counts(rows, "person"),
        "source_counts_by_tier1": counts(rows, "tier1"),
        "tasks": {},
    }
    for task, classes in cfg["data"]["tasks"].items():
        class_set = set(classes)
        task_rows = [row for row in rows if row["tier1"] in class_set]
        actual_classes = {str(row["tier1"]) for row in task_rows}
        if actual_classes != class_set:
            raise ValueError(f"Task {task} misses classes: {sorted(class_set - actual_classes)}")
        label_map = {
            "tier1": {name: index for index, name in enumerate(classes)},
            "__meta__": {"task": task, "class_order": classes, "source": str(source)},
        }
        task_audit = {
            "classes": classes,
            "total": len(task_rows),
            "counts_by_person": counts(task_rows, "person"),
            "counts_by_tier1": counts(task_rows, "tier1"),
            "folds": {},
        }
        for fold, fold_cfg in cfg["data"]["folds"].items():
            fold_dir = output / task / fold
            train_people = set(fold_cfg["train_people"])
            test_people = set(fold_cfg["test_people"])
            if train_people & test_people or train_people | test_people != expected_people:
                raise ValueError(f"Invalid people partition in {fold}")
            train = [row for row in task_rows if row["person"] in train_people]
            test = [row for row in task_rows if row["person"] in test_people]
            train_keys = {row["original_key"] for row in train}
            test_keys = {row["original_key"] for row in test}
            overlap = train_keys & test_keys
            if overlap:
                raise ValueError(f"Data leakage in {task}/{fold}: {list(overlap)[:5]}")
            if len(train) + len(test) != len(task_rows):
                raise ValueError(f"Incomplete partition in {task}/{fold}")
            write_jsonl(fold_dir / "train.jsonl", train)
            write_jsonl(fold_dir / "test.jsonl", test)
            (fold_dir / "label_map.json").write_text(
                json.dumps(label_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            fold_audit = {
                "role": fold_cfg["role"],
                "train_people": fold_cfg["train_people"],
                "test_people": fold_cfg["test_people"],
                "train_rows": len(train),
                "test_rows": len(test),
                "train_counts_by_tier1": counts(train, "tier1"),
                "test_counts_by_tier1": counts(test, "tier1"),
                "overlap_original_key": 0,
            }
            (fold_dir / "audit.json").write_text(
                json.dumps(fold_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            task_audit["folds"][fold] = fold_audit
        audit["tasks"][task] = task_audit
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "OK", "output": str(output), "tasks": audit["tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

