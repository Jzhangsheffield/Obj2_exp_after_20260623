#!/usr/bin/env python3
"""Build audited 15/17-class subject-development and final-refit manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PEOPLE = ("M", "J", "MR", "N")
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def key(row: dict) -> str:
    return str(row.get("sample_name") or row.get("original_key"))


def describe(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "people": dict(sorted(Counter(str(row.get("person")) for row in rows).items())),
        "tier1": dict(sorted(Counter(str(row.get("tier1")) for row in rows).items())),
        "lighting": dict(sorted(Counter(str(row.get("lighting")) for row in rows).items())),
    }


def load_union(dataset: Path) -> tuple[list[dict], list[dict]]:
    records: dict[str, dict] = {}
    sources: list[dict] = []
    for person in PEOPLE:
        fold_dir = dataset / f"{person}_as_test"
        for split in SPLITS:
            path = fold_dir / f"{split}_manifest.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            sources.append({"path": str(path), "sha256": digest(path), "rows": len(rows)})
            for row in rows:
                row_key = key(row)
                if not row_key or row_key == "None":
                    raise ValueError(f"Record without stable sample key in {path}")
                previous = records.get(row_key)
                if previous is not None and previous != row:
                    raise RuntimeError(f"Conflicting duplicate record: {row_key}")
                records[row_key] = row
    rows = sorted(records.values(), key=lambda item: (str(item.get("person")), key(item)))
    if set(str(row.get("person")) for row in rows) != set(PEOPLE):
        raise RuntimeError("The deduplicated dataset does not contain exactly M/J/MR/N")
    return rows, sources


def disjoint(*groups: list[dict]) -> bool:
    seen: set[str] = set()
    for rows in groups:
        current = {key(row) for row in rows}
        if seen & current:
            return False
        seen.update(current)
    return True


def emit_fold(root: Path, train: list[dict], val: list[dict] | None, test: list[dict] | None, meta: dict) -> dict:
    write_jsonl(root / "train.jsonl", train)
    if val is not None:
        write_jsonl(root / "val.jsonl", val)
    if test is not None:
        write_jsonl(root / "test.jsonl", test)
    groups = [train] + ([val] if val is not None else []) + ([test] if test is not None else [])
    if not disjoint(*groups):
        raise RuntimeError(f"Sample overlap detected in {root}")
    output = {
        **meta,
        "train": describe(train),
        "validation": describe(val or []),
        "test": describe(test or []),
        "sample_overlap": 0,
        "files": {
            "train": str(root / "train.jsonl"),
            "validation": str(root / "val.jsonl") if val is not None else None,
            "test": str(root / "test.jsonl") if test is not None else None,
        },
    }
    (root / "audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved-final-subject", default="N", choices=PEOPLE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {output}; pass --overwrite to rebuild deterministic manifests")
    rows, sources = load_union(args.dataset_root.resolve())
    tasks = {
        "t15": [row for row in rows if str(row.get("tier1")) not in {"take", "put"}],
        "t17": list(rows),
    }
    report = {
        "schema_version": 2,
        "reserved_final_subject": args.reserved_final_subject,
        "source_manifests": sources,
        "deduplicated_full_dataset": describe(rows),
        "tasks": {},
    }
    development_people = [person for person in PEOPLE if person != args.reserved_final_subject]
    for task, task_rows in tasks.items():
        task_report = {"all": describe(task_rows), "subject_dev": {}, "final_refit": {}}
        for heldout in development_people:
            train_people = [person for person in development_people if person != heldout]
            train = [row for row in task_rows if str(row.get("person")) in train_people]
            val = [row for row in task_rows if str(row.get("person")) == heldout]
            fold_root = output / task / "subject_dev" / f"holdout_{heldout}"
            task_report["subject_dev"][heldout] = emit_fold(
                fold_root, train, val, None,
                {"task": task, "protocol": "subject_dev", "heldout_subject": heldout,
                 "train_subjects": train_people, "reserved_final_subject": args.reserved_final_subject},
            )
        for test_subject in PEOPLE:
            train_people = [person for person in PEOPLE if person != test_subject]
            train = [row for row in task_rows if str(row.get("person")) in train_people]
            test = [row for row in task_rows if str(row.get("person")) == test_subject]
            fold_root = output / task / "final_refit" / f"test_{test_subject}"
            task_report["final_refit"][test_subject] = emit_fold(
                fold_root, train, None, test,
                {"task": task, "protocol": "final_refit", "test_subject": test_subject,
                 "train_subjects": train_people, "validation": "disabled"},
            )
        report["tasks"][task] = task_report
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "OK", "output": str(output),
        "t15_samples": len(tasks["t15"]), "t17_samples": len(tasks["t17"]),
        "take_put_samples": len(tasks["t17"]) - len(tasks["t15"]),
    }, indent=2))


if __name__ == "__main__":
    main()
