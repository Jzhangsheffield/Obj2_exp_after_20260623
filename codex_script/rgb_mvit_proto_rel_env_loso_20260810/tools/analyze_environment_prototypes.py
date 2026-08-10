#!/usr/bin/env python3
"""Measure whether per-class prototype assignments align with lighting."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def purity(assignments: list[int], environments: list[str]) -> float:
    buckets: dict[int, Counter] = defaultdict(Counter)
    for proto, env in zip(assignments, environments):
        buckets[int(proto)][env] += 1
    total = sum(sum(counts.values()) for counts in buckets.values())
    return sum(max(counts.values()) for counts in buckets.values()) / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    states = sorted(args.diagnostic_dir.glob("proto_state_epoch_*.pt"))
    args.output.mkdir(parents=True, exist_ok=True)
    if not states:
        (args.output / "environment_prototype_status.json").write_text(
            json.dumps({"status": "not_applicable_or_missing", "diagnostic_dir": str(args.diagnostic_dir)}, indent=2), encoding="utf-8"
        )
        print("No prototype state files; wrote not-applicable status")
        return

    summary_rows: list[dict] = []
    final_contingency: list[dict] = []
    for state_path in states:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assignment = state.get("sample_to_proto")
        sample_class = state.get("sample_to_class")
        valid = state.get("valid_sample_mask")
        if assignment is None or len(assignment) != len(rows):
            raise ValueError(f"Assignment length mismatch in {state_path}: {None if assignment is None else len(assignment)} vs {len(rows)}")
        assignment = assignment.long().numpy()
        sample_class = sample_class.long().numpy() if sample_class is not None else np.array([-1] * len(rows))
        valid = valid.bool().numpy() if valid is not None else assignment >= 0
        epoch = int(state.get("epoch", state_path.stem.rsplit("_", 1)[-1]))
        per_class = []
        for class_name in sorted({str(row.get("tier1")) for row in rows}):
            idx = [i for i, row in enumerate(rows) if str(row.get("tier1")) == class_name and valid[i] and assignment[i] >= 0]
            if len(idx) < 2:
                continue
            a = [int(assignment[i]) for i in idx]
            e = [str(rows[i].get("lighting")) for i in idx]
            nmi = normalized_mutual_info_score(e, a)
            ari = adjusted_rand_score(e, a)
            pur = purity(a, e)
            per_class.append((nmi, ari, pur))
            if state_path == states[-1]:
                counts = Counter((int(assignment[i]), str(rows[i].get("lighting"))) for i in idx)
                for (proto, lighting), count in sorted(counts.items()):
                    final_contingency.append({"class": class_name, "prototype": proto, "lighting": lighting, "count": count})
        summary_rows.append({
            "epoch": epoch,
            "valid_samples": int(valid.sum()),
            "mean_class_nmi": float(np.mean([x[0] for x in per_class])) if per_class else None,
            "mean_class_ari": float(np.mean([x[1] for x in per_class])) if per_class else None,
            "mean_class_purity": float(np.mean([x[2] for x in per_class])) if per_class else None,
            "classes_analyzed": len(per_class),
            "assignment_ids_used": int(len(set(int(x) for x in assignment[valid] if x >= 0))),
        })

    for filename, records in (("environment_prototype_summary.csv", summary_rows), ("final_environment_contingency.csv", final_contingency)):
        with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["status"])
            writer.writeheader()
            if records:
                writer.writerows(records)
    (args.output / "environment_prototype_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_rows[-1], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
