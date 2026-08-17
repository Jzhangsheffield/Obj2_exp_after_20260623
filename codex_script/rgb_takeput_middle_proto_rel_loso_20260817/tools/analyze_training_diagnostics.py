#!/usr/bin/env python3
"""Summarize old Proto/Rel state, gradient, update, feature and loss diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        out[prefix] = float(value)
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    debug = read_jsonl(args.pretrain_dir / "debug_train_log.jsonl")
    flat_debug = []
    by_epoch: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    nonfinite_events = []
    for row in debug:
        flat = flatten_numeric(row)
        epoch = int(row.get("epoch", flat.get("epoch", -1)))
        flat["epoch"] = epoch
        flat_debug.append(flat)
        for key, value in flat.items():
            if key != "epoch":
                by_epoch[epoch][key].append(value)
        check = row.get("nonfinite_check", {})
        if isinstance(check, dict) and check.get("has_nonfinite"):
            nonfinite_events.append(row)
    epoch_rows = []
    for epoch, metrics in sorted(by_epoch.items()):
        out = {"epoch": epoch, "debug_records": max((len(v) for v in metrics.values()), default=0)}
        for key, values in metrics.items():
            out[f"{key}.mean"] = mean(values)
            out[f"{key}.max"] = max(values)
        epoch_rows.append(out)

    proto_rows = []
    diag_dir = args.pretrain_dir / "prototype_diagnostics"
    for path in sorted(diag_dir.glob("*.json")) if diag_dir.is_dir() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_numeric(payload)
        flat["file"] = path.name
        proto_rows.append(flat)
    write_csv(args.output / "debug_records_flat.csv", flat_debug)
    write_csv(args.output / "debug_by_epoch.csv", epoch_rows)
    write_csv(args.output / "prototype_diagnostics_flat.csv", proto_rows)
    summary = {
        "pretrain_dir": str(args.pretrain_dir.resolve()),
        "debug_records": len(debug),
        "epochs_with_debug": sorted(by_epoch),
        "prototype_diagnostic_files": len(proto_rows),
        "nonfinite_event_count": len(nonfinite_events),
        "checkpoint_files": [path.name for path in sorted(args.pretrain_dir.glob("checkpoint_*.pth"))],
        "expected_checkpoint_files": ["checkpoint_0050.pth", "checkpoint_0100.pth", "checkpoint_0150.pth", "checkpoint_0200.pth"],
        "coverage": {
            "loss": any("loss" in key for row in flat_debug for key in row),
            "gradient": any("grad" in key for row in flat_debug for key in row),
            "parameter_update": any("param_update" in key for row in flat_debug for key in row),
            "features": any("feature" in key for row in flat_debug for key in row),
            "prototype": bool(proto_rows) or any("proto" in key for row in flat_debug for key in row),
        },
    }
    (args.output / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

