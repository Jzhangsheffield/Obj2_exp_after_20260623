#!/usr/bin/env python3
"""Join classifier predictions to protocol metadata and add run identity."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ["person", "action", "segment", "tier1", "tier2", "tier3", "lighting", "pos", "camera_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-id", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    by_key, by_name = {}, {}
    with args.manifest.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            by_key[str(row.get("original_key", ""))] = row
            by_name[str(row.get("sample_name", ""))] = row

    with args.input.open("r", newline="", encoding="utf-8-sig") as stream:
        predictions = list(csv.DictReader(stream))
    if not predictions:
        raise RuntimeError(f"No predictions in {args.input}")

    output = []
    unmatched = []
    for prediction in predictions:
        metadata = by_key.get(prediction.get("original_key", "")) or by_name.get(prediction.get("sample_name", ""))
        if metadata is None:
            unmatched.append(prediction.get("original_key") or prediction.get("sample_name") or "<unknown>")
            metadata = {}
        output.append({
            "config_id": args.config,
            "full_id": args.full_id,
            "fold": args.fold,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            **prediction,
            **{field: metadata.get(field, "") for field in FIELDS},
        })
    if unmatched:
        raise RuntimeError(f"Could not join {len(unmatched)} predictions to manifest; first={unmatched[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"output": str(args.output), "samples": len(output), "unmatched": 0}, indent=2))


if __name__ == "__main__":
    main()
