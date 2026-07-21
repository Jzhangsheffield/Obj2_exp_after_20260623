#!/usr/bin/env python3
"""Audit dual-camera availability and frame-count alignment in JSONL manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def inspect_manifest(dataset_root: Path, manifest: Path, check_files: bool = False) -> dict:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing_143 = missing_152 = missing_files = 0
    diffs = []
    for row in rows:
        p143, p152 = row.get("rgb_cam_00143"), row.get("rgb_cam_00152")
        missing_143 += int(not p143)
        missing_152 += int(not p152)
        if p143 and p152:
            diffs.append(abs(int(row["num_rgb_cam_00143_frames"]) - int(row["num_rgb_cam_00152_frames"])))
        if check_files:
            missing_files += int(bool(p143) and not (dataset_root / p143).is_file())
            missing_files += int(bool(p152) and not (dataset_root / p152).is_file())
    counts = Counter(diffs)
    report = {
        "manifest": str(manifest),
        "samples": len(rows),
        "missing_cam00143": missing_143,
        "missing_cam00152": missing_152,
        "missing_tensor_files": missing_files if check_files else None,
        "equal_frame_counts": counts[0],
        "difference_le_1": sum(v for k, v in counts.items() if k <= 1),
        "difference_le_2": sum(v for k, v in counts.items() if k <= 2),
        "max_frame_difference": max(diffs) if diffs else None,
        "mean_frame_difference": sum(diffs) / len(diffs) if diffs else None,
        "frame_difference_histogram": dict(sorted(counts.items())),
        "alignment_assumption": "The tensors contain no timestamps; equal normalized positions within each segmented clip are treated as synchronized.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", action="append")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifests = args.manifest or [
        "N_as_test/train_manifest_except_take_put.jsonl",
        "N_as_test/val_manifest_except_take_put.jsonl",
        "N_as_test/test_manifest_except_take_put.jsonl",
    ]
    reports = [inspect_manifest(root, root / item, args.check_files) for item in manifests]
    print(json.dumps(reports, indent=2))
    for report in reports:
        if report["missing_cam00143"] or report["missing_cam00152"] or (report["missing_tensor_files"] or 0):
            raise SystemExit("Dual-camera manifest audit failed")


if __name__ == "__main__":
    main()
