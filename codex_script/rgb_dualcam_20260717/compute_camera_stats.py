#!/usr/bin/env python3
"""Reproducibly estimate per-camera RGB mean/std from the training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="N_as_test/train_manifest_except_take_put.jsonl")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--spatial-stride", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    rows = [json.loads(x) for x in (root / args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    output = {}
    for camera in ("00143", "00152"):
        key = f"rgb_cam_{camera}"
        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0
        for row in rows:
            obj = torch.load(root / row[key], map_location="cpu")
            video = obj["frames"] if isinstance(obj, dict) else obj
            frame_ids = torch.linspace(0, video.shape[0] - 1, steps=min(args.frames_per_clip, video.shape[0])).round().long().unique()
            sample = video[frame_ids, :, ::args.spatial_stride, ::args.spatial_stride].double().div_(255.0)
            total += sample.sum((0, 2, 3))
            total_sq += (sample * sample).sum((0, 2, 3))
            count += sample.shape[0] * sample.shape[2] * sample.shape[3]
        mean = total / count
        std = (total_sq / count - mean.square()).clamp_min(0).sqrt()
        output[camera] = {"pixels_per_channel": count, "mean": mean.tolist(), "std": std.tolist()}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
