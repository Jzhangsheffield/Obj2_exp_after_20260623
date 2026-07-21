#!/usr/bin/env python3
"""Streaming RGB statistics and manifest audit for one packed .pt field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--rgb-key", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else args.dataset_root / args.manifest
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    samples = 0
    missing = []
    fallback_full_frame = 0
    aspect_ratios = []

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rel = rec.get(args.rgb_key)
            path = args.dataset_root / rel if rel else None
            if path is None or not path.is_file():
                missing.append({"sample_name": rec.get("sample_name"), "path": rel})
                continue
            obj = torch.load(path, map_location="cpu")
            video = obj["frames"] if isinstance(obj, dict) else obj
            if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
                raise ValueError(f"Invalid RGB tensor at {path}: {type(video)} / {getattr(video, 'shape', None)}")
            x = video.to(torch.float64)
            if video.dtype == torch.uint8:
                x = x / 255.0
            channel_sum += x.sum(dim=(0, 2, 3))
            channel_sq_sum += (x * x).sum(dim=(0, 2, 3))
            pixel_count += int(video.shape[0] * video.shape[2] * video.shape[3])
            samples += 1
            size = rec.get(args.rgb_key + "_size")
            if isinstance(size, list) and len(size) == 2 and float(size[1]) > 0:
                aspect_ratios.append(float(size[0]) / float(size[1]))
            if rec.get(args.rgb_key + "_motion_found") is False:
                fallback_full_frame += 1

    if missing:
        raise FileNotFoundError(f"{len(missing)} RGB files are missing; first entries: {missing[:5]}")
    if pixel_count == 0:
        raise RuntimeError("No RGB pixels were read")
    mean = channel_sum / pixel_count
    var = channel_sq_sum / pixel_count - mean.square()
    std = torch.sqrt(torch.clamp(var, min=0.0))
    aspect_ratios.sort()

    def percentile(q: float):
        if not aspect_ratios:
            return None
        return aspect_ratios[round(q * (len(aspect_ratios) - 1))]

    payload = {
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest": str(manifest.resolve()),
        "rgb_key": args.rgb_key,
        "num_samples": samples,
        "pixel_count_per_channel": pixel_count,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "fallback_full_frame": fallback_full_frame,
        "aspect_ratio": {
            "min": percentile(0.0), "p05": percentile(0.05),
            "median": percentile(0.5), "p95": percentile(0.95), "max": percentile(1.0)
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
