#!/usr/bin/env python3
"""Validate Python imports and the two backbone adapters without training."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True, type=Path)
    args = p.parse_args()
    src = args.src_root.resolve()
    sys.path.insert(0, str(src))

    for path in src.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    import torch
    import torchvision
    from backbone.video_backbone import generate_video_model
    from utils_.mapstype_dataloader_with_index import resolve_rgb_manifest_key

    r3d = generate_video_model("resnet3d18", num_classes=15)
    mvit = generate_video_model("mvit_v2_s", num_classes=15)
    checks = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "resnet_has_fc": hasattr(r3d, "fc"),
        "mvit_has_fc": hasattr(mvit, "fc"),
        "mvit_feature_dim": getattr(mvit, "feature_dim", None),
        "crop_manifest_key": resolve_rgb_manifest_key("00143_motion_crop_m32"),
    }
    if not checks["resnet_has_fc"] or not checks["mvit_has_fc"]:
        raise RuntimeError("Backbone adapter validation failed")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
