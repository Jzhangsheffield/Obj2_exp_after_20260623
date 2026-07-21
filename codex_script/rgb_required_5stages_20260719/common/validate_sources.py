#!/usr/bin/env python3
"""Freeze/check Stage-7 pretrained source checkpoints before fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from config_utils import load_stage, roots


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    master, stage, _ = load_stage(Path(args.config).resolve())
    project, _ = roots(master, args.project_root, None)
    rows = []
    missing = []
    for source in stage.get("source_candidates", []):
        rel = source.get("pretrained_path_rel")
        temporal_mode = source.get("backbone_temporal_mode", "current")
        if rel is None:
            rows.append({"id": source["id"], "backbone_temporal_mode": temporal_mode,
                         "path": None, "sha256": None, "status": "scratch"})
            continue
        path = project / rel
        if not path.is_file():
            missing.append(str(path))
            rows.append({"id": source["id"], "backbone_temporal_mode": temporal_mode,
                         "path": str(path), "sha256": None, "status": "missing"})
        else:
            rows.append({"id": source["id"], "backbone_temporal_mode": temporal_mode,
                         "path": str(path), "sha256": sha256(path), "status": "ok"})
    out = project / stage["finetune_output_rel"] / "selected_pretrain_sources.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing Stage-7 sources:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
