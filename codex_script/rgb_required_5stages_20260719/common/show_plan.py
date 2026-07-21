#!/usr/bin/env python3
"""Print the fully resolved experiment table, including generated FT grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config_utils import load_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kind", choices=["pretrain", "finetune", "all"], default="all")
    args = parser.parse_args()
    _, stage, _ = load_stage(Path(args.config).resolve())
    payload = {"stage": stage["id"]}
    if args.kind in ("pretrain", "all"):
        payload["pretrain_experiments"] = stage.get("pretrain_experiments", [])
    if args.kind in ("finetune", "all"):
        payload["finetune_experiments"] = stage.get("finetune_experiments", [])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
