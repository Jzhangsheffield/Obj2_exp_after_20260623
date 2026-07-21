#!/usr/bin/env python3
"""Rank completed validation-balanced checkpoints without touching test data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config_utils import load_stage, roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    args = parser.parse_args()
    master, stage, _ = load_stage(Path(args.config).resolve())
    project, _ = roots(master, args.project_root, None)
    weights_root = project / stage["finetune_output_rel"] / "weights"
    rows = []
    for plan in stage.get("finetune_experiments", []):
        matches = sorted((weights_root / plan["id"]).rglob("best_val_balanced.pth")) if (weights_root / plan["id"]).is_dir() else []
        rows.append({"experiment_id": plan["id"], "checkpoint_count": len(matches),
                     "best_val_balanced_checkpoint": str(matches[0]) if len(matches) == 1 else ""})
    out = weights_root / "validation_selected_checkpoints.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["experiment_id"])
        writer.writeheader()
        writer.writerows(rows)
    print("Validation-selected checkpoint manifest: %s" % out)


if __name__ == "__main__":
    main()

