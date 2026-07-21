#!/usr/bin/env python3
"""Rank dual-camera test results and report camera generalization gaps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config" / "dualcam_config.json"))
    parser.add_argument("--project-root")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    summary = root / cfg["finetune_output_rel"] / "weights" / "_batch_test" / "dualcam" / "summary" / "rgb_dualcam_test_results.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    df = pd.read_csv(summary, dtype={"rgb_camera_id": str})
    df["dualcam_task"] = df["weight_path"].astype(str).map(
        lambda p: next((x["id"] for x in cfg["finetune_plan"] if x["id"] in p), "unknown")
    )
    metric = "test_balanced_acc"
    ranked = df.sort_values(["rgb_camera_id", metric, "test_acc"], ascending=[True, False, False])
    ranked_path = summary.with_name("rgb_dualcam_test_results_ranked.csv")
    ranked.to_csv(ranked_path, index=False)

    pivot = df.pivot_table(index="dualcam_task", columns="rgb_camera_id", values=metric, aggfunc="max")
    if "00143" in pivot and "00152" in pivot:
        pivot["mean_balanced_acc"] = pivot[["00143", "00152"]].mean(axis=1)
        pivot["cam152_minus_cam143"] = pivot["00152"] - pivot["00143"]
        pivot = pivot.sort_values("mean_balanced_acc", ascending=False)
    gap_path = summary.with_name("rgb_dualcam_camera_gap.csv")
    pivot.to_csv(gap_path)
    print(ranked[[c for c in ["dualcam_task", "rgb_camera_id", "test_acc", metric, "test_macro_f1"] if c in ranked]].to_string(index=False))
    print(f"\nRanked: {ranked_path}\nCamera gap: {gap_path}")


if __name__ == "__main__":
    main()
