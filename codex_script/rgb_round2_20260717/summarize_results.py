#!/usr/bin/env python3
"""Create a compact ranked CSV from round-2 batch-test output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config" / "round2_config.json"))
    parser.add_argument("--project-root")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.project_root or os.environ.get("PROJECT_ROOT", cfg["project_root"]))
    summary = root / cfg["finetune_output_rel"] / "weights" / "_batch_test" / "round2_fixed" / "summary" / "rgb_test_results.csv"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    df = pd.read_csv(summary)
    path_col = "weight_path"
    df["round2_task"] = df[path_col].astype(str).map(
        lambda p: next((x["id"] for x in cfg["finetune_plan"] if x["id"] in p), "unknown")
    )
    keep = [c for c in ["round2_task", "weight_name", "test_acc", "test_balanced_acc", "test_macro_f1", "test_loss", "num_samples", path_col] if c in df]
    ranked = df[keep].sort_values(["test_balanced_acc", "test_acc"], ascending=False)
    out = summary.with_name("rgb_test_results_ranked.csv")
    ranked.to_csv(out, index=False)
    print(ranked.to_string(index=False))
    print(f"\nRanked CSV: {out}")


if __name__ == "__main__":
    main()
