#!/usr/bin/env python3
"""Summarize validation-only fine-tuning outcomes; never reads the test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from config_utils import load_stage, roots


def family_id(run_id: str) -> str:
    value = run_id[:-3] if run_id.endswith("_ft") else run_id
    return re.sub(r"_s[123]$", "", value)


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    args = parser.parse_args()
    master, stage, _ = load_stage(Path(args.config).resolve())
    project, _ = roots(master, args.project_root, None)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to read fine-tuning checkpoints") from exc

    rows = []
    ft_root = project / stage["finetune_output_rel"]
    for exp in stage.get("finetune_experiments", []):
        weight_root = ft_root / "weights" / exp["id"]
        matches = sorted(weight_root.rglob("best_val_balanced.pth")) if weight_root.is_dir() else []
        checkpoint = matches[-1] if matches else weight_root / "best_val_balanced.pth"
        row = {
            "experiment_id": exp["id"], "family": family_id(exp["id"]),
            "seed": exp.get("seed", 1), "status": "missing", "checkpoint": str(checkpoint),
            "epoch": "", "val_acc_pct": "", "val_balanced_acc_pct": "", "val_macro_f1_pct": "",
        }
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            metrics = payload.get("extra_info", payload)
            row["status"] = "complete"
            row["epoch"] = payload.get("epoch", "")
            for source, target in (("val_acc", "val_acc_pct"), ("val_balanced_acc", "val_balanced_acc_pct"), ("val_macro_f1", "val_macro_f1_pct")):
                value = finite(metrics.get(source))
                row[target] = 100.0 * value if value is not None else ""
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    families = []
    for name, items in sorted(grouped.items()):
        vals = [finite(x["val_balanced_acc_pct"]) for x in items]
        vals = [x for x in vals if x is not None]
        f1s = [finite(x["val_macro_f1_pct"]) for x in items]
        f1s = [x for x in f1s if x is not None]
        families.append({
            "family": name, "completed_seeds": len(vals), "expected_seeds": len(items),
            "mean_val_balanced_acc_pct": statistics.mean(vals) if vals else "",
            "std_val_balanced_acc_pp": statistics.stdev(vals) if len(vals) > 1 else "",
            "mean_val_macro_f1_pct": statistics.mean(f1s) if f1s else "",
        })

    output = ft_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for path, data in ((output / "validation_runs.csv", rows), (output / "validation_family_summary.csv", families)):
        fields = list(data[0].keys()) if data else ["status"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    lines = [f"# {stage['id']} validation summary", "", "This report uses validation checkpoints only; it does not read the locked test set.", "", "| family | complete | balanced accuracy mean ± SD (pp) | macro-F1 mean (%) |", "|---|---:|---:|---:|"]
    for row in families:
        mean, sd, f1 = row["mean_val_balanced_acc_pct"], row["std_val_balanced_acc_pp"], row["mean_val_macro_f1_pct"]
        metric = "missing" if mean == "" else f"{mean:.3f} ± {sd:.3f}" if sd != "" else f"{mean:.3f}"
        f1_text = "missing" if f1 == "" else f"{f1:.3f}"
        lines.append(f"| {row['family']} | {row['completed_seeds']}/{row['expected_seeds']} | {metric} | {f1_text} |")
    (output / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(families, indent=2, ensure_ascii=False))
    print(f"Wrote validation summaries under {output}")


if __name__ == "__main__":
    main()
