#!/usr/bin/env python3
"""Stage-0 audit: compare final encoder weights for SupLoss and zero-weight branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config_utils import load_stage, roots


def state_dict(payload):
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict):
            return value
    return payload if isinstance(payload, dict) else {}


def compare(left, right):
    common = sorted(set(left) & set(right))
    tensors = 0; total_sq = 0.0; max_abs = 0.0
    for key in common:
        a, b = left[key], right[key]
        if not hasattr(a, "shape") or a.shape != b.shape:
            continue
        diff = (a.detach().float().cpu() - b.detach().float().cpu())
        tensors += 1; total_sq += float((diff * diff).sum()); max_abs = max(max_abs, float(diff.abs().max()))
    return {"common_tensors": tensors, "l2_difference": total_sq ** 0.5, "max_abs_difference": max_abs}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--project-root")
    args = parser.parse_args(); master, stage, _ = load_stage(Path(args.config).resolve()); project, _ = roots(master, args.project_root, None)
    if stage["id"] != "stage0": raise SystemExit("audit_null_paths.py is only for stage0")
    import torch
    root = project / stage["pretrain_output_rel"]
    checkpoints = {}
    for exp in stage["pretrain_experiments"]:
        epochs = int(exp.get("epochs", master["pretrain_common"]["epochs"]))
        path = root / exp["id"] / f"checkpoint_{epochs:04d}.pth"
        if path.is_file():
            try: payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError: payload = torch.load(path, map_location="cpu")
            checkpoints[exp["id"]] = state_dict(payload)
    baseline = stage["pretrain_experiments"][0]["id"]
    results = {"baseline": baseline, "interpretation": "Exact equality is expected only if the disabled auxiliary branch does not perturb RNG/state; any difference must be investigated before Stage 1.", "comparisons": []}
    for exp in stage["pretrain_experiments"][1:]:
        item = {"experiment_id": exp["id"], "status": "missing"}
        if baseline in checkpoints and exp["id"] in checkpoints:
            item.update(compare(checkpoints[baseline], checkpoints[exp["id"]])); item["status"] = "identical" if item["max_abs_difference"] == 0.0 else "different"
        results["comparisons"].append(item)
    out = root / "analysis"; out.mkdir(parents=True, exist_ok=True)
    (out / "null_path_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Stage 0 null-path audit", "", results["interpretation"], "", "| comparison | status | L2 difference | max absolute difference |", "|---|---|---:|---:|"]
    for item in results["comparisons"]: lines.append(f"| {baseline} vs {item['experiment_id']} | {item['status']} | {item.get('l2_difference', '')} | {item.get('max_abs_difference', '')} |")
    (out / "null_path_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2)); print(f"Wrote {out / 'null_path_audit.md'}")


if __name__ == "__main__": main()
