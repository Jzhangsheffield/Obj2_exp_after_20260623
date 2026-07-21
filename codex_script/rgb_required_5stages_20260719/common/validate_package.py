#!/usr/bin/env python3
"""Static/package validation plus parse-only validation of generated commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config_utils import load_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--commands", action="store_true", help="also parse one pretrain, fine-tune and test command")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    master, stage, _ = load_stage(config_path)
    package = Path(__file__).resolve().parents[1]
    for script in (package / "common").glob("*.py"):
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    try:
        import torch
        from proto_loss_modes import prototype_contrastive_loss_soft_responsibility
        q = torch.randn(4, 8, requires_grad=True)
        bank = torch.randn(3, 2, 8)
        loss = prototype_contrastive_loss_soft_responsibility(
            q=q,
            labels=torch.tensor([0, 1, 2, 0]),
            proto_ids=torch.tensor([0, 1, 0, 1]),
            prototype_bank=bank,
            class_num_prototypes=torch.tensor([2, 2, 2]),
        )
        loss.backward()
        if not bool(torch.isfinite(loss)) or q.grad is None or not bool(torch.isfinite(q.grad).all()):
            raise FloatingPointError("soft-responsibility loss unit check is non-finite")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
    pre = stage.get("pretrain_experiments", [])
    ft = stage.get("finetune_experiments", [])
    pre_indices = [int(row["index"]) for row in pre]
    ft_indices = [int(row["index"]) for row in ft]
    if pre_indices != list(range(len(pre_indices))):
        raise ValueError("Pretrain indices must be contiguous from zero")
    if ft_indices != list(range(len(ft_indices))):
        raise ValueError("Fine-tune indices must be contiguous from zero")
    ids = [row["id"] for row in pre] + [row["id"] for row in ft]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment IDs must be unique within a stage")
    for row in pre:
        if row.get("backbone_temporal_mode", "current") not in ("current", "t3_lfb"):
            raise ValueError("Unknown backbone temporal mode: %s" % row["id"])
        if row["ablation_mode"] == "contrastive_only" and (row.get("lambda_proto", 0) or row.get("lambda_rel", 0)):
            raise ValueError("contrastive_only row has nonzero auxiliary weight: %s" % row["id"])
        if row.get("rel_same_weight", 1) == 0 and row.get("rel_diff_weight", 1) == 0 and row.get("lambda_rel", 0) > 0:
            raise ValueError("Relation loss has both components disabled: %s" % row["id"])
    pre_by_id = {row["id"]: row for row in pre}
    for row in ft:
        mode = row.get("backbone_temporal_mode", "current")
        if mode not in ("current", "t3_lfb"):
            raise ValueError("Unknown fine-tune temporal mode: %s" % row["id"])
        source_id = row.get("pretrain_id")
        if source_id in pre_by_id and mode != pre_by_id[source_id].get("backbone_temporal_mode", "current"):
            raise ValueError("Pretrain/fine-tune temporal mode mismatch: %s" % row["id"])
    if args.commands:
        common = package / "common"
        project = args.project_root or master["project_root"]
        if pre:
            subprocess.run([args.python_bin, str(common / "run_pretrain.py"), "--config", str(config_path),
                            "--index", "0", "--project-root", project, "--validate-command"], check=True)
        if ft:
            subprocess.run([args.python_bin, str(common / "run_finetune.py"), "--config", str(config_path),
                            "--index", "0", "--project-root", project, "--validate-command"], check=True)
            subprocess.run([args.python_bin, str(common / "run_test.py"), "--config", str(config_path),
                            "--index", "0", "--project-root", project, "--validate-command"], check=True)
    print(json.dumps({"stage": stage["id"], "pretrain_count": len(pre), "finetune_count": len(ft), "status": "OK"}, indent=2))


if __name__ == "__main__":
    main()
