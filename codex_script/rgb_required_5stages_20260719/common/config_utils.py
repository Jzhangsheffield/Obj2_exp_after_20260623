#!/usr/bin/env python3
"""Configuration, command-building, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    stage_ref = load_json(config_path)
    master_path = (config_path.parent / stage_ref.get("master_plan", "../../common/experiment_plan.json")).resolve()
    master = load_json(master_path)
    stage_id = stage_ref["stage_id"]
    if stage_id not in master["stages"]:
        raise KeyError("Unknown stage_id: %s" % stage_id)
    stage = master["stages"][stage_id]
    if stage_ref.get("source_overrides"):
        override_path = (config_path.parent / stage_ref["source_overrides"]).resolve()
        stage["source_candidates"] = load_json(override_path)["source_candidates"]
    if stage.get("auto_finetune") and not stage.get("finetune_experiments"):
        stage["finetune_experiments"] = [
            {
                "index": index,
                "id": row["id"] + "_ft",
                "pretrain_id": row["id"],
                "seed": row.get("seed", 1),
                "finetune_mode": "full",
                "backbone_lr": 0.0003,
                "head_lr": 0.001,
                "backbone_temporal_mode": row.get("backbone_temporal_mode", "current"),
            }
            for index, row in enumerate(stage.get("pretrain_experiments", []))
        ]
    if stage_id == "stage7" and not stage.get("finetune_experiments"):
        plans = []
        for source in stage.get("source_candidates", []):
            modes = stage["scratch_modes"] if source["id"] == "scratch" else stage["pretrained_modes"]
            for mode in modes:
                plan = {
                    "index": len(plans),
                    "id": source["id"] + "_" + mode["id"],
                    "pretrained_path_rel": source.get("pretrained_path_rel"),
                    "seed": 1,
                    "finetune_mode": mode["finetune_mode"],
                    "head_lr": 0.001,
                    "backbone_temporal_mode": source.get("backbone_temporal_mode", "current"),
                }
                if mode.get("backbone_lr") is not None:
                    plan["backbone_lr"] = mode["backbone_lr"]
                plans.append(plan)
        stage["finetune_experiments"] = plans
    return master, stage, stage_ref


def choose(rows: List[Dict[str, Any]], index: Optional[int], experiment_id: Optional[str]) -> Dict[str, Any]:
    if experiment_id is not None:
        matches = [row for row in rows if row["id"] == experiment_id]
    else:
        matches = [row for row in rows if int(row["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError("Selection is not unique: index=%r id=%r" % (index, experiment_id))
    return matches[0]


def roots(master: Dict[str, Any], project_root: Optional[str], dataset_root: Optional[str]) -> Tuple[Path, Path]:
    project = Path(project_root or os.environ.get("PROJECT_ROOT", master["project_root"]))
    dataset = Path(dataset_root or os.environ.get("DATASET_ROOT", master["dataset_root"]))
    return project, dataset


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(project: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_provenance(output: Path, payload: Dict[str, Any], project: Path, sources: List[Path]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["git_commit"] = git_commit(project)
    record["source_sha256"] = {str(path): file_sha256(path) for path in sources}
    record["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    record["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
    record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    (output / "codex_run_provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def resolve_pretrained(project: Path, master: Dict[str, Any], stage: Dict[str, Any], plan: Dict[str, Any]) -> Optional[Path]:
    if plan.get("pretrained_path_rel"):
        return project / plan["pretrained_path_rel"]
    if plan.get("pretrain_id"):
        return project / stage["pretrain_output_rel"] / plan["pretrain_id"] / (
            "checkpoint_%04d.pth" % int(master["pretrain_common"]["epochs"])
        )
    return None
