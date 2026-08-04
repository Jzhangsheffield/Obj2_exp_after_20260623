from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ref = load_json(config_path)
    master = load_json((config_path.parent / ref["master_plan"]).resolve())
    stage = master["stages"][ref["stage_id"]]
    if stage.get("auto_finetune") and not stage.get("finetune_experiments"):
        stage["finetune_experiments"] = [
            {"index": i, "id": row["id"] + "_ft", "pretrain_id": row["id"],
             "seed": row.get("seed", 1), "finetune_mode": "full",
             "backbone_lr": 0.0003, "head_lr": 0.001}
            for i, row in enumerate(stage.get("pretrain_experiments", []))
            if row.get("run_finetune", True)
        ]
    return master, stage, ref


def choose(rows: List[Dict[str, Any]], index: Optional[int], experiment_id: Optional[str]) -> Dict[str, Any]:
    matches = [r for r in rows if r["id"] == experiment_id] if experiment_id else [r for r in rows if int(r["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError(f"Selection is not unique: index={index!r}, id={experiment_id!r}")
    return matches[0]


def roots(master: Dict[str, Any], project_root: Optional[str], dataset_root: Optional[str]) -> Tuple[Path, Path]:
    return Path(project_root or os.environ.get("PROJECT_ROOT", master["project_root"])), Path(dataset_root or os.environ.get("DATASET_ROOT", master["dataset_root"]))


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(output: Path, payload: Dict[str, Any], project: Path, sources: List[Path]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    record = dict(payload)
    record.update({"git_commit": commit, "source_sha256": {str(p): file_sha256(p) for p in sources}, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")})
    (output / "codex_run_provenance.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

