from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PACKAGE_ROOT / "config" / "experiment_plan.json"
SELECTION_PATH = PACKAGE_ROOT / "config" / "final_selection.json"

def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def load_plan() -> dict[str, Any]: return read_json(PLAN_PATH)
def platform_name(value: str) -> str: return ("windows" if os.name == "nt" else "hpc") if value == "auto" else value
def roots(plan, platform, project, dataset):
    p = platform_name(platform)
    a = Path(project or os.environ.get("PROJECT_ROOT") or plan["roots"][f"{p}_project"])
    b = Path(dataset or os.environ.get("DATASET_ROOT") or plan["roots"][f"{p}_dataset"])
    return a.resolve(), b.resolve()
def select(rows, index, experiment):
    matches = [x for x in rows if x["id"] == experiment] if experiment is not None else [x for x in rows if int(x["index"]) == int(index)]
    if len(matches) != 1: raise ValueError(f"Experiment selection is not unique: index={index}, experiment={experiment}")
    return matches[0]
def resolve_rgb(plan, exp, crop_stats: Path):
    task = plan["task"]
    if exp["rgb_source"] == "original": return {"camera": task["original_rgb_key"], "mean": task["original_mean"], "std": task["original_std"], "preserve": False}
    if exp["rgb_source"] != "motion_crop": raise ValueError(f"Unknown rgb_source={exp['rgb_source']}")
    if not crop_stats.is_file(): raise FileNotFoundError(f"Motion-crop statistics missing: {crop_stats}; run Stage 0 prepare first")
    stats = read_json(crop_stats)
    return {"camera": task["motion_crop_rgb_key"], "mean": stats["mean"], "std": stats["std"], "preserve": True}
def append(cmd: list[str], flag: str, value: Any):
    cmd.append(flag); cmd.extend(str(x) for x in value) if isinstance(value, (list, tuple)) else cmd.append(str(value))
