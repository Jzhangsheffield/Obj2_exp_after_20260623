from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PACKAGE_ROOT / "config" / "experiment_plan.json"
SELECTION_PATH = PACKAGE_ROOT / "config" / "selection.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan() -> dict[str, Any]:
    return read_json(PLAN_PATH)


def platform_name(value: str) -> str:
    return ("windows" if os.name == "nt" else "hpc") if value == "auto" else value


def roots(plan: dict, platform: str, project: str | None, dataset: str | None) -> tuple[Path, Path]:
    key = platform_name(platform)
    project_path = Path(project or os.environ.get("PROJECT_ROOT") or plan["roots"][f"{key}_project"])
    dataset_path = Path(dataset or os.environ.get("DATASET_ROOT") or plan["roots"][f"{key}_dataset"])
    return project_path.resolve(), dataset_path.resolve()


def append(cmd: list[str], flag: str, value: Any) -> None:
    cmd.append(flag)
    if isinstance(value, (list, tuple)):
        cmd.extend(str(x) for x in value)
    else:
        cmd.append(str(value))


def select(rows: list[dict], index: int | None, experiment: str | None) -> dict:
    matches = [x for x in rows if x["id"] == experiment] if experiment is not None else [x for x in rows if int(x["index"]) == int(index)]
    if len(matches) != 1:
        raise ValueError(f"Experiment selection is not unique: index={index}, experiment={experiment}")
    return matches[0]
