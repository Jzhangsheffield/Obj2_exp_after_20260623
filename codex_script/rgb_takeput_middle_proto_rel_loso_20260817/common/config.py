from __future__ import annotations

import json
import os
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "experiment_config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get("RGB_EXP_CONFIG", DEFAULT_CONFIG)).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(config_path)
    cfg["_package_root"] = str(PACKAGE_ROOT)
    return cfg


def resolve_paths(cfg: dict[str, Any], platform: str, overrides: dict[str, str | None]) -> dict[str, Path | str]:
    if platform == "auto":
        platform = "windows" if os.name == "nt" else "hpc"
    if platform not in cfg["paths"]:
        raise ValueError(f"Unknown platform {platform!r}; choose one of {sorted(cfg['paths'])}")
    raw = deepcopy(cfg["paths"][platform])
    for key, value in overrides.items():
        if value:
            raw[key] = value
    return {
        "platform": platform,
        "project_root": Path(raw["project_root"]).expanduser().resolve(),
        "dataset_root": Path(raw["dataset_root"]).expanduser().resolve(),
        "results_root": Path(raw["results_root"]).expanduser().resolve(),
        "python_bin": str(raw["python_bin"]),
        "package_root": PACKAGE_ROOT,
        "config_path": Path(cfg["_config_path"]),
    }


def task_for_stage(stage: str) -> str:
    if stage.startswith("takeput_"):
        return "take_put"
    if stage.startswith("middle_"):
        return "middle"
    raise ValueError(f"Cannot infer task from stage {stage!r}")


def manifest_dir(paths: dict[str, Any], cfg: dict[str, Any], task: str, fold: str) -> Path:
    return paths["package_root"] / cfg["data"]["manifest_output_rel"] / task / fold


def experiment_row(cfg: dict[str, Any], stage: str, index: int | None, experiment_id: str | None):
    rows = cfg["experiment_grids"][stage]
    if experiment_id is not None:
        matches = [row for row in rows if row["id"] == experiment_id]
        if len(matches) != 1:
            raise ValueError(f"Experiment id {experiment_id!r} is not unique in {stage}")
        return deepcopy(matches[0])
    if index is None:
        raise ValueError("Specify --index or --experiment-id")
    if not 0 <= index < len(rows):
        raise IndexError(f"Index {index} outside 0..{len(rows)-1} for {stage}")
    return deepcopy(rows[index])


def command_text(command: list[str]) -> str:
    if os.name == "nt":
        return __import__("subprocess").list2cmdline(command)
    return shlex.join(command)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_project_imports(project_root: Path) -> None:
    for path in (project_root, PACKAGE_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

