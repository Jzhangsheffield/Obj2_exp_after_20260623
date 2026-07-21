#!/usr/bin/env python
"""Extract test features for the three selected models per modality only."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
REPORT_OUT = ROOT / "analysis" / "obj2_experiment_report_20260719"
OLD_SCRIPT = ROOT / "analysis" / "obj2a_260716" / "scripts" / "pilot_features.py"


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("obj2_pilot_features", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT_ROOT = REPORT_OUT / "targeted_features"
    module.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    return module


def main() -> None:
    pilot = load_pilot_module()
    table = pd.read_csv(REPORT_OUT / "tables" / "selected_module_models.csv")
    specs = []
    for _, row in table.iterrows():
        specs.append(pilot.ModelSpec(
            model_id=str(row.feature_model_id),
            modality=str(row.modality),
            stage=str(row.finetune_mode) if row.model_role != "scratch" else "scratch",
            checkpoint=str(row.local_checkpoint_path),
            config=str(row.local_config_path),
            config_kind="ft",
            selection_metric="test_balanced_acc",
            selection_value=float(row.test_balanced_acc) if pd.notna(row.test_balanced_acc) else math.nan,
        ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot.log(f"Targeted feature extraction device: {device}; models={len(specs)}")
    for spec in specs:
        batch_size = 12 if spec.modality == "rgb" else 128
        pilot.extract_job(spec, "test", device, batch_size=batch_size, num_workers=4)
    print(f"Extracted or reused {len(specs)} targeted test feature sets in {pilot.OUT_ROOT}")


if __name__ == "__main__":
    main()
