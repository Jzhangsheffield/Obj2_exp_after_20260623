"""Refresh only the targeted feature diagnostics and UMAP figures.

This lightweight entry point reuses the already-built model-selection tables so
that feature-space figures can be regenerated without rescanning every run.
"""
from pathlib import Path
import importlib.util
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extended_module_analysis", HERE / "extended_module_analysis.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

selected_models = pd.read_csv(module.TABLES / "selected_module_models.csv")
per_class = pd.read_csv(module.TABLES / "selected_module_per_class.csv")
feature_diag, harmed = module.targeted_feature_analysis(selected_models, per_class)
feature_diag.to_csv(module.TABLES / "targeted_feature_diagnostics.csv", index=False, encoding="utf-8-sig")
harmed.to_csv(module.TABLES / "harmed_class_selection.csv", index=False, encoding="utf-8-sig")
print(f"refreshed feature diagnostics: {len(feature_diag)} rows; harmed classes: {len(harmed)}")
