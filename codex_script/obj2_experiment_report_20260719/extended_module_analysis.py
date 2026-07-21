#!/usr/bin/env python
"""Extended module-centric analysis for Prototype and Relation losses.

The script is intentionally retrospective and descriptive: all training runs use
seed=1, so configuration rows are not treated as independent statistical
replicates. Paired bootstrap intervals quantify test-sample uncertainty only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "obj2_experiment_report_20260719"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
FEATURE_ROOT = OUT / "targeted_features" / "01_features"

METHOD_ORDER = ["scratch", "supcon", "prototype", "relation", "prototype+relation"]
METHOD_LABEL = {
    "scratch": "Scratch",
    "supcon": "SupCon",
    "prototype": "SupCon+Proto",
    "relation": "SupCon+Rel",
    "prototype+relation": "SupCon+Proto+Rel",
}
ROLE_LABEL = {"scratch": "Scratch", "supcon": "SupCon", "module": "Best module"}
ROLE_COLOR = {"scratch": "#64748B", "supcon": "#2F6B8A", "module": "#D9772A"}
FAMILY_ORDER = ["base", "random_kqueue", "relation_topk", "sampler", "stage5_topk", "depth10", "round2", "dualcam", "take_put"]
FAMILY_LABEL = {
    "base": "Base losses", "random_kqueue": "Queue K", "relation_topk": "Relation Top-k",
    "sampler": "Balanced sampler", "stage5_topk": "Early loss + Top-k", "depth10": "ResNet-10",
    "round2": "RGB Round-2", "dualcam": "Dual camera", "take_put": "Take/Put",
}
MODULE_FAMILIES_WITH_EXACT_SUPCON = {"base", "random_kqueue", "sampler", "depth10"}
CLASS_THRESHOLD = 0.05
N_BOOT = 2000
SEED = 260720

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "figure.dpi": 150,
})


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def local_path(value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def method_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["dataset_scope", "family", "family_label", "modality", "finetune_mode", "loss_group"]
    for key, group in selected.groupby(keys, dropna=False):
        group = group.sort_values("test_balanced_acc", ascending=False)
        best = group.iloc[0]
        dataset_scope, family, family_label, modality, mode, method = key
        scratch = selected[
            (selected.dataset_scope == dataset_scope) & (selected.family == family) &
            (selected.modality == modality) & (selected.finetune_mode == "scratch")
        ]
        supcon = selected[
            (selected.dataset_scope == dataset_scope) & (selected.family == family) &
            (selected.modality == modality) & (selected.finetune_mode == mode) &
            (selected.loss_group == "supcon")
        ]
        scratch_score = float(scratch.test_balanced_acc.max()) if len(scratch) else math.nan
        supcon_score = float(supcon.test_balanced_acc.max()) if len(supcon) else math.nan
        vals = group.test_balanced_acc.astype(float)
        rows.append({
            "dataset_scope": dataset_scope, "family": family, "family_label": family_label,
            "modality": modality, "finetune_mode": mode, "method": method,
            "method_label": METHOD_LABEL.get(method, str(method)), "n_configs": len(group),
            "mean_balanced_acc": vals.mean(), "median_balanced_acc": vals.median(),
            "std_balanced_acc": vals.std(ddof=1) if len(vals) > 1 else math.nan,
            "best_balanced_acc": vals.max(), "worst_balanced_acc": vals.min(),
            "best_accuracy": float(best.test_acc), "best_macro_f1": float(best.test_macro_f1),
            "best_source_key": best.source_key, "best_run_name": best.run_name,
            "best_checkpoint": best.checkpoint_selection,
            "scratch_reference": scratch_score, "supcon_reference_best": supcon_score,
            "best_delta_vs_scratch": vals.max() - scratch_score if np.isfinite(scratch_score) else math.nan,
            "best_delta_vs_supcon": vals.max() - supcon_score if np.isfinite(supcon_score) else math.nan,
            "median_delta_vs_scratch": vals.median() - scratch_score if np.isfinite(scratch_score) else math.nan,
            "median_delta_vs_supcon": vals.median() - supcon_score if np.isfinite(supcon_score) else math.nan,
            "win_rate_vs_scratch": float((vals > scratch_score).mean()) if np.isfinite(scratch_score) else math.nan,
            "win_rate_vs_best_supcon": float((vals > supcon_score).mean()) if np.isfinite(supcon_score) else math.nan,
        })
    return pd.DataFrame(rows)


def sampler_scheme(source_key: str) -> str:
    s = str(source_key).lower()
    if "balanced_batch" in s:
        return "balanced_batch"
    if "sampler_random" in s:
        return "random"
    return "default"


def exact_supcon_match(row: pd.Series, selected: pd.DataFrame) -> pd.DataFrame:
    q = selected[
        (selected.dataset_scope == row.dataset_scope) & (selected.family == row.family) &
        (selected.modality == row.modality) & (selected.finetune_mode == row.finetune_mode) &
        (selected.loss_group == "supcon")
    ].copy()
    if not len(q):
        return q
    if row.family == "random_kqueue":
        q = q[pd.to_numeric(q.queue_size_name, errors="coerce") == pd.to_numeric(pd.Series([row.queue_size_name]), errors="coerce").iloc[0]]
    elif row.family == "sampler":
        q = q[q.source_key.map(sampler_scheme) == sampler_scheme(row.source_key)]
        if pd.notna(row.queue_size_name):
            q = q[pd.to_numeric(q.queue_size_name, errors="coerce") == float(row.queue_size_name)]
    return q.sort_values("test_balanced_acc", ascending=False)


def matched_pairs(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    candidates = selected[
        selected.loss_group.isin(["prototype", "relation", "prototype+relation"]) &
        selected.family.isin(MODULE_FAMILIES_WITH_EXACT_SUPCON | {"take_put"})
    ]
    for _, row in candidates.iterrows():
        match = exact_supcon_match(row, selected)
        if not len(match):
            continue
        base = match.iloc[0]
        records.append({
            "dataset_scope": row.dataset_scope, "family": row.family, "modality": row.modality,
            "finetune_mode": row.finetune_mode, "method": row.loss_group,
            "module_source_key": row.source_key, "supcon_source_key": base.source_key,
            "queue_size": row.queue_size_name, "sampler_scheme": sampler_scheme(row.source_key),
            "module_balanced_acc": row.test_balanced_acc, "supcon_balanced_acc": base.test_balanced_acc,
            "delta_vs_matched_supcon": row.test_balanced_acc - base.test_balanced_acc,
            "module_macro_f1": row.test_macro_f1, "supcon_macro_f1": base.test_macro_f1,
            "macro_f1_delta": row.test_macro_f1 - base.test_macro_f1,
            "module_checkpoint": row.checkpoint_selection, "supcon_checkpoint": base.checkpoint_selection,
        })
    return pd.DataFrame(records)


def choose_selected_models(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base_data = selected[selected.dataset_scope == "except_take_put"]
    for modality in ["rgb", "emg", "imu"]:
        candidates = base_data[
            (base_data.modality == modality) &
            base_data.loss_group.isin(["prototype", "relation", "prototype+relation"]) &
            base_data.family.isin(MODULE_FAMILIES_WITH_EXACT_SUPCON)
        ].sort_values("test_balanced_acc", ascending=False)
        selected_module = None
        selected_supcon = None
        for _, candidate in candidates.iterrows():
            match = exact_supcon_match(candidate, base_data)
            if len(match):
                selected_module = candidate
                selected_supcon = match.iloc[0]
                break
        if selected_module is None:
            raise RuntimeError(f"No module/SupCon matched candidate for {modality}")
        scratch = base_data[
            (base_data.family == selected_module.family) & (base_data.modality == modality) &
            (base_data.finetune_mode == "scratch")
        ].sort_values("test_balanced_acc", ascending=False).iloc[0]
        for role, row in [("module", selected_module), ("supcon", selected_supcon), ("scratch", scratch)]:
            checkpoint_path = local_path(row.run_dir) / f"{row.checkpoint_selection}.pth"
            config_path = local_path(row.config_path)
            rec = row.to_dict()
            rec.update({
                "model_role": role, "model_role_label": ROLE_LABEL[role],
                "feature_model_id": f"{modality}_{role}",
                "local_checkpoint_path": str(checkpoint_path),
                "local_config_path": str(config_path),
                "checkpoint_exists": checkpoint_path.is_file(), "config_exists_local": config_path.is_file(),
            })
            records.append(rec)
    return pd.DataFrame(records)


def load_predictions(row: pd.Series) -> pd.DataFrame:
    p = local_path(row.per_sample_path)
    df = pd.read_csv(p)
    keep = ["sample_name", "true_label_name", "pred_label_name", "pred_confidence", "correct"]
    return df[keep].copy()


def align_predictions(rows_by_role: dict[str, pd.Series]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for role, row in rows_by_role.items():
        d = load_predictions(row).rename(columns={
            "true_label_name": f"true_{role}", "pred_label_name": f"pred_{role}",
            "pred_confidence": f"confidence_{role}", "correct": f"correct_{role}",
        })
        merged = d if merged is None else merged.merge(d, on="sample_name", how="inner", validate="one_to_one")
    assert merged is not None
    true_cols = [f"true_{r}" for r in rows_by_role]
    if not merged[true_cols].nunique(axis=1).eq(1).all():
        raise RuntimeError("True labels differ across aligned model predictions")
    merged["true_label_name"] = merged[true_cols[0]]
    return merged


def stratified_bootstrap_delta(df: pd.DataFrame, role_a: str, role_b: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    by_class = [g.index.to_numpy() for _, g in df.groupby("true_label_name")]
    observed = float(np.mean([
        (df.loc[idx, f"correct_{role_a}"].to_numpy() - df.loc[idx, f"correct_{role_b}"].to_numpy()).mean()
        for idx in by_class
    ]))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        values = []
        for idx in by_class:
            draw = rng.choice(idx, size=len(idx), replace=True)
            values.append((df.loc[draw, f"correct_{role_a}"].to_numpy() - df.loc[draw, f"correct_{role_b}"].to_numpy()).mean())
        boots[b] = np.mean(values)
    return observed, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def per_class_selected(selected_models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for modality, group in selected_models.groupby("modality"):
        rows_by_role = {r.model_role: r for _, r in group.iterrows()}
        aligned = align_predictions(rows_by_role)
        for role in ["scratch", "supcon", "module"]:
            conf = aligned[f"confidence_{role}"].astype(float).to_numpy()
            corr = aligned[f"correct_{role}"].astype(float).to_numpy()
            edges = np.linspace(0.0, 1.0, 11)
            for b in range(10):
                lo, hi = edges[b], edges[b + 1]
                mask = (conf >= lo) & ((conf < hi) if b < 9 else (conf <= hi))
                n = int(mask.sum())
                reliability_rows.append({
                    "modality": modality, "role": role, "bin": b + 1, "bin_lower": lo, "bin_upper": hi,
                    "n": n, "mean_confidence": float(conf[mask].mean()) if n else math.nan,
                    "accuracy": float(corr[mask].mean()) if n else math.nan,
                    "ece_contribution": float(n / len(conf) * abs(corr[mask].mean() - conf[mask].mean())) if n else 0.0,
                })
        for class_name, cls in aligned.groupby("true_label_name"):
            support = len(cls)
            recalls = {role: float(cls[f"correct_{role}"].mean()) for role in ["scratch", "supcon", "module"]}
            row = {"modality": modality, "class_name": class_name, "support": support, **{f"recall_{k}": v for k, v in recalls.items()}}
            for base in ["scratch", "supcon"]:
                diff = cls["correct_module"].astype(float).to_numpy() - cls[f"correct_{base}"].astype(float).to_numpy()
                rng = np.random.default_rng(SEED + sum(map(ord, modality + class_name + base)))
                boots = np.empty(N_BOOT, dtype=float)
                for i in range(N_BOOT):
                    boots[i] = rng.choice(diff, size=support, replace=True).mean()
                row[f"delta_vs_{base}"] = float(diff.mean())
                row[f"delta_vs_{base}_ci_low"] = float(np.quantile(boots, 0.025))
                row[f"delta_vs_{base}_ci_high"] = float(np.quantile(boots, 0.975))
            class_rows.append(row)
        for base in ["scratch", "supcon"]:
            observed, low, high = stratified_bootstrap_delta(aligned, "module", base, N_BOOT, SEED + len(modality + base))
            module_row = rows_by_role["module"]
            base_row = rows_by_role[base]
            bootstrap_rows.append({
                "modality": modality, "comparison": f"module_vs_{base}", "delta_balanced_acc": observed,
                "ci_low": low, "ci_high": high, "n_samples": len(aligned), "n_classes": aligned.true_label_name.nunique(),
                "module_family": module_row.family, "module_mode": module_row.finetune_mode,
                "module_method": module_row.loss_group, "module_source_key": module_row.source_key,
                "baseline_source_key": base_row.source_key,
                "interval_scope": "paired stratified test-sample bootstrap; does not include training-seed uncertainty",
            })
    reliability = pd.DataFrame(reliability_rows)
    ece = reliability.groupby(["modality", "role"], as_index=False).ece_contribution.sum().rename(columns={"ece_contribution": "ece_10bin"})
    reliability = reliability.merge(ece, on=["modality", "role"], how="left")
    return pd.DataFrame(class_rows), pd.DataFrame(bootstrap_rows), reliability


def best_model_per_class_effects(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = selected[(selected.dataset_scope == "except_take_put") & (selected.family == "base")]
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for modality in ["rgb", "emg", "imu"]:
        scratch = data[(data.modality == modality) & (data.finetune_mode == "scratch")].sort_values("test_balanced_acc", ascending=False).iloc[0]
        scratch_pred = load_predictions(scratch)
        support_map = scratch_pred.groupby("true_label_name").size().to_dict()
        scratch_recall = scratch_pred.groupby("true_label_name").correct.mean().to_dict()
        for mode in ["full", "head_only"]:
            supcon = data[(data.modality == modality) & (data.finetune_mode == mode) & (data.loss_group == "supcon")].sort_values("test_balanced_acc", ascending=False).iloc[0]
            supcon_recall = load_predictions(supcon).groupby("true_label_name").correct.mean().to_dict()
            for method in ["supcon", "prototype", "relation", "prototype+relation"]:
                choices = data[(data.modality == modality) & (data.finetune_mode == mode) & (data.loss_group == method)].sort_values("test_balanced_acc", ascending=False)
                if not len(choices):
                    continue
                best = choices.iloc[0]
                recall = load_predictions(best).groupby("true_label_name").correct.mean().to_dict()
                method_rows = []
                for class_name in sorted(set(scratch_recall) | set(recall)):
                    rec = recall.get(class_name, math.nan)
                    ds = rec - scratch_recall.get(class_name, math.nan)
                    du = rec - supcon_recall.get(class_name, math.nan)
                    effect_s = "benefit" if ds >= CLASS_THRESHOLD else "harm" if ds <= -CLASS_THRESHOLD else "neutral"
                    effect_u = "benefit" if du >= CLASS_THRESHOLD else "harm" if du <= -CLASS_THRESHOLD else "neutral"
                    item = {
                        "modality": modality, "finetune_mode": mode, "method": method,
                        "method_label": METHOD_LABEL[method], "source_key": best.source_key,
                        "class_name": class_name, "support": support_map.get(class_name, 0),
                        "recall": rec, "scratch_recall": scratch_recall.get(class_name, math.nan),
                        "supcon_recall": supcon_recall.get(class_name, math.nan),
                        "delta_vs_scratch": ds, "delta_vs_supcon": du,
                        "effect_vs_scratch": effect_s, "effect_vs_supcon": effect_u,
                    }
                    detail.append(item); method_rows.append(item)
                m = pd.DataFrame(method_rows)
                for baseline in ["scratch", "supcon"]:
                    col = f"delta_vs_{baseline}"
                    valid = m[np.isfinite(m[col])]
                    benefit = valid[valid[col] >= CLASS_THRESHOLD].sort_values(col, ascending=False)
                    harm = valid[valid[col] <= -CLASS_THRESHOLD].sort_values(col)
                    summary.append({
                        "modality": modality, "finetune_mode": mode, "method": method,
                        "baseline": baseline, "n_present_classes": len(valid),
                        "n_benefit_classes": len(benefit), "n_harm_classes": len(harm),
                        "mean_recall_delta": valid[col].mean(),
                        "benefit_classes": "; ".join(f"{r.class_name} ({r[col]:+.2f})" for _, r in benefit.iterrows()),
                        "harm_classes": "; ".join(f"{r.class_name} ({r[col]:+.2f})" for _, r in harm.iterrows()),
                    })
    return pd.DataFrame(detail), pd.DataFrame(summary)


def plot_method_heatmaps(summary: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("score", ["#F2F4F7", "#A8D5E5", "#2F6B8A"])
    for modality in ["rgb", "emg", "imu"]:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
        for ax, mode in zip(axes, ["full", "head_only"]):
            sub = summary[(summary.dataset_scope == "except_take_put") & (summary.modality == modality) & (summary.finetune_mode.isin([mode, "scratch"]))]
            families = [f for f in FAMILY_ORDER if f in set(sub.family)]
            mat = np.full((len(families), len(METHOD_ORDER)), np.nan)
            for i, fam in enumerate(families):
                for j, method in enumerate(METHOD_ORDER):
                    q = sub[(sub.family == fam) & (sub.method == method)]
                    if method == "scratch":
                        q = q[q.finetune_mode == "scratch"]
                    else:
                        q = q[q.finetune_mode == mode]
                    if len(q): mat[i, j] = q.best_balanced_acc.max()
            finite = mat[np.isfinite(mat)]
            vmin = max(0.0, float(finite.min() - 0.04)) if len(finite) else 0
            vmax = min(1.0, float(finite.max() + 0.02)) if len(finite) else 1
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABEL[m].replace("SupCon+", "+") for m in METHOD_ORDER], rotation=25, ha="right")
            ax.set_yticks(range(len(families)), [FAMILY_LABEL.get(f, f) for f in families])
            ax.set_title(f"{modality.upper()} - {'Full' if mode == 'full' else 'Head-only'}")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, "—" if not np.isfinite(mat[i, j]) else f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#111827")
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Best balanced accuracy")
        fig.suptitle(f"{modality.upper()}: loss-method performance across experiment families", fontsize=13, fontweight="bold")
        fig.savefig(FIGURES / f"module_method_performance_{modality}.png", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_selected_per_class(per_class: pd.DataFrame, selected_models: pd.DataFrame) -> None:
    for modality in ["rgb", "emg", "imu"]:
        sub = per_class[per_class.modality == modality].copy().sort_values("delta_vs_supcon")
        classes = sub.class_name.tolist(); x = np.arange(len(classes))
        fig, axes = plt.subplots(2, 1, figsize=(14, 7.8), constrained_layout=True, gridspec_kw={"height_ratios": [1.0, 1.25]})
        abs_mat = sub[["recall_scratch", "recall_supcon", "recall_module"]].T.to_numpy()
        im = axes[0].imshow(abs_mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        axes[0].set_yticks([0, 1, 2], ["Scratch", "SupCon", "Best module"])
        axes[0].set_xticks(x, classes, rotation=35, ha="right")
        axes[0].set_title("Absolute per-class recall")
        for i in range(abs_mat.shape[0]):
            for j in range(abs_mat.shape[1]): axes[0].text(j, i, f"{abs_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes[0], fraction=0.018, pad=0.01)
        width = 0.38
        for offset, base, color in [(-width/2, "scratch", "#64748B"), (width/2, "supcon", "#D9772A")]:
            vals = sub[f"delta_vs_{base}"].to_numpy()
            low = sub[f"delta_vs_{base}_ci_low"].to_numpy(); high = sub[f"delta_vs_{base}_ci_high"].to_numpy()
            yerr = np.vstack([vals - low, high - vals])
            axes[1].bar(x + offset, vals, width=width, color=color, label=f"Module - {base.title()}", alpha=.9, yerr=yerr, capsize=2)
        axes[1].axhline(0, color="#111827", linewidth=.8)
        axes[1].axhline(CLASS_THRESHOLD, color="#2E8B57", linewidth=.8, linestyle="--")
        axes[1].axhline(-CLASS_THRESHOLD, color="#B64747", linewidth=.8, linestyle="--")
        axes[1].set_xticks(x, classes, rotation=35, ha="right")
        axes[1].set_ylabel("Recall change")
        axes[1].set_title("Best module recall change with paired 95% test-sample bootstrap intervals")
        axes[1].legend(frameon=False, ncol=2)
        axes[1].grid(axis="y", alpha=.2)
        module = selected_models[(selected_models.modality == modality) & (selected_models.model_role == "module")].iloc[0]
        fig.suptitle(f"{modality.upper()}: {module.loss_group} / {module.family} / {module.finetune_mode}", fontsize=13, fontweight="bold")
        fig.savefig(FIGURES / f"selected_module_per_class_{modality}.png", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_class_effects(detail: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("delta", ["#B64747", "#F8FAFC", "#2E8B57"])
    module_methods = ["prototype", "relation", "prototype+relation"]
    for modality in ["rgb", "emg", "imu"]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
        for r, mode in enumerate(["full", "head_only"]):
            for c, baseline in enumerate(["scratch", "supcon"]):
                ax = axes[r, c]
                methods = (["supcon"] + module_methods) if baseline == "scratch" else module_methods
                q = detail[(detail.modality == modality) & (detail.finetune_mode == mode) & detail.method.isin(methods)]
                classes = sorted(q.class_name.dropna().unique())
                mat = np.full((len(methods), len(classes)), np.nan)
                for i, method in enumerate(methods):
                    s = q[q.method == method].set_index("class_name")
                    for j, cls in enumerate(classes):
                        if cls in s.index: mat[i, j] = s.loc[cls, f"delta_vs_{baseline}"]
                ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-.45, vmax=.45)
                ax.set_yticks(range(len(methods)), [METHOD_LABEL[m].replace("SupCon+", "+") for m in methods])
                ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=7)
                ax.set_title(f"{'Full' if mode == 'full' else 'Head-only'} vs {baseline.title()}")
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        if np.isfinite(mat[i,j]): ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=6)
        fig.suptitle(f"{modality.upper()}: which classes each loss method helps or harms (base family)", fontsize=13, fontweight="bold")
        fig.savefig(FIGURES / f"class_effects_{modality}.png", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_reliability(reliability: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, modality in zip(axes, ["rgb", "emg", "imu"]):
        q = reliability[reliability.modality == modality]
        ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=1, label="Perfect calibration")
        for role in ["scratch", "supcon", "module"]:
            s = q[(q.role == role) & q.n.gt(0)].sort_values("bin")
            ece = s.ece_10bin.iloc[0] if len(s) else math.nan
            ax.plot(s.mean_confidence, s.accuracy, marker="o", linewidth=1.8, color=ROLE_COLOR[role], label=f"{ROLE_LABEL[role]} (ECE {ece:.3f})")
        ax.set_title(modality.upper()); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Empirical accuracy")
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Reliability diagrams for the selected module and its two baselines", fontsize=13, fontweight="bold")
    fig.savefig(FIGURES / "selected_model_reliability.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def orthogonal_align(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    s = l2_normalize(source.astype(np.float64)); r = l2_normalize(reference.astype(np.float64))
    s -= s.mean(axis=0, keepdims=True); r -= r.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(s.T @ r, full_matrices=False)
    return (s @ (u @ vt)).astype(np.float32)


def feature_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x = l2_normalize(x.astype(np.float64))
    unique = np.unique(labels)
    sil = silhouette_score(x, labels, metric="cosine") if len(unique) > 1 and len(x) > len(unique) else math.nan
    k = min(6, len(x))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x)
    inds = nn.kneighbors(return_distance=False)
    purity = float(np.mean([np.mean(labels[row[1:]] == labels[i]) for i, row in enumerate(inds)])) if k > 1 else math.nan
    within = []
    centroids = {}
    for cls in unique:
        pts = x[labels == cls]; centroid = l2_normalize(pts.mean(axis=0, keepdims=True))[0]; centroids[cls] = centroid
        within.extend(1 - pts @ centroid)
    centroid_dist = []
    for cls in unique:
        others = [1 - centroids[cls] @ centroids[o] for o in unique if o != cls]
        if others: centroid_dist.append(min(others))
    return {
        "silhouette_cosine": float(sil), "knn5_purity": purity,
        "within_class_cosine_distance": float(np.mean(within)) if within else math.nan,
        "nearest_other_centroid_distance": float(np.mean(centroid_dist)) if centroid_dist else math.nan,
        "centroid_margin": float(np.mean(centroid_dist) - np.mean(within)) if centroid_dist and within else math.nan,
    }


def targeted_feature_analysis(selected_models: pd.DataFrame, per_class: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    umap = None
    diagnostics: list[dict[str, Any]] = []
    harmed_rows: list[dict[str, Any]] = []
    palette = plt.cm.tab10.colors
    for modality in ["rgb", "emg", "imu"]:
        print(f"feature analysis: {modality} load", flush=True)
        group = selected_models[selected_models.modality == modality]
        feature_data: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
        for _, row in group.iterrows():
            d = FEATURE_ROOT / row.feature_model_id / "test"
            if not (d / "features_512.npy").is_file():
                feature_data = {}; break
            feature_data[row.model_role] = (np.load(d / "features_512.npy", allow_pickle=False), pd.read_csv(d / "samples.csv"))
        if len(feature_data) != 3:
            continue
        if umap is None:
            try:
                import umap as umap_module
                umap = umap_module
            except Exception:
                umap = False
        reference_names = feature_data["supcon"][1].sample_name.astype(str).tolist()
        aligned_features: dict[str, np.ndarray] = {}
        aligned_meta: dict[str, pd.DataFrame] = {}
        for role, (features, meta) in feature_data.items():
            index = pd.Series(np.arange(len(meta)), index=meta.sample_name.astype(str)).to_dict()
            take = np.array([index[name] for name in reference_names])
            aligned_features[role] = features[take]
            aligned_meta[role] = meta.iloc[take].reset_index(drop=True)
        q = per_class[per_class.modality == modality].sort_values("delta_vs_supcon")
        # Visualise only classes whose recall genuinely decreased versus SupCon.
        # If no class decreased, retain the single worst class so the figure still
        # documents the closest failure mode without labelling neutral classes as harm.
        harmed = q[q.delta_vs_supcon < 0].class_name.head(5).tolist()
        if not harmed:
            harmed = q.class_name.head(1).tolist()
        module_row = group[group.model_role == "module"].iloc[0]
        module_pred = load_predictions(module_row)
        present_true = set(module_pred.true_label_name.astype(str))
        confused: list[str] = []
        for cls in harmed:
            e = module_pred[(module_pred.true_label_name == cls) & (module_pred.correct == 0)]
            if len(e):
                candidates = [str(x) for x in e.pred_label_name.value_counts().index if str(x) in present_true]
                if candidates:
                    confused.append(candidates[0])
        selected_classes = list(dict.fromkeys(harmed + confused))
        for cls in harmed:
            harmed_rows.append({"modality": modality, "harmed_class": cls, "delta_vs_supcon": float(q.set_index("class_name").loc[cls, "delta_vs_supcon"]), "included_confusion_classes": "; ".join(selected_classes)})
        for role in ["scratch", "supcon", "module"]:
            meta = aligned_meta[role]; mask = meta.true_action.astype(str).isin(selected_classes).to_numpy()
            metrics = feature_metrics(aligned_features[role][mask], meta.loc[mask, "true_action"].astype(str).to_numpy())
            diagnostics.append({"modality": modality, "role": role, "scope": "selected_harmed_and_confusion_classes", "classes": "; ".join(selected_classes), "n_samples": int(mask.sum()), **metrics})
            for cls in harmed:
                cls_mask = meta.true_action.astype(str).eq(cls).to_numpy()
                x = l2_normalize(aligned_features[role][cls_mask].astype(np.float64))
                centroid = l2_normalize(x.mean(axis=0, keepdims=True))[0]
                diagnostics.append({
                    "modality": modality, "role": role, "scope": "harmed_class", "classes": cls,
                    "n_samples": int(cls_mask.sum()), "silhouette_cosine": math.nan, "knn5_purity": math.nan,
                    "within_class_cosine_distance": float(np.mean(1 - x @ centroid)),
                    "nearest_other_centroid_distance": math.nan, "centroid_margin": math.nan,
                    "classification_recall": float(meta.loc[cls_mask, "correct"].mean()),
                })
        print(f"feature analysis: {modality} diagnostics complete", flush=True)
        if umap is False:
            continue
        # Reduce before Procrustes.  A direct 512x512 SVD is needlessly slow on
        # CPU and the low-dimensional common basis is sufficient for a 2-D map.
        stacked = np.vstack([aligned_features[r] for r in ["scratch", "supcon", "module"]])
        pca = PCA(n_components=min(50, stacked.shape[0] - 1, stacked.shape[1]), svd_solver="randomized", random_state=42)
        reduced_all = pca.fit_transform(stacked)
        print(f"feature analysis: {modality} PCA complete", flush=True)
        n = len(aligned_features["supcon"])
        reduced = {
            "scratch": reduced_all[:n],
            "supcon": reduced_all[n:2*n],
            "module": reduced_all[2*n:3*n],
        }
        ref = reduced["supcon"]
        procrustes = {"supcon": orthogonal_align(ref, ref)}
        procrustes["scratch"] = orthogonal_align(reduced["scratch"], ref)
        procrustes["module"] = orthogonal_align(reduced["module"], ref)
        combined = []; plot_meta = []
        for role in ["scratch", "supcon", "module"]:
            meta = aligned_meta[role]; mask = meta.true_action.astype(str).isin(selected_classes).to_numpy()
            combined.append(procrustes[role][mask])
            plot_meta.append(pd.DataFrame({"role": role, "class_name": meta.loc[mask, "true_action"].astype(str).to_numpy()}))
        x_all = np.vstack(combined); meta_all = pd.concat(plot_meta, ignore_index=True)
        reducer = umap.UMAP(n_neighbors=min(15, len(x_all)-1), min_dist=.12, metric="cosine", random_state=42)
        coords = reducer.fit_transform(x_all)
        print(f"feature analysis: {modality} UMAP complete", flush=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True, sharex=True, sharey=True)
        color_map = {cls: palette[i % len(palette)] for i, cls in enumerate(selected_classes)}
        for ax, role in zip(axes, ["scratch", "supcon", "module"]):
            role_mask = meta_all.role.eq(role).to_numpy()
            for cls in selected_classes:
                mask = role_mask & meta_all.class_name.eq(cls).to_numpy()
                marker = "o" if cls in harmed else "x"
                ax.scatter(coords[mask,0], coords[mask,1], s=18 if marker == "o" else 25, marker=marker, alpha=.78, color=color_map[cls], label=cls)
            ax.set_title(ROLE_LABEL[role]); ax.set_xlabel("Shared UMAP 1"); ax.grid(alpha=.15)
        axes[0].set_ylabel("Shared UMAP 2")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, .5), frameon=False, title="Class")
        fig.suptitle(f"{modality.upper()}: harmed classes in a Procrustes-aligned shared UMAP", fontsize=13, fontweight="bold")
        fig.savefig(FIGURES / f"harmed_class_feature_space_{modality}.png", dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return pd.DataFrame(diagnostics), pd.DataFrame(harmed_rows)


def metric_definitions() -> pd.DataFrame:
    rows = [
        ("Accuracy", "(number of correct predictions) / N", "Overall correctness", "Can be dominated by frequent classes"),
        ("Per-class recall", "TP_c / (TP_c + FN_c)", "Fraction of true class c recovered", "Large variance for classes with few samples"),
        ("Balanced accuracy", "mean_c Recall_c over present classes", "Primary class-balanced performance metric", "Each present class has equal weight"),
        ("Macro-F1", "mean_c 2 Precision_c Recall_c / (Precision_c + Recall_c)", "Balances false positives and false negatives per class", "Undefined class terms require an explicit convention"),
        ("Recall change", "Recall_module,c - Recall_baseline,c", "Shows which classes a module helps or harms", "Reported in absolute units or percentage points"),
        ("ECE (10 bins)", "sum_b (n_b/N) * |accuracy_b - mean_confidence_b|", "Measures confidence calibration", "Depends on binning; low ECE does not imply high accuracy"),
        ("Silhouette (cosine)", "mean_i (b_i-a_i)/max(a_i,b_i)", "Compares within-class compactness with nearest other-class separation", "Range [-1,1]; sensitive to class geometry and feature metric"),
        ("kNN-5 purity", "mean fraction of five nearest neighbours sharing the label", "Local class mixing in feature space", "Affected by class imbalance and neighbourhood size"),
        ("Within-class cosine distance", "mean 1-cos(x_i, class centroid)", "Class compactness; lower is better", "Does not directly measure separation from other classes"),
        ("Centroid margin", "nearest other-centroid distance - within-class distance", "Positive values indicate separation beyond class spread", "Centroids may miss multimodal class structure"),
        ("Paired bootstrap CI", "resample matched test samples within each class", "Test-sample uncertainty for a fixed pair of models", "Does not include training randomness; seed=1 remains the key limitation"),
        ("UMAP", "nonlinear neighbour-preserving 2D projection", "Qualitative visualization of local structure", "Distances and cluster sizes can be distorted; not used alone for conclusions"),
    ]
    return pd.DataFrame(rows, columns=["metric", "calculation", "purpose", "interpretation_limit"])


def figure_guide() -> pd.DataFrame:
    return pd.DataFrame([
        ("Module method heatmaps", "Rows are experiment families; columns are loss methods; cells are the best balanced accuracy. A dash means the method was not run.", "Compare within the same modality, mode and family; do not compare missing cells as zero."),
        ("Selected per-class charts", "Top: absolute recall for Scratch, SupCon and the selected module. Bottom: module-minus-baseline recall with paired 95% intervals.", "Positive bars help the class; negative bars harm it. Dashed lines mark +/-5 pp descriptive thresholds."),
        ("Class-effect heatmaps", "Rows are loss methods and columns are classes; color/value is recall change versus the named baseline.", "Green is beneficial, red is harmful, near-white is small change. Full and head-only are separate panels."),
        ("Reliability diagrams", "x-axis is mean confidence in a bin; y-axis is empirical accuracy; diagonal means perfect calibration.", "Curves below the diagonal are over-confident. ECE summarizes the weighted absolute gap."),
        ("Shared UMAP feature plots", "The same samples from three models are orthogonally aligned to SupCon before a joint UMAP. Circles are harmed classes; x marks are their common confusion targets.", "Compare mixing/compactness, not absolute coordinate meaning. Use silhouette, purity and distance metrics for confirmation."),
        ("Existing family bar charts", "Bars show the best balanced accuracy found in each experiment family.", "Base/older groups often use last while newer scans use best_val; read checkpoint labels before ranking close results."),
        ("Checkpoint-effect chart", "Each bar is mean(best_val - last) within a family/modality/mode.", "Positive favors best_val; a near-zero mean can still hide large individual-run differences."),
    ], columns=["figure_type", "encoding", "how_to_read"])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = read_csv("selected_best_available.csv")
    summary = method_summary(selected)
    pairs = matched_pairs(selected)
    selected_models = choose_selected_models(selected)
    per_class, bootstrap, reliability = per_class_selected(selected_models)
    effects, effect_summary = best_model_per_class_effects(selected)
    plot_method_heatmaps(summary)
    plot_selected_per_class(per_class, selected_models)
    plot_class_effects(effects)
    plot_reliability(reliability)
    feature_diag, harmed = targeted_feature_analysis(selected_models, per_class)
    outputs = {
        "module_method_summary.csv": summary,
        "module_matched_pairs.csv": pairs,
        "selected_module_models.csv": selected_models,
        "selected_module_per_class.csv": per_class,
        "paired_bootstrap_summary.csv": bootstrap,
        "reliability_bins_selected.csv": reliability,
        "class_effect_matrix.csv": effects,
        "class_effect_summary.csv": effect_summary,
        "targeted_feature_diagnostics.csv": feature_diag,
        "harmed_class_selection.csv": harmed,
        "metric_definitions.csv": metric_definitions(),
        "figure_reading_guide.csv": figure_guide(),
    }
    for name, df in outputs.items():
        df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")
    module_rows = selected_models[selected_models.model_role == "module"]
    payload = {
        "selected_modules": module_rows[["modality", "family", "finetune_mode", "loss_group", "source_key", "test_balanced_acc"]].to_dict("records"),
        "feature_analysis_available": not feature_diag.empty,
        "class_effect_threshold": CLASS_THRESHOLD,
        "paired_bootstrap_iterations": N_BOOT,
        "notes": [
            "SupCon-only and scratch are both baselines.",
            "Matched intervals quantify test-sample uncertainty only; all training uses seed=1.",
            "Base-family class-effect matrices provide a complete four-method comparison.",
        ],
    }
    (OUT / "extended_module_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
