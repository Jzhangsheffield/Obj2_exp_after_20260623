#!/usr/bin/env python3
"""Build reproducible tables and figures for the Middle backbone/init analysis."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parents[1]
RESULTS = PROJECT / "results" / "rgb_takeput_middle_proto_rel_loso_20260817"
ASSETS = PACKAGE / "report_assets"
UMAP_ROOT = ASSETS / "middle_umap"
HIST_PROBE = PROJECT / "analysis" / "N_as_test" / "umap_rgb_pretrain" / "pretrain_linear_probe_knn_summary.csv"
HIST_FT_DIRS = [
    "ft_rgb_N_except_take_put_adamw_stage5_reltopk_random_9_seed1",
    "ft_rgb_N_take_put_adamw_22_seed1",
    "ft_rgb_N_except_take_put_adamw_22_seed1",
    "ft_rgb_N_except_take_put_adamw_depth10_random_4_seed1",
    "ft_rgb_N_except_take_put_adamw_random_kqueue_6_seed1",
    "ft_rgb_N_except_take_put_adamw_reltopk_random_6_seed1",
    "ft_rgb_N_except_take_put_adamw_sampler_9_seed1",
]

RUN_LABELS = {
    "r3d_rand_sup": "R3D random",
    "r3d_k400_sup": "R3D K400",
    "mvit_rand_sup": "MViT random",
    "mvit_k400_sup": "MViT K400",
}


def classifier_rows() -> list[dict]:
    base = RESULTS / "classifier" / "middle" / "dev_N"
    rows: list[dict] = []
    for path in base.rglob("summary.json"):
        rel = path.relative_to(base).parts
        route, experiment, policy = rel[0], rel[1], rel[2]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if route == "middle_direct":
            method = "Direct"
            backbone = "MViT-v2-S" if experiment.startswith("mvit") else "R3D-18"
            initialization = "K400" if "k400" in experiment else "Random"
        else:
            method = "SupLoss"
            backbone = "MViT-v2-S" if experiment.startswith("mvit") else "R3D-18"
            initialization = "K400" if "k400" in experiment else "Random"
        rows.append({
            "route": route,
            "experiment": experiment,
            "method": method,
            "backbone": backbone,
            "initialization": initialization,
            "policy": policy,
            "best_val_ba": payload["best_val_balanced_acc"],
            "best_val_macro_f1": payload["best_val_macro_f1"],
            "best_val_acc": payload["best_val_acc"],
            "best_val_ba_epoch": payload["best_val_balanced_epoch"],
            "final_val_ba": payload["final_val_balanced_acc"],
            "final_val_macro_f1": payload["final_val_macro_f1"],
            "final_train_ba": payload["final_train_balanced_acc"],
            "best_to_final_ba_drop": payload["best_val_balanced_acc"] - payload["final_val_balanced_acc"],
        })
    return sorted(rows, key=lambda r: (r["method"], r["backbone"], r["initialization"], r["policy"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_rows() -> list[dict]:
    rows = []
    for run, label in RUN_LABELS.items():
        payload = json.loads((UMAP_ROOT / run / "feature_quality.json").read_text(encoding="utf-8"))
        for representation, info in payload["representations"].items():
            rows.append({
                "run": run,
                "model": label,
                "representation": representation,
                "train_silhouette": info["train_geometry"]["silhouette_cosine"],
                "test_silhouette": info["test_geometry"]["silhouette_cosine"],
                "linear_ba": info["held_out_predictive"]["linear_balanced_accuracy"],
                "linear_macro_f1": info["held_out_predictive"]["linear_macro_f1"],
                "knn1_ba": info["held_out_predictive"]["knn1_balanced_accuracy"],
                "test_effective_rank": info["test_geometry"]["effective_rank"],
                "test_top5_variance_fraction": info["test_geometry"]["top5_variance_fraction"],
            })
    return rows


def debug_rows() -> list[dict]:
    base = RESULTS / "pretrain" / "middle" / "dev_N" / "middle_backbone_pretrain"
    rows = []
    for run in RUN_LABELS:
        path = base / run / "debug_train_log.jsonl"
        by_epoch: dict[int, list[dict]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            by_epoch.setdefault(int(item["epoch"]), []).append(item)
        for epoch, items in sorted(by_epoch.items()):
            rows.append({
                "run": run,
                "model": RUN_LABELS[run],
                "epoch": epoch,
                "loss": float(np.mean([x["loss"] for x in items])),
                "q_feature_std_mean": float(np.mean([x["feature_stats"]["q_feature_std_mean"] for x in items])),
                "total_grad_norm": float(np.mean([x["grad_stats"]["total_grad_norm"] for x in items])),
            })
    return rows


def historical_probe_rows() -> list[dict]:
    with HIST_PROBE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keep = []
    for row in rows:
        keep.append({
            "task": row["group"],
            "checkpoint": row["checkpoint_slug"],
            "representation": row["representation"],
            "num_classes": int(row["num_classes"]),
            "linear_ba": float(row["linear_probe_balanced_acc"]),
            "linear_macro_f1": float(row["linear_probe_macro_f1"]),
            "knn1_ba": float(row["knn_k1_balanced_acc"]),
        })
    return keep


def historical_finetune_rows() -> list[dict]:
    output = []
    for folder in HIST_FT_DIRS:
        root = PROJECT / "results" / folder
        records = []
        for path in root.rglob("summary.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("best_val_balanced_acc") is None:
                continue
            records.append({
                "run": path.parent.name,
                "policy": path.parent.parent.name,
                "best_val_ba": float(payload["best_val_balanced_acc"]),
                "best_val_macro_f1": float(payload["best_val_macro_f1"]),
            })
        scratch = [r for r in records if r["policy"] == "scratch_full"]
        pretrained = [r for r in records if r["policy"] == "full"]
        best_scratch = max(scratch, key=lambda r: r["best_val_ba"])
        best_pretrained = max(pretrained, key=lambda r: r["best_val_ba"])
        output.append({
            "family": folder,
            "num_summaries": len(records),
            "best_scratch_run": best_scratch["run"],
            "best_scratch_ba": best_scratch["best_val_ba"],
            "best_pretrained_run": best_pretrained["run"],
            "best_pretrained_ba": best_pretrained["best_val_ba"],
            "pretrained_minus_scratch_pp": 100 * (best_pretrained["best_val_ba"] - best_scratch["best_val_ba"]),
        })
    return output


def plot_performance(rows: list[dict]) -> None:
    full = [r for r in rows if r["policy"] == "full"]
    order = [
        ("R3D-18", "Random"), ("R3D-18", "K400"),
        ("MViT-v2-S", "Random"), ("MViT-v2-S", "K400"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    x = np.arange(len(order))
    width = 0.36
    for offset, method, color in [(-width / 2, "Direct", "#4C78A8"), (width / 2, "SupLoss", "#F58518")]:
        vals = []
        for backbone, init in order:
            match = [r for r in full if r["method"] == method and r["backbone"] == backbone and r["initialization"] == init]
            vals.append(100 * match[0]["best_val_ba"] if match else np.nan)
        bars = ax.bar(x + offset, vals, width, label=method, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(x, ["R3D\nrandom", "R3D\nK400", "MViT\nrandom", "MViT\nK400"])
    ax.set_ylabel("Held-out N best balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(100 / 11, color="#888888", linestyle="--", linewidth=1, label="11-class chance")
    ax.set_title("Middle (11 classes): backbone, initialization, and SupLoss")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "middle_best_ba.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_feature_bars(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    order = list(RUN_LABELS)
    for axis, representation in zip(axes, ["backbone", "projection"]):
        vals = [100 * next(r["linear_ba"] for r in rows if r["run"] == run and r["representation"] == representation) for run in order]
        colors = ["#9ECAE1", "#3182BD", "#FDD0A2", "#E6550D"]
        bars = axis.bar(np.arange(4), vals, color=colors)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
        axis.set_xticks(np.arange(4), ["R3D\nrandom", "R3D\nK400", "MViT\nrandom", "MViT\nK400"])
        axis.set_title(f"{representation.capitalize()} feature")
        axis.axhline(100 / 11, color="#777777", linestyle="--", linewidth=1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Frozen linear probe BA on N (%)")
    axes[0].set_ylim(0, 100)
    fig.suptitle("Middle SupLoss epoch-200 frozen-feature transfer")
    fig.tight_layout()
    fig.savefig(ASSETS / "middle_frozen_probe.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_umap_grid() -> None:
    label_map = json.loads((PACKAGE / "runtime" / "manifests" / "middle" / "dev_N" / "label_map.json").read_text(encoding="utf-8"))
    if "tier1" in label_map:
        names = {int(v): k for k, v in label_map["tier1"].items()}
    elif isinstance(label_map, dict) and all(str(k).isdigit() for k in label_map):
        names = {int(k): v for k, v in label_map.items()}
    else:
        names = {int(v): k for k, v in label_map.items()}
    cmap = plt.get_cmap("tab20")
    colors = {i: cmap(i) for i in names}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, run in enumerate(RUN_LABELS):
        for row, rep in enumerate(["backbone", "projection"]):
            ax = axes[row, col]
            arr = np.load(UMAP_ROOT / run / f"{rep}_umap_trainfit.npz")
            for label in sorted(names):
                tr = arr["train_y"] == label
                te = arr["test_y"] == label
                ax.scatter(arr["train"][tr, 0], arr["train"][tr, 1], s=4, alpha=0.16, color=colors[label], linewidths=0)
                ax.scatter(arr["test"][te, 0], arr["test"][te, 1], s=12, alpha=0.72, color=colors[label], marker="x", linewidths=0.7)
            q = json.loads((UMAP_ROOT / run / "feature_quality.json").read_text(encoding="utf-8"))["representations"][rep]
            ax.set_title(f"{RUN_LABELS[run]}\nN BA={100*q['held_out_predictive']['linear_balanced_accuracy']:.1f}%, sil={q['test_geometry']['silhouette_cosine']:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(rep.capitalize())
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i], markersize=6, label=names[i]) for i in sorted(names)]
    handles += [Line2D([0], [0], marker="o", color="#666", linestyle="none", markersize=4, label="train: dot"), Line2D([0], [0], marker="x", color="#666", linestyle="none", markersize=6, label="held-out N: x")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Middle SupLoss epoch-200 UMAP (fit on M/MR/J; N transformed without refitting)", fontsize=14)
    fig.subplots_adjust(top=0.87, bottom=0.15, wspace=0.06, hspace=0.18)
    fig.savefig(ASSETS / "middle_umap_grid.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for run in RUN_LABELS:
        part = [r for r in rows if r["run"] == run]
        epoch = [r["epoch"] for r in part]
        axes[0].plot(epoch, [r["loss"] for r in part], label=RUN_LABELS[run], linewidth=1.3)
        axes[1].plot(epoch, [r["q_feature_std_mean"] for r in part], label=RUN_LABELS[run], linewidth=1.3)
        axes[2].plot(epoch, [r["total_grad_norm"] for r in part], label=RUN_LABELS[run], linewidth=1.2)
    axes[0].set_title("SupLoss"); axes[0].set_ylabel("Mean loss")
    axes[1].set_title("Projection dispersion"); axes[1].set_ylabel("Mean per-dimension std")
    axes[2].set_title("Gradient activity"); axes[2].set_ylabel("Mean total gradient norm"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Pretraining epoch"); ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Middle SupLoss training dynamics")
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    fig.savefig(ASSETS / "middle_pretrain_dynamics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cls = classifier_rows()
    feat = feature_rows()
    dbg = debug_rows()
    hist = historical_probe_rows()
    hist_ft = historical_finetune_rows()
    write_csv(ASSETS / "middle_classifier_summary.csv", cls)
    write_csv(ASSETS / "middle_feature_summary.csv", feat)
    write_csv(ASSETS / "middle_pretrain_debug_by_epoch.csv", dbg)
    write_csv(ASSETS / "historical_r3d_pretrain_probe_summary.csv", hist)
    write_csv(ASSETS / "historical_r3d_finetune_family_summary.csv", hist_ft)
    plot_performance(cls)
    plot_feature_bars(feat)
    plot_umap_grid()
    plot_dynamics(dbg)
    print(json.dumps({"classifier_rows": len(cls), "feature_rows": len(feat), "debug_rows": len(dbg), "historical_probe_rows": len(hist), "historical_ft_families": len(hist_ft)}, indent=2))


if __name__ == "__main__":
    main()
