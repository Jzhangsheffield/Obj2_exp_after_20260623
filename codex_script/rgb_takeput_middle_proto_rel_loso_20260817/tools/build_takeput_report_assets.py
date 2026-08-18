#!/usr/bin/env python3
"""Build compact, reproducible tables and figures for the Take/Put report."""

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
UMAP_ROOT = ASSETS / "umap"

RUNS = {
    "pretrain_r3d_rand": ("R3D-18", "Random", "SupLoss pretrain"),
    "pretrain_r3d_k400": ("R3D-18", "K400", "SupLoss pretrain"),
    "pretrain_mvit_rand": ("MViT-v2-S", "Random", "SupLoss pretrain"),
    "pretrain_mvit_k400": ("MViT-v2-S", "K400", "SupLoss pretrain"),
    "direct_r3d_rand": ("R3D-18", "Random", "Direct full"),
    "direct_r3d_k400": ("R3D-18", "K400", "Direct full"),
    "direct_mvit_rand": ("MViT-v2-S", "Random", "Direct full"),
    "direct_mvit_k400": ("MViT-v2-S", "K400", "Direct full"),
    "supfull_r3d_rand": ("R3D-18", "Random", "SupLoss + full FT"),
    "supfull_r3d_k400": ("R3D-18", "K400", "SupLoss + full FT"),
    "supfull_mvit_rand": ("MViT-v2-S", "Random", "SupLoss + full FT"),
    "supfull_mvit_k400": ("MViT-v2-S", "K400", "SupLoss + full FT"),
}

COLORS = {0: "#2878B5", 1: "#E1812C"}
CLASS_NAMES = {0: "take", 1: "put"}


def quality(run: str, representation: str = "backbone") -> dict:
    payload = json.loads((UMAP_ROOT / run / "feature_quality.json").read_text(encoding="utf-8"))
    return payload["representations"][representation]


def write_feature_summary() -> None:
    fields = [
        "run", "backbone", "initialization", "stage", "representation",
        "train_silhouette", "test_silhouette", "linear_ba", "linear_macro_f1",
        "knn1_ba", "test_effective_rank", "test_top5_variance_fraction",
        "take_recall", "put_recall",
    ]
    with (ASSETS / "takeput_feature_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in RUNS:
            for representation in ("backbone", "projection"):
                info = quality(run, representation)
                matrix = info["held_out_predictive"]["linear_confusion_matrix"]
                writer.writerow({
                    "run": run,
                    "backbone": RUNS[run][0],
                    "initialization": RUNS[run][1],
                    "stage": RUNS[run][2],
                    "representation": representation,
                    "train_silhouette": info["train_geometry"]["silhouette_cosine"],
                    "test_silhouette": info["test_geometry"]["silhouette_cosine"],
                    "linear_ba": info["held_out_predictive"]["linear_balanced_accuracy"],
                    "linear_macro_f1": info["held_out_predictive"]["linear_macro_f1"],
                    "knn1_ba": info["held_out_predictive"]["knn1_balanced_accuracy"],
                    "test_effective_rank": info["test_geometry"]["effective_rank"],
                    "test_top5_variance_fraction": info["test_geometry"]["top5_variance_fraction"],
                    "take_recall": matrix[0][0] / sum(matrix[0]),
                    "put_recall": matrix[1][1] / sum(matrix[1]),
                })


def classifier_rows() -> list[dict]:
    rows = []
    base = RESULTS / "classifier" / "take_put" / "dev_N"
    for path in base.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(base).parts
        route, experiment, policy = rel[0], rel[1], rel[2]
        rows.append({
            "route": route,
            "experiment": experiment,
            "policy": policy,
            "final_train_ba": payload["final_train_balanced_acc"],
            "best_val_ba": payload["best_val_balanced_acc"],
            "best_val_macro_f1": payload["best_val_macro_f1"],
            "best_val_accuracy": payload["best_val_acc"],
            "best_val_ba_epoch": payload["best_val_balanced_epoch"],
            "final_val_ba": payload["final_val_balanced_acc"],
            "final_val_macro_f1": payload["final_val_macro_f1"],
            "final_val_loss": payload["final_val_loss"],
            "best_to_final_ba_drop": payload["best_val_balanced_acc"] - payload["final_val_balanced_acc"],
        })
    return sorted(rows, key=lambda row: (row["route"], row["experiment"], row["policy"]))


def write_classifier_summary(rows: list[dict]) -> None:
    with (ASSETS / "takeput_classifier_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scatter_panel(axis, run: str, representation: str) -> None:
    arrays = np.load(UMAP_ROOT / run / f"{representation}_umap_trainfit.npz")
    for label in (0, 1):
        train_mask = arrays["train_y"] == label
        test_mask = arrays["test_y"] == label
        axis.scatter(
            arrays["train"][train_mask, 0], arrays["train"][train_mask, 1],
            s=5, alpha=0.23, c=COLORS[label], marker="o", linewidths=0,
        )
        axis.scatter(
            arrays["test"][test_mask, 0], arrays["test"][test_mask, 1],
            s=15, alpha=0.72, c=COLORS[label], marker="x", linewidths=0.8,
        )
    info = quality(run, representation)
    ba = 100 * info["held_out_predictive"]["linear_balanced_accuracy"]
    sil = info["test_geometry"]["silhouette_cosine"]
    rank = info["test_geometry"]["effective_rank"]
    axis.set_title(f"{RUNS[run][0]} | {RUNS[run][1]}\nN probe BA={ba:.1f}%, sil={sil:.2f}, rank={rank:.1f}", fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.6)


def add_legend(fig) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[0], markeredgecolor="none", markersize=6, label="take"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[1], markeredgecolor="none", markersize=6, label="put"),
        Line2D([0], [0], marker="o", color="#666666", linestyle="none", markersize=4, label="train: dot"),
        Line2D([0], [0], marker="x", color="#666666", linestyle="none", markersize=6, label="held-out N: x"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)


def pretrain_umap_grid(representation: str) -> None:
    order = [
        "pretrain_r3d_rand", "pretrain_r3d_k400",
        "pretrain_mvit_rand", "pretrain_mvit_k400",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=False)
    for axis, run in zip(axes.flat, order):
        scatter_panel(axis, run, representation)
    title = "SupLoss epoch-200 backbone features" if representation == "backbone" else "SupLoss epoch-200 projection features"
    fig.suptitle(f"Take/Put UMAP — {title}\nUMAP fitted on M/MR/J only; N transformed without refitting", fontsize=14)
    add_legend(fig)
    fig.subplots_adjust(top=0.87, bottom=0.09, wspace=0.06, hspace=0.18)
    fig.savefig(ASSETS / f"umap_pretrain_{representation}_grid.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def downstream_umap_grid() -> None:
    order = [
        "direct_r3d_rand", "direct_r3d_k400", "direct_mvit_rand", "direct_mvit_k400",
        "supfull_r3d_rand", "supfull_r3d_k400", "supfull_mvit_rand", "supfull_mvit_k400",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=False)
    for axis, run in zip(axes.flat, order):
        scatter_panel(axis, run, "backbone")
    axes[0, 0].set_ylabel("Direct full", fontsize=12)
    axes[1, 0].set_ylabel("SupLoss + full FT", fontsize=12)
    fig.suptitle("Take/Put downstream backbone UMAP\nBest validation-BA checkpoint; UMAP fitted on M/MR/J only", fontsize=14)
    add_legend(fig)
    fig.subplots_adjust(top=0.84, bottom=0.11, wspace=0.06, hspace=0.18)
    fig.savefig(ASSETS / "umap_downstream_backbone_grid.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def pretrain_dynamics() -> None:
    folders = {
        "R3D random": "r3d_rand_sup",
        "R3D K400": "r3d_k400_sup",
        "MViT random": "mvit_rand_sup",
        "MViT K400": "mvit_k400_sup",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for label, folder in folders.items():
        path = RESULTS / "analysis" / "take_put" / "dev_N" / "takeput_pretrain" / folder / "training_diagnostics" / "debug_by_epoch.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epoch = np.asarray([int(float(row["epoch"])) for row in rows])
        loss = np.asarray([float(row["loss.mean"]) for row in rows])
        dim_std = np.asarray([float(row["feature_stats.q_feature_std_mean.mean"]) for row in rows])
        grad = np.asarray([float(row["grad_stats.total_grad_norm.mean"]) for row in rows])
        axes[0].plot(epoch, loss, label=label, linewidth=1.5)
        axes[1].plot(epoch, dim_std, label=label, linewidth=1.5)
        axes[2].plot(epoch, grad, label=label, linewidth=1.2)
    axes[0].set_title("SupLoss")
    axes[0].set_ylabel("Mean loss")
    axes[1].set_title("Projection dispersion")
    axes[1].set_ylabel("Mean per-dimension std")
    axes[2].set_title("Gradient activity")
    axes[2].set_ylabel("Mean total gradient norm")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Pretraining epoch")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Take/Put SupLoss training dynamics")
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    fig.savefig(ASSETS / "pretrain_dynamics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


LOG_PATTERN = re.compile(
    r"^\[(?P<epoch>\d+)\].*?train_balanced_acc: (?P<train>[0-9.]+).*?val_balanced_acc: (?P<val>[0-9.]+)"
)


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epoch, train, val = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_PATTERN.match(line)
        if match:
            epoch.append(int(match.group("epoch")) + 1)
            train.append(float(match.group("train")))
            val.append(float(match.group("val")))
    return np.asarray(epoch), np.asarray(train), np.asarray(val)


def classifier_curves() -> None:
    configs = [
        ("R3D-18 random", "r3d_rand_full", "r3d_rand_sup"),
        ("R3D-18 K400", "r3d_k400_full", "r3d_k400_sup"),
        ("MViT-v2-S random", "mvit_rand_full", "mvit_rand_sup"),
        ("MViT-v2-S K400", "mvit_k400_full", "mvit_k400_sup"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, (title, direct_id, sup_id) in zip(axes.flat, configs):
        direct_log = next((RESULTS / "classifier" / "take_put" / "dev_N" / "takeput_direct" / direct_id / "full").rglob("train_logs.txt"))
        sup_log = next((RESULTS / "classifier" / "take_put" / "dev_N" / "takeput_pretrain" / sup_id / "full").rglob("train_logs.txt"))
        for label, path, color in (("Direct full", direct_log, "#4C78A8"), ("SupLoss + full FT", sup_log, "#F58518")):
            epoch, _, val = read_curve(path)
            axis.plot(epoch, 100 * val, label=label, color=color, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Classifier epoch")
        axis.set_ylabel("N balanced accuracy (%)")
        axis.set_ylim(45, 100)
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Held-out N learning curves")
    fig.subplots_adjust(top=0.9, bottom=0.1, wspace=0.18, hspace=0.25)
    fig.savefig(ASSETS / "classifier_val_ba_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def performance_bars(rows: list[dict]) -> None:
    lookup = {(row["route"], row["experiment"], row["policy"]): row for row in rows}
    groups = [
        ("R3D\nrandom", "r3d_rand_full", "r3d_rand_sup"),
        ("R3D\nK400", "r3d_k400_full", "r3d_k400_sup"),
        ("MViT\nrandom", "mvit_rand_full", "mvit_rand_sup"),
        ("MViT\nK400", "mvit_k400_full", "mvit_k400_sup"),
    ]
    values = {"Direct full": [], "SupLoss head": [], "SupLoss full": []}
    for _, direct_id, sup_id in groups:
        values["Direct full"].append(100 * lookup[("takeput_direct", direct_id, "full")]["best_val_ba"])
        values["SupLoss head"].append(100 * lookup[("takeput_pretrain", sup_id, "head_only")]["best_val_ba"])
        values["SupLoss full"].append(100 * lookup[("takeput_pretrain", sup_id, "full")]["best_val_ba"])
    x = np.arange(len(groups))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, (label, series), color in zip((-width, 0, width), values.items(), ("#4C78A8", "#54A24B", "#F58518")):
        bars = axis.bar(x + offset, series, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.axhline(50, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xticks(x, [group[0] for group in groups])
    axis.set_ylim(45, 100)
    axis.set_ylabel("Best N balanced accuracy (%)")
    axis.set_title("Take/Put downstream performance (single seed, dev-N)")
    axis.legend(frameon=False, ncol=3, loc="lower center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "best_val_ba_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = classifier_rows()
    write_feature_summary()
    write_classifier_summary(rows)
    pretrain_umap_grid("backbone")
    pretrain_umap_grid("projection")
    downstream_umap_grid()
    pretrain_dynamics()
    classifier_curves()
    performance_bars(rows)
    print(f"Wrote report assets to {ASSETS}")


if __name__ == "__main__":
    main()
