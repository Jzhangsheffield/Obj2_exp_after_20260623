#!/usr/bin/env python
"""Build comparison-first analysis tables and figures for all Obj2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OLD_ANALYSIS = ROOT / "analysis" / "obj2a_260716"
DEFAULT_OUT = ROOT / "analysis" / "obj2_experiment_report_20260719"

BLUE, ORANGE, GREEN, RED, PURPLE = "#2F6B8A", "#D9772A", "#2E8B57", "#B64747", "#7562A8"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED, "#4F9DA6", "#B38B3D", "#6C757D"]

FAMILY_LABELS = {
    "base": "基础损失组合", "take_put": "Take/Put 二分类",
    "random_kqueue": "队列容量 K 扫描", "relation_topk": "关系损失 Top-k 扫描",
    "sampler": "平衡批采样", "stage5_topk": "阶段延迟 + Top-k",
    "depth10": "RGB ResNet-10 深度", "round2": "RGB 动作语义保持增强",
    "dualcam": "RGB 双相机预训练", "unknown": "其他",
}


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def new_canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((width // 2, 28), title, fill="#17324D", font=font(30, True), anchor="ma")
    return img, draw


def axes(draw, box, xmin=0.0, xmax=1.0, xticks=5):
    x0, y0, x1, y1 = box
    draw.line((x0, y0, x0, y1), fill="#444444", width=2)
    draw.line((x0, y1, x1, y1), fill="#444444", width=2)
    for i in range(xticks + 1):
        v = xmin + (xmax - xmin) * i / xticks
        x = x0 + (x1 - x0) * i / xticks
        draw.line((x, y0, x, y1), fill="#E8ECF0", width=1)
        draw.text((x, y1 + 8), f"{v:.2f}", fill="#444444", font=font(16), anchor="ma")


def hbars(draw, box, labels, values, colors=None, xmin=0.0, xmax=None, value_fmt="{:.3f}"):
    x0, y0, x1, y1 = box
    if not labels:
        draw.text(((x0+x1)//2, (y0+y1)//2), "No data", fill="#777777", font=font(18), anchor="mm")
        return
    xmax = max(values) * 1.12 if xmax is None else xmax
    xmin = min(xmin, min(values, default=0))
    xmax = max(xmax, xmin + 1e-6)
    row_h = (y1 - y0) / len(labels)
    zero_x = x0 + (0 - xmin) / (xmax - xmin) * (x1 - x0)
    draw.line((zero_x, y0, zero_x, y1), fill="#555555", width=2)
    for i, (label, value) in enumerate(zip(labels, values)):
        cy = y0 + row_h * (i + .5)
        bar_h = max(10, row_h * .58)
        vx = x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)
        color = colors[i] if colors else PALETTE[i % len(PALETTE)]
        draw.rectangle((min(zero_x, vx), cy - bar_h/2, max(zero_x, vx), cy + bar_h/2), fill=color)
        draw.text((x0 - 12, cy), str(label), fill="#333333", font=font(15), anchor="rm")
        draw.text((vx + (6 if value >= 0 else -6), cy), value_fmt.format(value), fill="#222222", font=font(14), anchor="lm" if value >= 0 else "rm")


def line_plot(draw, box, series, xlabels, ymin=None, ymax=None):
    x0, y0, x1, y1 = box
    all_vals = [v for _, vals, _ in series for v in vals if pd.notna(v)]
    if not all_vals:
        return
    ymin = min(all_vals) - .03 if ymin is None else ymin
    ymax = max(all_vals) + .03 if ymax is None else ymax
    for i in range(6):
        y = y1 - (y1-y0)*i/5
        val = ymin + (ymax-ymin)*i/5
        draw.line((x0,y,x1,y), fill="#E8ECF0", width=1)
        draw.text((x0-8,y), f"{val:.2f}", fill="#444", font=font(14), anchor="rm")
    n = max(1, len(xlabels)-1)
    for i,label in enumerate(xlabels):
        x=x0+(x1-x0)*i/n
        draw.text((x,y1+8), str(label), fill="#444", font=font(14), anchor="ma")
    for name, vals, color in series:
        pts=[]
        for i,v in enumerate(vals):
            if pd.isna(v): continue
            x=x0+(x1-x0)*i/n; y=y1-(v-ymin)/(ymax-ymin)*(y1-y0); pts.append((x,y)); draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
        if len(pts)>1: draw.line(pts,fill=color,width=4)
    lx=x1-10; ly=y0+8
    for name,_,color in reversed(series):
        draw.rectangle((lx-170,ly,lx-150,ly+16),fill=color); draw.text((lx-142,ly+8),name,fill="#333",font=font(14),anchor="lm"); ly+=24


def read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def json_scalar(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def short_id(value: str, prefix: str = "m") -> str:
    return prefix + hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:10]


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def top_folder(path: Path) -> str:
    try:
        return path.relative_to(RESULTS).parts[0]
    except Exception:
        return ""


def family_from_text(text: str) -> str:
    s = text.lower()
    if "dualcam" in s: return "dualcam"
    if "round2" in s or "action_preserving" in s: return "round2"
    if "take_put" in s and "except_take_put" not in s: return "take_put"
    if "stage5" in s: return "stage5_topk"
    if "reltopk" in s: return "relation_topk"
    if "random_kqueue" in s: return "random_kqueue"
    if "sampler" in s: return "sampler"
    if "depth10" in s: return "depth10"
    if "except_take_put" in s or "_44" in s or "_22" in s: return "base"
    return "unknown"


def dataset_scope(text: str) -> str:
    s = text.lower()
    if "dualcam" in s: return "dualcam_N"
    if "round2" in s: return "round2_except_take_put"
    if "except_take_put" in s: return "except_take_put"
    if "take_put" in s: return "take_put"
    return "unknown"


def modality_from_text(text: str, fallback: str = "") -> str:
    s = text.lower()
    if "signal_emg" in s or re.search(r"(^|[\\/_])emg([\\/_]|$)", s): return "emg"
    if "signal_imu" in s or re.search(r"(^|[\\/_])imu([\\/_]|$)", s): return "imu"
    if "rgb" in s or fallback.lower() == "rgb": return "rgb"
    return fallback.lower() or "unknown"


def mode_from_text(text: str, config_mode: str = "") -> str:
    s = text.lower()
    if "scratch" in s: return "scratch"
    if "head_only" in s: return "head_only"
    if re.search(r"(^|[\\/])full([\\/]|$)", s): return "full"
    return config_mode or "unknown"


def source_key(run_name: str) -> str:
    s = re.sub(r"^run_\d+_", "", str(run_name))
    s = re.sub(r"^signal_(?:emg|imu)_", "", s)
    return re.sub(r"_checkpoint_\d+$", "", s)


def loss_group(text: str) -> str:
    s = text.lower()
    if "scratch" in s: return "scratch"
    if "rel" in s and ("proto" in s or "rel_ce" in s or "ce050" in s): return "prototype+relation"
    if "proto" in s: return "prototype"
    if "rel" in s: return "relation"
    if "sup" in s: return "supcon"
    return "other"


def extract_tokens(text: str) -> dict[str, Any]:
    s = text.lower()
    m = re.search(r"(?:^|_)k(\d+)(?:_|$)", s)
    q = int(m.group(1)) if m else np.nan
    m = re.search(r"topk(all|\d+)", s)
    topk = (999 if m and m.group(1) == "all" else int(m.group(1))) if m else np.nan
    m = re.search(r"(?:^|_)p([123])(?:_|$)", s)
    p = int(m.group(1)) if m else np.nan
    m = re.search(r"prem(0?\.\d+|\d+)", s)
    prem = float(m.group(1)) if m else np.nan
    m = re.search(r"blr([0-9]+e-?[0-9]+)", s)
    try: blr = float(m.group(1)) if m else np.nan
    except ValueError: blr = np.nan
    return {"queue_size_name": q, "topk_name": topk, "num_prototypes_name": p,
            "preview_momentum_name": prem, "backbone_lr_name": blr}


def remote_tail(value: str, marker: str = "weights") -> list[str]:
    parts = [p for p in re.split(r"[\\/]+", str(value)) if p]
    lowered = [p.lower() for p in parts]
    if marker not in lowered: return []
    idx = max(i for i, p in enumerate(lowered) if p == marker)
    return parts[idx + 1:]


def local_run_dir(ft_root: Path, weight_path: str) -> Path:
    tail = remote_tail(weight_path)
    return ft_root / "weights" / Path(*tail[:-1]) if len(tail) >= 2 else Path()


def get_config_fields(run_dir: Path) -> dict[str, Any]:
    obj = read_json(run_dir / "config.json")
    args = obj.get("args", {}) if isinstance(obj.get("args", {}), dict) else {}
    keys = ["seed", "model_depth", "num_classes", "epochs", "batch_size", "optimizer",
            "learning_rate", "backbone_learning_rate", "head_learning_rate", "weight_decay",
            "finetune_mode", "use_discriminative_lr", "use_weighted_sampler", "use_weighted_ce",
            "use_focal", "focal_gamma", "disable_train_augmentation", "rgb_vflip_p", "rgb_hflip_p",
            "rgb_jitter_p", "rgb_gray_p", "rgb_blur_p", "rrc_scale_min", "rrc_scale_max",
            "mindrove_apply_normalization", "mindrove_target_len", "mindrove_signals",
            "pretrained_tag_anchor", "pretrained_tag_mode"]
    out = {f"cfg_{k}": json_scalar(args.get(k)) for k in keys}
    out.update(config_exists=(run_dir / "config.json").is_file(), config_path=relpath(run_dir / "config.json"))
    return out


def parse_test_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(RESULTS.rglob("*_test_results.csv")):
        if "_batch_test" not in str(summary_path): continue
        try: df = pd.read_csv(summary_path)
        except Exception: continue
        ft_root = next((p for p in [summary_path, *summary_path.parents] if p.name.startswith("ft_")), None)
        if ft_root is None: continue
        for rec in df.to_dict("records"):
            run_dir = local_run_dir(ft_root, str(rec.get("weight_path", "")))
            run_name = run_dir.name
            folder = ft_root.name
            camera = rec.get("rgb_camera_id", np.nan)
            item = dict(rec)
            item.update(model_id=short_id(f"{run_dir}|{camera}|{rec.get('test_manifest_used', '')}"),
                        ft_folder=folder, family=family_from_text(folder),
                        family_label=FAMILY_LABELS[family_from_text(folder)],
                        dataset_scope=dataset_scope(folder + " " + str(rec.get("test_manifest_used", ""))),
                        modality=modality_from_text(str(run_dir), str(rec.get("use_modality", ""))),
                        finetune_mode=mode_from_text(str(run_dir)), run_name=run_name,
                        source_key=source_key(run_name), loss_group=loss_group(run_name),
                        run_dir=relpath(run_dir), summary_csv=relpath(summary_path),
                        checkpoint_selection=("best_val" if str(rec.get("weight_name", "")).startswith("best_val") else "last"),
                        camera_id=(int(camera) if pd.notna(camera) else ""))
            item.update(extract_tokens(item["source_key"]))
            cfg_fields = get_config_fields(run_dir)
            item.update(cfg_fields)
            if item["finetune_mode"] == "unknown" and cfg_fields.get("cfg_finetune_mode"):
                item["finetune_mode"] = str(cfg_fields["cfg_finetune_mode"])
            for key in ("test_acc", "test_loss", "test_balanced_acc", "test_macro_f1", "num_samples"):
                item[key] = pd.to_numeric(item.get(key), errors="coerce")
            local_ps = run_dir / Path(str(rec.get("per_sample_csv", ""))).name
            item.update(per_sample_path=relpath(local_ps), per_sample_exists=local_ps.is_file())
            rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty: out = out.drop_duplicates(["weight_path", "test_manifest_used", "camera_id"], keep="last")
    return out


def select_best_available(test: pd.DataFrame) -> pd.DataFrame:
    d = test.copy()
    d["selection_priority"] = np.where(d.checkpoint_selection.eq("best_val"), 0, 1)
    keys = ["run_dir", "test_manifest_used", "camera_id"]
    d = d.sort_values(keys + ["selection_priority"]).drop_duplicates(keys, keep="first")
    d["analysis_view"] = "best_available"
    return d.drop(columns="selection_priority")


def parse_ft_training() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.rglob("summary.json")):
        if "_batch_test" in str(path): continue
        obj = read_json(path)
        if not obj: continue
        run_dir, folder = path.parent, top_folder(path)
        run_name = str(obj.get("run_name", run_dir.name))
        row = {"model_id": short_id(str(run_dir.resolve())), "ft_folder": folder,
               "family": family_from_text(folder), "family_label": FAMILY_LABELS[family_from_text(folder)],
               "dataset_scope": dataset_scope(folder + " " + str(obj.get("train_manifest_used", ""))),
               "modality": modality_from_text(str(run_dir)),
               "finetune_mode": mode_from_text(str(run_dir), str(obj.get("finetune_mode", ""))),
               "run_name": run_name, "source_key": source_key(run_name), "loss_group": loss_group(run_name),
               "run_dir": relpath(run_dir), "summary_path": relpath(path)}
        row.update(extract_tokens(row["source_key"])); row.update(get_config_fields(run_dir))
        keys = ["final_train_acc", "final_train_loss", "final_train_macro_f1", "final_train_balanced_acc",
                "best_val_acc", "best_val_acc_loss", "best_val_acc_epoch", "best_val_macro_f1",
                "best_val_macro_f1_epoch", "best_val_balanced_acc", "best_val_balanced_epoch",
                "final_val_acc", "final_val_loss", "final_val_macro_f1", "final_val_balanced_acc",
                "num_trainable_params", "backbone_learning_rate", "head_learning_rate"]
        for key in keys: row[key] = pd.to_numeric(obj.get(key), errors="coerce")
        row["train_val_acc_gap"] = row["final_train_acc"] - row["final_val_acc"]
        row["best_to_final_balanced_drop"] = row["best_val_balanced_acc"] - row["final_val_balanced_acc"]
        rows.append(row)
    return pd.DataFrame(rows)


def parse_cl_configs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(RESULTS.rglob("args.json")):
        if top_folder(path).startswith("ft_"): continue
        obj = read_json(path)
        if not obj: continue
        folder, run_dir = top_folder(path), path.parent
        run_name = run_dir.name
        keys = ["ablation_mode", "default_num_prototypes", "preview_ema_momentum", "K_queue",
                "rel_topk_diff_classes", "lambda_rel", "temperature", "model_depth", "epochs", "batch_size",
                "optimizer", "learning_rate", "weight_decay", "seed", "sampler_type",
                "balanced_classes_per_batch", "balanced_samples_per_class", "proto_loss_start_epoch",
                "rel_loss_start_epoch", "rel_loss_end_epoch", "warmup_epochs", "rgb_vflip_p", "rgb_hflip_p",
                "rgb_jitter_p", "rgb_gray_p", "rgb_blur_p", "rrc_scale", "rrc_ratio", "mindrove_target_len",
                "mindrove_emg_target_len", "mindrove_imu_target_len", "mindrove_signals"]
        row = {"cl_folder": folder, "family": family_from_text(folder),
               "family_label": FAMILY_LABELS[family_from_text(folder)],
               "dataset_scope": dataset_scope(folder + " " + str(obj.get("train_manifest_name", ""))),
               "modality": modality_from_text(str(run_dir), str(obj.get("use_modality", ""))),
               "run_name": run_name, "source_key": source_key(run_name), "loss_group": loss_group(run_name),
               "run_dir": relpath(run_dir), "config_rel": str(run_dir.relative_to(RESULTS / folder)),
               "args_path": relpath(path)}
        for key in keys: row[key] = json_scalar(obj.get(key))
        rows.append(row)
    all_df = pd.DataFrame(rows)
    all_df["canonical_folder"] = all_df.cl_folder.str.replace(r"^cl_", "", regex=True)
    keys = ["canonical_folder", "modality", "config_rel"]
    all_df["prefer_cl"] = all_df.cl_folder.str.startswith("cl_").astype(int)
    dedup = all_df.sort_values("prefer_cl", ascending=False).drop_duplicates(keys, keep="first")
    duplicates = all_df.groupby(keys, as_index=False).size().query("size > 1")
    return dedup.drop(columns="prefer_cl"), duplicates


def add_baselines(selected: pd.DataFrame) -> pd.DataFrame:
    d = selected.copy()
    for col in ("delta_vs_family_scratch", "delta_vs_base_supcon"): d[col] = np.nan
    d["scratch_reference"] = ""; d["supcon_reference"] = ""
    for idx, row in d.iterrows():
        same = d[(d.ft_folder == row.ft_folder) & (d.modality == row.modality) &
                 (d.dataset_scope == row.dataset_scope) & (d.camera_id.astype(str) == str(row.camera_id))]
        scratch = same[same.finetune_mode.eq("scratch")]
        if len(scratch):
            ref = scratch.sort_values("test_balanced_acc", ascending=False).iloc[0]
            d.at[idx, "delta_vs_family_scratch"] = row.test_balanced_acc - ref.test_balanced_acc
            d.at[idx, "scratch_reference"] = ref.run_name
        base = d[d.family.eq("base") & d.dataset_scope.eq(row.dataset_scope) & d.modality.eq(row.modality) &
                 d.finetune_mode.eq(row.finetune_mode) & d.source_key.str.fullmatch("suploss_only", na=False)]
        if len(base):
            ref = base.sort_values("test_balanced_acc", ascending=False).iloc[0]
            d.at[idx, "delta_vs_base_supcon"] = row.test_balanced_acc - ref.test_balanced_acc
            d.at[idx, "supcon_reference"] = ref.run_name
    return d


def family_summaries(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["dataset_scope", "family", "family_label", "modality", "finetune_mode"]
    rows, tops = [], []
    for keys, g in selected.groupby(cols, dropna=False):
        g = g.sort_values("test_balanced_acc", ascending=False); best = g.iloc[0]
        row = dict(zip(cols, keys)); row.update(n_models=len(g),
            checkpoint_basis=("mixed" if g.checkpoint_selection.nunique() > 1 else g.checkpoint_selection.iloc[0]),
            mean_balanced_acc=g.test_balanced_acc.mean(), std_balanced_acc=g.test_balanced_acc.std(ddof=1),
            median_balanced_acc=g.test_balanced_acc.median(), best_balanced_acc=best.test_balanced_acc,
            best_acc=best.test_acc, best_macro_f1=best.test_macro_f1, best_run=best.run_name,
            best_source_key=best.source_key, best_checkpoint=best.checkpoint_selection,
            best_delta_vs_scratch=best.delta_vs_family_scratch, best_delta_vs_base_supcon=best.delta_vs_base_supcon)
        rows.append(row); tops.extend(g.head(3).to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(tops)


def checkpoint_pairs(test: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_dir", "test_manifest_used", "camera_id"]
    p = test.pivot_table(index=keys, columns="checkpoint_selection",
                         values=["test_acc", "test_balanced_acc", "test_macro_f1"], aggfunc="first")
    if p.empty or not {"best_val", "last"}.issubset(p.columns.get_level_values(1)): return pd.DataFrame()
    p.columns = [f"{a}_{b}" for a, b in p.columns]; p = p.reset_index()
    meta_cols = keys + ["ft_folder", "family", "family_label", "dataset_scope", "modality", "finetune_mode", "run_name", "source_key"]
    meta = test.sort_values("checkpoint_selection").drop_duplicates(keys)[meta_cols]
    p = p.merge(meta, on=keys, how="left")
    for metric in ("test_acc", "test_balanced_acc", "test_macro_f1"):
        p[f"{metric}_best_minus_last"] = p[f"{metric}_best_val"] - p[f"{metric}_last"]
    return p


def parse_per_class(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in selected.to_dict("records"):
        try: obj = json.loads(rec.get("test_per_class_acc_json", "{}"))
        except Exception: obj = {}
        for label, value in obj.items():
            rows.append({"model_id": rec["model_id"], "ft_folder": rec["ft_folder"], "family": rec["family"],
                "dataset_scope": rec["dataset_scope"], "modality": rec["modality"], "finetune_mode": rec["finetune_mode"],
                "run_name": rec["run_name"], "source_key": rec["source_key"],
                "checkpoint_selection": rec["checkpoint_selection"], "camera_id": rec["camera_id"],
                "class_name": label, "class_recall": pd.to_numeric(value, errors="coerce")})
    return pd.DataFrame(rows)


def per_sample_diagnostics(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics, high_conf, confusions = [], [], Counter()
    for rec in selected.to_dict("records"):
        path = ROOT / str(rec.get("per_sample_path", ""))
        if not path.is_file(): continue
        try: d = pd.read_csv(path)
        except Exception: continue
        if not {"correct", "pred_confidence"}.issubset(d.columns): continue
        d.correct = pd.to_numeric(d.correct, errors="coerce").fillna(0)
        d.pred_confidence = pd.to_numeric(d.pred_confidence, errors="coerce")
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (d.pred_confidence >= lo) & ((d.pred_confidence < hi) if hi < 1 else (d.pred_confidence <= hi))
            if mask.any(): ece += mask.mean() * abs(d.loc[mask, "correct"].mean() - d.loc[mask, "pred_confidence"].mean())
        errors = d[d.correct.eq(0)]
        diagnostics.append({"model_id": rec["model_id"], "ft_folder": rec["ft_folder"], "family": rec["family"],
            "dataset_scope": rec["dataset_scope"], "modality": rec["modality"], "finetune_mode": rec["finetune_mode"],
            "run_name": rec["run_name"], "checkpoint_selection": rec["checkpoint_selection"], "camera_id": rec["camera_id"],
            "n_samples": len(d), "ece_10bin": ece, "mean_confidence": d.pred_confidence.mean(),
            "high_confidence_errors": int(((d.correct == 0) & (d.pred_confidence >= 0.8)).sum()),
            "error_rate": 1 - d.correct.mean()})
        if {"true_label_name", "pred_label_name"}.issubset(d.columns):
            for pair, count in errors.groupby(["true_label_name", "pred_label_name"]).size().items():
                confusions[(rec["dataset_scope"], rec["modality"], pair[0], pair[1])] += int(count)
        if len(errors):
            take = errors.nlargest(min(5, len(errors)), "pred_confidence").copy()
            take["model_id"], take["run_name"], take["modality"], take["family"] = rec["model_id"], rec["run_name"], rec["modality"], rec["family"]
            high_conf.append(take)
    confusion_rows = [{"dataset_scope": k[0], "modality": k[1], "true_class": k[2],
                       "predicted_class": k[3], "error_count": v} for k, v in confusions.items()]
    high = pd.concat(high_conf, ignore_index=True) if high_conf else pd.DataFrame()
    confusion = pd.DataFrame(confusion_rows)
    if len(confusion): confusion = confusion.sort_values("error_count", ascending=False)
    return pd.DataFrame(diagnostics), confusion, high


def top_models(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in selected.groupby(["dataset_scope", "modality", "finetune_mode"], dropna=False):
        for rank, (_, rec) in enumerate(g.sort_values("test_balanced_acc", ascending=False).head(5).iterrows(), 1):
            row = rec.to_dict(); row["rank"] = rank; rows.append(row)
    return pd.DataFrame(rows)


def plot_family_best(family: pd.DataFrame, out: Path) -> None:
    d = family[(family.dataset_scope == "except_take_put") & family.finetune_mode.isin(["full", "head_only"])].copy()
    if d.empty: return
    panels = [(m, mode) for m in ("rgb", "emg", "imu") for mode in ("full", "head_only")]
    img, draw = new_canvas(1900, 1800, "Best balanced accuracy by experiment family (except_take_put)")
    for n, (modality, mode) in enumerate(panels):
        col, row = n % 2, n // 2
        left, top = 220 + col*920, 110 + row*550
        right, bottom = left+700, top+450
        g = d[(d.modality == modality) & (d.finetune_mode == mode)].sort_values("best_balanced_acc")
        draw.text(((left+right)//2, top-30), f"{modality.upper()} - {mode}", fill="#17324D", font=font(22, True), anchor="mm")
        hbars(draw, (left, top, right, bottom), g.family_label.tolist(), g.best_balanced_acc.tolist(), xmax=.75)
    img.save(out / "family_best_balanced_accuracy.png", quality=95)


def plot_checkpoint_effect(pairs: pd.DataFrame, out: Path) -> None:
    if pairs.empty: return
    g = pairs.groupby(["family_label", "modality"], as_index=False).test_balanced_acc_best_minus_last.agg(["mean", "median", "count"]).reset_index()
    g = g[pd.to_numeric(g["mean"], errors="coerce").notna()].copy()
    if g.empty: return
    g = g.sort_values("mean")
    labels = (g.family_label + " / " + g.modality.str.upper()).tolist(); vals = g["mean"].tolist()
    img, draw = new_canvas(1650, max(650, 100+len(g)*58), "Checkpoint selection effect: best_val - last")
    hbars(draw, (440, 100, 1500, img.height-70), labels, vals, [GREEN if v>=0 else RED for v in vals], xmin=min(-.08,min(vals)), xmax=max(.08,max(vals)), value_fmt="{:+.3f}")
    img.save(out / "checkpoint_selection_effect.png", quality=95)


def plot_hyperparameters(selected: pd.DataFrame, out: Path) -> None:
    img, draw = new_canvas(1800, 720, "Hyperparameter sensitivity")
    k = selected[(selected.family == "random_kqueue") & selected.source_key.str.contains("suploss_only", na=False)].copy()
    k_series=[]; k_labels=[]
    if len(k):
        k["K"] = pd.to_numeric(k.queue_size_name, errors="coerce")
        k_labels=[str(int(v)) for v in sorted(k.K.dropna().unique())]
        for modality, g in k.groupby("modality"):
            q = g.groupby("K", as_index=False).test_balanced_acc.mean().set_index("K")
            k_series.append((modality.upper(),[q.test_balanced_acc.get(float(v),np.nan) for v in k_labels],PALETTE[len(k_series)]))
    draw.text((470,105),"Queue-size sweep (SupCon-only)",fill="#17324D",font=font(22,True),anchor="mm")
    line_plot(draw,(120,150,820,610),k_series,k_labels)
    t = selected[selected.family.isin(["relation_topk", "stage5_topk"]) & selected.topk_name.notna()].copy()
    t_series=[]; t_labels=[]
    if len(t):
        topks=sorted(t.topk_name.dropna().unique()); t_labels=["all" if v==999 else str(int(v)) for v in topks]
        for modality, g in t.groupby("modality"):
            q=g.groupby("topk_name",as_index=False).test_balanced_acc.mean().set_index("topk_name")
            t_series.append((modality.upper(),[q.test_balanced_acc.get(v,np.nan) for v in topks],PALETTE[len(t_series)]))
    draw.text((1330,105),"Relation Top-k sweep",fill="#17324D",font=font(22,True),anchor="mm")
    line_plot(draw,(980,150,1680,610),t_series,t_labels)
    img.save(out / "hyperparameter_sensitivity.png", quality=95)


def plot_class_delta(selected: pd.DataFrame, per_class: pd.DataFrame, out: Path) -> None:
    rows = []
    for modality in ("rgb", "emg", "imu"):
        candidates = selected[(selected.dataset_scope == "except_take_put") & (selected.modality == modality) & selected.finetune_mode.isin(["full", "head_only"])]
        scratch = selected[(selected.dataset_scope == "except_take_put") & (selected.modality == modality) & (selected.finetune_mode == "scratch")]
        if candidates.empty or scratch.empty: continue
        best, ref = candidates.sort_values("test_balanced_acc", ascending=False).iloc[0], scratch.sort_values("test_balanced_acc", ascending=False).iloc[0]
        a = per_class[per_class.model_id == best.model_id].set_index("class_name").class_recall
        b = per_class[per_class.model_id == ref.model_id].set_index("class_name").class_recall
        rows.extend({"modality": modality.upper(), "class_name": c, "delta": a.get(c, np.nan)-b.get(c, np.nan)} for c in sorted(set(a.index)|set(b.index)))
    d = pd.DataFrame(rows)
    if d.empty: return
    p = d.pivot(index="modality", columns="class_name", values="delta")
    cell_w, cell_h = 95, 95
    img, draw = new_canvas(max(1500,250+len(p.columns)*cell_w), 560, "Per-class recall change: best pretrained model vs scratch")
    x0,y0=220,150
    for j,c in enumerate(p.columns): draw.text((x0+j*cell_w+cell_w/2,y0-12),str(c),fill="#333",font=font(14),anchor="ls")
    for i in range(len(p.index)):
        draw.text((x0-15,y0+i*cell_h+cell_h/2),str(p.index[i]),fill="#333",font=font(18,True),anchor="rm")
        for j in range(len(p.columns)):
            v=p.iloc[i,j]
            if pd.isna(v): color="#EEEEEE"
            elif v>=0: color=(int(245-80*min(v,.5)/.5),int(245-10*min(v,.5)/.5),int(245-90*min(v,.5)/.5))
            else: color=(int(245-10*min(-v,.5)/.5),int(245-100*min(-v,.5)/.5),int(245-100*min(-v,.5)/.5))
            box=(x0+j*cell_w,y0+i*cell_h,x0+(j+1)*cell_w-3,y0+(i+1)*cell_h-3); draw.rectangle(box,fill=color)
            if pd.notna(v): draw.text(((box[0]+box[2])/2,(box[1]+box[3])/2),f"{v:+.2f}",fill="#222",font=font(15),anchor="mm")
    img.save(out / "class_recall_delta_vs_scratch.png", quality=95)


def plot_feature_gap(out: Path) -> pd.DataFrame:
    source = OLD_ANALYSIS / "06_statistics" / "pilot_feature_distance_summary.csv"
    if not source.is_file(): return pd.DataFrame()
    d = pd.read_csv(source); d = d[d.feature_dim.eq(512)].copy()
    p = d.pivot_table(index=["model_id", "stage", "modality"], columns="split", values="silhouette_cosine", aggfunc="first").reset_index()
    if {"train", "test"}.issubset(p.columns): p["train_test_gap"] = p.train-p.test
    img, draw = new_canvas(1800,800,"Feature separation generalization gap (reused pilot features, 512D)")
    x0,y0,x1,y1=120,120,1700,650; ymin=min(-.3,float(p[["train","test"]].min().min())-.05); ymax=max(1.0,float(p[["train","test"]].max().max())+.05)
    for tick in np.linspace(ymin, ymax, 6):
        ty=y1-(tick-ymin)/(ymax-ymin)*(y1-y0); draw.line((x0,ty,x1,ty),fill="#E8ECF0",width=1); draw.text((x0-10,ty),f"{tick:.2f}",fill="#444",font=font(14),anchor="rm")
    n=len(p); group_w=(x1-x0)/n; zero=y1-(0-ymin)/(ymax-ymin)*(y1-y0); draw.line((x0,zero,x1,zero),fill="#444",width=2)
    for i,rec in p.iterrows():
        cx=x0+group_w*(i+.5)
        for off,key,color in [(-.16,"train",BLUE),(.16,"test",ORANGE)]:
            v=rec.get(key,np.nan); bx=cx+off*group_w; by=y1-(v-ymin)/(ymax-ymin)*(y1-y0); draw.rectangle((bx-group_w*.12,min(zero,by),bx+group_w*.12,max(zero,by)),fill=color)
        draw.text((cx,y1+12),rec.model_id,fill="#333",font=font(14),anchor="ma")
    draw.rectangle((1450,90,1470,108),fill=BLUE); draw.text((1480,99),"Train",fill="#333",font=font(15),anchor="lm"); draw.rectangle((1550,90,1570,108),fill=ORANGE); draw.text((1580,99),"Test",fill="#333",font=font(15),anchor="lm")
    img.save(out / "feature_separation_train_test.png", quality=95); return p


def plot_round2_dualcam(selected: pd.DataFrame, out: Path) -> None:
    img, draw = new_canvas(1900,850,"Focused RGB follow-up experiments")
    r = selected[selected.family=="round2"].sort_values("test_balanced_acc")
    draw.text((470,105),"Round-2 action-preserving augmentation",fill="#17324D",font=font(21,True),anchor="mm")
    if len(r): hbars(draw,(330,145,850,760),r.source_key.tolist(),r.test_balanced_acc.tolist(),[BLUE]*len(r),xmax=.75)
    d=selected[selected.family=="dualcam"].copy()
    draw.text((1420,105),"Dual-camera mean across held-out cameras",fill="#17324D",font=font(21,True),anchor="mm")
    if len(d):
        q=d.pivot_table(index="run_name",columns="camera_id",values="test_balanced_acc",aggfunc="first"); q["mean"]=q.mean(axis=1); q=q.sort_values("mean")
        labels=[source_key(v) for v in q.index]
        hbars(draw,(1240,145,1800,760),labels,q["mean"].tolist(),[ORANGE]*len(q),xmax=.75)
    img.save(out/"round2_and_dualcam.png",quality=95)


def copy_selected_umaps(out: Path) -> list[str]:
    picks={"rgb_head_test_umap.png":OLD_ANALYSIS/"02_umap"/"rgb_head"/"test"/"f512"/"umap_2d.png",
           "rgb_full_test_umap.png":OLD_ANALYSIS/"02_umap"/"rgb_full"/"test"/"f512"/"umap_2d.png"}
    copied=[]
    for name,src in picks.items():
        if src.is_file(): shutil.copy2(src,out/name); copied.append(name)
    return copied


def build_quality_issues(test: pd.DataFrame, selected: pd.DataFrame, ft: pd.DataFrame, dup: pd.DataFrame) -> pd.DataFrame:
    seeds=sorted(set(pd.to_numeric(ft.get("cfg_seed"),errors="coerce").dropna().astype(int)))
    rgb_scratch=test[(test.dataset_scope=="except_take_put") & (test.modality=="rgb") & (test.finetune_mode=="scratch")]
    scratch_evidence=(f"相同 seed=1、表面配置相同的 RGB scratch：last balanced accuracy "
                      f"{rgb_scratch[rgb_scratch.checkpoint_selection=='last'].test_balanced_acc.min():.3f}-"
                      f"{rgb_scratch[rgb_scratch.checkpoint_selection=='last'].test_balanced_acc.max():.3f}；"
                      f"best_val 为 {rgb_scratch[rgb_scratch.checkpoint_selection=='best_val'].test_balanced_acc.min():.3f}-"
                      f"{rgb_scratch[rgb_scratch.checkpoint_selection=='best_val'].test_balanced_acc.max():.3f}。")
    issues=[
        ["High","缺少多随机种子",f"FT 配置中的非空 seed 仅为 {seeds}。","无法估计方差或判断单次结果是否偶然。","对候选最佳、scratch、SupCon 基线补跑 3-5 个 seed。"],
        ["High","RGB scratch 跨实验族不一致",scratch_evidence,"实验族间差异可能包含运行环境、代码版本或未记录随机性的影响。","固定代码 commit、cuDNN 确定性和数据顺序；使用同一 scratch checkpoint 作为共享基线。"],
        ["High","checkpoint 选择不一致",f"测试汇总包含 {int(test.checkpoint_selection.eq('best_val').sum())} 条 best_val 和 {int(test.checkpoint_selection.eq('last').sum())} 条 last；基础与 take_put 组主要只有 last。","直接混排会把模型差异与 checkpoint 选择差异混在一起。","补测基础组 best_val；当前报告保留严格 last 与 best-available 两套视图。"],
        ["Medium","RGB CL 目录重复",f"检测到 {len(dup)} 个 canonical 配置同时出现在 cl_ 与非 cl_ 目录。","会造成实验数量翻倍和存储浪费。","保留一个权威路径，并在 manifest 中记录别名/哈希。"],
        ["Medium","RGB 基础增强包含垂直翻转","基础 RGB 配置记录 rgb_vflip_p=0.5，而动作语义保持/双相机后续配置使用更保守增强。","垂直方向敏感动作可能被错误增强。","用 vflip=0 的同数据划分消融验证，重点检查 insert/pull_out/wrap。"],
        ["Check","MindRove take_put 标签映射需核对","CL args 可见通用 label_map.json，而 FT 使用 take_put 专用映射。","若 ID/过滤逻辑不一致，预训练与下游标签语义可能错位。","保存并逐项核对最终 action->id 映射。"],
    ]
    missing=int((~selected.per_sample_exists.astype(bool)).sum())
    if missing: issues.append(["Medium","逐样本文件缺失",f"最佳可用视图有 {missing} 条记录未找到逐样本 CSV。","无法完成混淆与校准回溯。","补齐 per-sample 输出并固定命名。"])
    return pd.DataFrame(issues,columns=["severity","issue","evidence","impact","recommended_action"])


def compact_summary(test, selected, family, pairs, ft, cl, dup, diagnostics, confusion, feature, umaps):
    best_rows=[]
    for keys,g in selected.groupby(["dataset_scope","modality","finetune_mode"],dropna=False):
        rec=g.sort_values("test_balanced_acc",ascending=False).iloc[0]
        best_rows.append({"dataset_scope":keys[0],"modality":keys[1],"finetune_mode":keys[2],"family":rec.family,"family_label":rec.family_label,"run_name":rec.run_name,"checkpoint":rec.checkpoint_selection,"balanced_acc":rec.test_balanced_acc,"accuracy":rec.test_acc,"macro_f1":rec.test_macro_f1,"delta_vs_scratch":rec.delta_vs_family_scratch,"delta_vs_base_supcon":rec.delta_vs_base_supcon})
    pair_summary=[]
    if len(pairs):
        for keys,g in pairs.groupby(["family","modality"]): pair_summary.append({"family":keys[0],"modality":keys[1],"n":len(g),"mean_delta_balanced":g.test_balanced_acc_best_minus_last.mean(),"median_delta_balanced":g.test_balanced_acc_best_minus_last.median()})
    diag=[]
    if len(diagnostics): diag=diagnostics.groupby("modality",as_index=False).agg(n_models=("model_id","count"),mean_ece=("ece_10bin","mean"),mean_confidence=("mean_confidence","mean"),mean_error_rate=("error_rate","mean"),high_confidence_errors=("high_confidence_errors","sum")).to_dict("records")
    return {"generated_at":pd.Timestamp.now().isoformat(),"inventory":{"cl_args_files_raw":sum(1 for _ in RESULTS.rglob("args.json")),"cl_configs_deduplicated":len(cl),"duplicate_cl_configs":len(dup),"ft_training_runs":len(ft),"test_metric_rows":len(test),"best_val_test_rows":int(test.checkpoint_selection.eq("best_val").sum()),"last_test_rows":int(test.checkpoint_selection.eq("last").sum()),"selected_models_or_camera_views":len(selected),"per_sample_available":int(selected.per_sample_exists.astype(bool).sum())},"best_models":best_rows,"family_best":family.sort_values("best_balanced_acc",ascending=False).head(50).to_dict("records") if len(family) else [],"checkpoint_effect":pair_summary,"diagnostics_by_modality":diag,"top_confusions":confusion.head(20).to_dict("records") if len(confusion) else [],"feature_separation":feature.to_dict("records") if len(feature) else [],"selected_umaps":umaps}


def write_csv(df: pd.DataFrame, path: Path) -> None: df.to_csv(path,index=False,encoding="utf-8-sig")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=DEFAULT_OUT); args=parser.parse_args(); out=args.out.resolve(); tables=out/"tables"; figures=out/"figures"; tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    test=parse_test_metrics(); selected=add_baselines(select_best_available(test)); last=add_baselines(test[test.checkpoint_selection.eq("last")].copy()); ft=parse_ft_training(); cl,duplicates=parse_cl_configs(); family,family_top=family_summaries(selected); pairs=checkpoint_pairs(test); per_class=parse_per_class(selected); diagnostics,confusion,high_conf=per_sample_diagnostics(selected); tops=top_models(selected); issues=build_quality_issues(test,selected,ft,duplicates)
    for df,name in [(test,"all_test_metrics.csv"),(selected,"selected_best_available.csv"),(last,"strict_last_checkpoint.csv"),(ft,"ft_training_runs.csv"),(cl,"cl_configs_deduplicated.csv"),(duplicates,"duplicate_cl_configs.csv"),(family,"family_summary.csv"),(family_top,"family_top3.csv"),(pairs,"checkpoint_pairs.csv"),(per_class,"per_class_recall.csv"),(diagnostics,"model_diagnostics.csv"),(confusion,"confusion_pairs.csv"),(high_conf,"high_confidence_error_examples.csv"),(tops,"top_models.csv"),(issues,"quality_issues.csv")]: write_csv(df,tables/name)
    plot_family_best(family,figures); plot_checkpoint_effect(pairs,figures); plot_hyperparameters(selected,figures); plot_class_delta(selected,per_class,figures); feature=plot_feature_gap(figures); write_csv(feature,tables/"feature_separation_pilot.csv"); plot_round2_dualcam(selected,figures); umaps=copy_selected_umaps(figures)
    summary=compact_summary(test,selected,family,pairs,ft,cl,duplicates,diagnostics,confusion,feature,umaps)
    summary["analysis_runtime"] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "plot_backend": "Pillow (deterministic static charts)",
    }
    (out/"analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (out/"SNAPSHOT.txt").write_text(
        f"Results snapshot analyzed at {summary['generated_at']}\nSource: {RESULTS}\n"
        f"Python: {sys.executable}\nThe source results directory was not modified.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["inventory"],ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
