#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 3D UMAP visualisation for RGB finetuned last.pth checkpoints.

The script reuses the 2D analysis helpers, computes UMAP with n_components=3,
and writes self-contained HTML files with a small native Canvas viewer. No
Plotly or network access is required to open the outputs.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import umap_rgb_lastpth_by_label as base


TAB20 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def compute_umap3d(x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap
    if x.shape[0] < 4:
        raise ValueError("3D UMAP needs at least 4 samples.")
    n_neighbors = min(args.umap_n_neighbors, x.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    return reducer.fit_transform(x).astype(np.float32)


def save_coords_csv_3d(
    path: Path,
    coords: np.ndarray,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> None:
    rows: List[Dict[str, Any]] = []
    for i in range(coords.shape[0]):
        rows.append({
            "sample_id": str(sample_ids[i]),
            "split": str(splits[i]),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
            "true_label_id": int(labels[i]),
            "true_label_name": str(label_names[i]),
        })
    base.save_csv(path, rows)


def build_points_json(
    coords: np.ndarray,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for i in range(coords.shape[0]):
        points.append({
            "id": str(sample_ids[i]),
            "split": str(splits[i]),
            "label": int(labels[i]),
            "name": str(label_names[i]),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
        })
    return points


def write_interactive_html(
    path: Path,
    *,
    title: str,
    coords: np.ndarray,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
) -> None:
    base.ensure_dir(path.parent)
    points = build_points_json(coords, sample_ids, splits, labels, label_names)
    unique_labels = []
    for lab, name in zip(labels.astype(int).tolist(), label_names.tolist()):
        if not any(item["label"] == lab for item in unique_labels):
            unique_labels.append({
                "label": int(lab),
                "name": str(name),
                "color": TAB20[len(unique_labels) % len(TAB20)],
            })
    payload = {
        "title": title,
        "points": points,
        "labels": unique_labels,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    safe_title = html.escape(title)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
  :root {{
    color-scheme: light;
    --bg: #f6f7f9;
    --panel: #ffffff;
    --ink: #1f2933;
    --muted: #697586;
    --line: #d8dee8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    color: var(--ink);
    background: var(--bg);
  }}
  .app {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    height: 100vh;
    min-height: 620px;
  }}
  .stage {{
    position: relative;
    min-width: 0;
    background: #fbfcfe;
    border-right: 1px solid var(--line);
  }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}
  aside {{
    background: var(--panel);
    padding: 16px;
    overflow: auto;
  }}
  h1 {{
    font-size: 15px;
    line-height: 1.35;
    margin: 0 0 12px;
    font-weight: 700;
  }}
  .sub {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
  }}
  .toolbar {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
  }}
  button {{
    border: 1px solid var(--line);
    background: #fff;
    color: var(--ink);
    border-radius: 6px;
    padding: 7px 8px;
    font-size: 12px;
    cursor: pointer;
  }}
  button:hover {{ background: #f0f3f8; }}
  .legend {{
    display: grid;
    gap: 8px;
  }}
  label {{
    display: grid;
    grid-template-columns: 16px 14px 1fr;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    line-height: 1.25;
  }}
  .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 1px solid rgba(0,0,0,.15);
  }}
  .tip {{
    position: absolute;
    min-width: 220px;
    max-width: 340px;
    pointer-events: none;
    display: none;
    padding: 9px 10px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.96);
    border-radius: 6px;
    box-shadow: 0 8px 24px rgba(15,23,42,.15);
    font-size: 12px;
    line-height: 1.4;
  }}
  .tip strong {{ display: block; margin-bottom: 3px; }}
  .hint {{
    position: absolute;
    left: 14px;
    bottom: 12px;
    padding: 6px 8px;
    color: #4b5565;
    font-size: 12px;
    background: rgba(255,255,255,.82);
    border: 1px solid var(--line);
    border-radius: 6px;
  }}
  @media (max-width: 860px) {{
    .app {{ grid-template-columns: 1fr; grid-template-rows: minmax(420px, 1fr) auto; }}
    .stage {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    aside {{ max-height: 280px; }}
  }}
</style>
</head>
<body>
<div class="app">
  <main class="stage">
    <canvas id="scene"></canvas>
    <div id="tip" class="tip"></div>
    <div class="hint">Drag to rotate · Wheel to zoom · Hover points for details</div>
  </main>
  <aside>
    <h1>{safe_title}</h1>
    <div class="sub"><span id="count"></span> points · true-label colouring · 3D UMAP</div>
    <div class="toolbar">
      <button id="reset">Reset View</button>
      <button id="toggle">Toggle All</button>
    </div>
    <div id="legend" class="legend"></div>
  </aside>
</div>
<script>
const payload = {payload_json};
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const tip = document.getElementById('tip');
const stage = document.querySelector('.stage');
const visible = new Map(payload.labels.map(d => [d.label, true]));
const colors = new Map(payload.labels.map(d => [d.label, d.color]));
let rotX = -0.48;
let rotY = 0.66;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let projected = [];

function centerAndScale(points) {{
  const xs = points.map(p => p.x), ys = points.map(p => p.y), zs = points.map(p => p.z);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
  let maxR = 1e-6;
  for (const p of points) {{
    p.cx = p.x - cx;
    p.cy = p.y - cy;
    p.cz = p.z - cz;
    maxR = Math.max(maxR, Math.hypot(p.cx, p.cy, p.cz));
  }}
  for (const p of points) {{
    p.cx /= maxR;
    p.cy /= maxR;
    p.cz /= maxR;
  }}
}}

function resize() {{
  const rect = stage.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function rotatePoint(p) {{
  const sx = Math.sin(rotX), cx = Math.cos(rotX);
  const sy = Math.sin(rotY), cy = Math.cos(rotY);
  let y = p.cy * cx - p.cz * sx;
  let z = p.cy * sx + p.cz * cx;
  let x = p.cx * cy + z * sy;
  z = -p.cx * sy + z * cy;
  return {{x, y, z}};
}}

function draw() {{
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#fbfcfe';
  ctx.fillRect(0, 0, w, h);
  const scale = Math.min(w, h) * 0.39 * zoom;
  const cx = w / 2, cy = h / 2;
  projected = [];
  for (const p of payload.points) {{
    if (!visible.get(p.label)) continue;
    const r = rotatePoint(p);
    projected.push({{
      p,
      sx: cx + r.x * scale,
      sy: cy - r.y * scale,
      depth: r.z
    }});
  }}
  projected.sort((a, b) => a.depth - b.depth);
  for (const item of projected) {{
    const radius = 3.0 + (item.depth + 1) * 0.9;
    ctx.globalAlpha = 0.80;
    ctx.fillStyle = colors.get(item.p.label) || '#444';
    ctx.beginPath();
    ctx.arc(item.sx, item.sy, radius, 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;
  document.getElementById('count').textContent = projected.length + ' / ' + payload.points.length;
}}

function nearestPoint(mx, my) {{
  let best = null;
  let bestD = 13 * 13;
  for (const item of projected) {{
    const d = (item.sx - mx) ** 2 + (item.sy - my) ** 2;
    if (d < bestD) {{
      bestD = d;
      best = item;
    }}
  }}
  return best;
}}

function showTip(item, ev) {{
  if (!item) {{
    tip.style.display = 'none';
    return;
  }}
  const p = item.p;
  tip.innerHTML =
    '<strong>' + escapeHtml(p.name) + ' [' + p.label + ']</strong>' +
    'split: ' + escapeHtml(p.split) + '<br>' +
    'sample: ' + escapeHtml(p.id) + '<br>' +
    'x/y/z: ' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ', ' + p.z.toFixed(3);
  const rect = stage.getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - rect.left + 14, rect.width - 360) + 'px';
  tip.style.top = Math.max(8, ev.clientY - rect.top + 14) + 'px';
  tip.style.display = 'block';
}}

function escapeHtml(text) {{
  return String(text).replace(/[&<>"']/g, c => ({{
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
  }}[c]));
}}

function buildLegend() {{
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  for (const item of payload.labels) {{
    const row = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', () => {{
      visible.set(item.label, cb.checked);
      draw();
    }});
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = item.color;
    const text = document.createElement('span');
    text.textContent = item.name + ' [' + item.label + ']';
    row.appendChild(cb);
    row.appendChild(sw);
    row.appendChild(text);
    legend.appendChild(row);
  }}
}}

canvas.addEventListener('mousedown', ev => {{
  dragging = true;
  lastX = ev.clientX;
  lastY = ev.clientY;
}});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', ev => {{
  if (dragging) {{
    rotY += (ev.clientX - lastX) * 0.008;
    rotX += (ev.clientY - lastY) * 0.008;
    lastX = ev.clientX;
    lastY = ev.clientY;
    tip.style.display = 'none';
    draw();
  }} else {{
    const rect = canvas.getBoundingClientRect();
    showTip(nearestPoint(ev.clientX - rect.left, ev.clientY - rect.top), ev);
  }}
}});
canvas.addEventListener('mouseleave', () => tip.style.display = 'none');
canvas.addEventListener('wheel', ev => {{
  ev.preventDefault();
  zoom *= Math.exp(-ev.deltaY * 0.001);
  zoom = Math.min(8, Math.max(0.25, zoom));
  draw();
}}, {{passive: false}});
document.getElementById('reset').addEventListener('click', () => {{
  rotX = -0.48; rotY = 0.66; zoom = 1.0; draw();
}});
document.getElementById('toggle').addEventListener('click', () => {{
  const anyVisible = Array.from(visible.values()).some(Boolean);
  for (const key of visible.keys()) visible.set(key, !anyVisible);
  for (const cb of document.querySelectorAll('#legend input')) cb.checked = !anyVisible;
  draw();
}});
window.addEventListener('resize', resize);

centerAndScale(payload.points);
buildLegend();
resize();
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def visualize_split_3d(
    out_dir: Path,
    group_key: str,
    weight_slug: str,
    split_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    label_names: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, str]:
    x = base.l2_normalize(features) if args.feature_l2_normalize else features.astype(np.float32, copy=False)
    x = base.maybe_pca(x, args.pca_dim, args.seed)
    coords = compute_umap3d(x, args)
    coords_csv = out_dir / f"{split_name}_umap3d_coords.csv"
    html_path = out_dir / f"{split_name}_umap3d_by_label_interactive.html"
    splits = np.array([split_name] * len(labels), dtype=str)
    if split_name == "combined":
        splits = np.array(["train"] * args._combined_train_n + ["test"] * (len(labels) - args._combined_train_n), dtype=str)
    save_coords_csv_3d(coords_csv, coords, sample_ids, splits, labels, label_names)
    write_interactive_html(
        html_path,
        title=f"{group_key} | {weight_slug} | {split_name} | 3D UMAP by true label",
        coords=coords,
        sample_ids=sample_ids,
        splits=splits,
        labels=labels,
        label_names=label_names,
    )
    return {"coords_csv": str(coords_csv), "html": str(html_path)}


def analyze_weight_3d(
    group: base.GroupConfig,
    weight_path: Path,
    loaders: Dict[str, Any],
    reverse_label_map: Dict[int, str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    slug = base.run_slug(weight_path, group)
    out_dir = args.output_dir / group.key / slug
    base.ensure_dir(out_dir)
    print(f"\n[weight] {group.key} | {slug}")
    print(f"[path] {weight_path}")
    model = base.make_model(group.num_classes, args, device)
    load_report = base.load_checkpoint(model, weight_path)
    print(json.dumps(load_report, indent=2))

    train_features, train_labels, train_ids = base.extract_features(model, loaders["train"], device, args.tier_mode, "train")
    test_features, test_labels, test_ids = base.extract_features(model, loaders["test"], device, args.tier_mode, "test")
    train_names = base.label_names_from_ids(train_labels, reverse_label_map)
    test_names = base.label_names_from_ids(test_labels, reverse_label_map)

    outputs: Dict[str, Any] = {}
    outputs["train"] = visualize_split_3d(out_dir, group.key, slug, "train", train_features, train_labels, train_ids, train_names, args)
    outputs["test"] = visualize_split_3d(out_dir, group.key, slug, "test", test_features, test_labels, test_ids, test_names, args)

    args._combined_train_n = int(train_features.shape[0])
    combined_features = np.concatenate([train_features, test_features], axis=0)
    combined_labels = np.concatenate([train_labels, test_labels], axis=0)
    combined_ids = np.concatenate([train_ids, test_ids], axis=0)
    combined_names = np.concatenate([train_names, test_names], axis=0)
    outputs["combined"] = visualize_split_3d(out_dir, group.key, slug, "combined", combined_features, combined_labels, combined_ids, combined_names, args)

    meta = {
        "group": group.key,
        "weight_path": str(weight_path),
        "weight_slug": slug,
        "num_classes": group.num_classes,
        "rgb_mean": group.rgb_mean,
        "rgb_std": group.rgb_std,
        "train_manifest": group.train_manifest,
        "test_manifest": group.test_manifest,
        "load_report": load_report,
        "num_train_samples": int(train_features.shape[0]),
        "num_test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "outputs": outputs,
    }
    base.save_json(out_dir / "umap3d_meta.json", meta)
    return meta


def build_argparser() -> argparse.ArgumentParser:
    p = base.build_argparser()
    p.description = "Interactive 3D RGB last.pth UMAP visualisation by true label."
    p.set_defaults(output_dir=PROJECT_ROOT / "analysis" / "umap_rgb_last_3d")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir = args.output_dir.resolve()
    base.ensure_dir(args.output_dir)
    base.seed_everything(args.seed)
    base.save_json(args.output_dir / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "project_root": str(PROJECT_ROOT),
        "umap_components": 3,
        "viewer": "native_canvas_3d",
    })

    selected = set(base.parse_group_keys(args.groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[output] {args.output_dir}")

    summary: List[Dict[str, Any]] = []
    t0 = time.time()
    for group in base.GROUPS:
        if group.key not in selected:
            continue
        print(f"\n[group] {group.key}")
        reverse_label_map = base.build_reverse_label_map(group.label_map_json, args.tier_mode)
        loaders = {
            "train": base.build_loader(group, group.train_manifest, args),
            "test": base.build_loader(group, group.test_manifest, args),
        }
        weights = base.find_last_checkpoints(group)
        if not weights:
            raise FileNotFoundError(f"No last.pth files found under {group.result_dir / 'weights'}")
        print(f"[weights] found {len(weights)} last.pth files")
        for weight_path in weights:
            meta = analyze_weight_3d(group, weight_path, loaders, reverse_label_map, args, device)
            summary.append({
                "group": meta["group"],
                "weight_slug": meta["weight_slug"],
                "weight_path": meta["weight_path"],
                "num_train_samples": meta["num_train_samples"],
                "num_test_samples": meta["num_test_samples"],
                "feature_dim": meta["feature_dim"],
                "train_html": meta["outputs"]["train"]["html"],
                "test_html": meta["outputs"]["test"]["html"],
                "combined_html": meta["outputs"]["combined"]["html"],
            })
            base.save_csv(args.output_dir / "summary_partial.csv", summary)

    base.save_csv(args.output_dir / "summary.csv", summary)
    base.save_json(args.output_dir / "summary.json", {"rows": summary, "elapsed_seconds": time.time() - t0})
    print(f"\n[done] elapsed seconds: {time.time() - t0:.1f}")
    print(f"[summary] {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
