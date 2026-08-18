#!/usr/bin/env python3
"""Checkpoint-level temporal perturbation diagnostics for Take/Put SupLoss models.

The frozen linear probes are fitted only on chronological M/MR/J features.  The
held-out N clips are then evaluated under deterministic, sample-specific temporal
perturbations without changing the spatial pixels or retraining the encoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch


PERTURBATIONS = (
    "original",
    "reverse",
    "global_shuffle",
    "block4_shuffle",
    "within_block_shuffle",
    "repeat_center",
    "temporal_mean_repeat",
)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def checkpoint_state(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise TypeError(f"{path} does not contain a state dict")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in obj.items()}


def load_encoder(src_root: Path, backbone: str, init: str, checkpoint: Path, device: torch.device):
    from codex_script.rgb_supcon_repair_20260806.common.runtime_patch import install

    install(src_root, "rgb", "current", init, False)
    from backbone.MoCo_VAR_supcon_wds import MoCo3D
    from backbone.video_backbone import generate_video_model

    model = MoCo3D(
        partial(generate_video_model, backbone_name=backbone, model_depth=18),
        dim=128,
        K=1088,
        T=0.07,
        mlp=True,
        exclude_invalid_queue=True,
    )
    message = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    unexpected = [key for key in message.unexpected_keys if not key.startswith("round2_aux_classifier")]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys in {checkpoint}: {unexpected[:10]}")
    return model.encoder_q.to(device).eval()


def build_loader(args, manifest: Path):
    from utils_.mapstype_dataloader_with_index import (
        PackedMultiModalConfig,
        build_packed_mapstyle_dataset,
        build_packed_mapstyle_loader_from_dataset,
        load_label_map_json,
    )

    label_map = load_label_map_json(str(args.label_map))
    cfg = PackedMultiModalConfig(
        n_frames=16,
        rgb_two_views=False,
        rgb_camera_id="00143",
        use_modalities=("rgb",),
        load_labels=True,
        label_map_path=str(args.label_map),
        tier_mode="tier1",
        is_train=False,
        rgb_out_hw=(224, 224),
        rgb_mean=(0.45, 0.45, 0.45),
        rgb_std=(0.225, 0.225, 0.225),
    )
    dataset = build_packed_mapstyle_dataset(
        args.dataset_root, str(manifest), cfg, label_map=label_map, verify_paths_on_init=True
    )
    return build_packed_mapstyle_loader_from_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers else None,
    )


def nonidentity_permutation(rng: np.random.Generator, size: int) -> np.ndarray:
    result = rng.permutation(size)
    if np.array_equal(result, np.arange(size)):
        result = np.roll(result, 1)
    return result


def sample_permutations(indices: np.ndarray, mode: str, length: int, seed: int, device) -> torch.Tensor:
    rows = []
    for sample_index in indices.tolist():
        mode_offset = {"global_shuffle": 11, "block4_shuffle": 23, "within_block_shuffle": 37}[mode]
        rng = np.random.default_rng(seed + 1_000_003 * int(sample_index) + mode_offset)
        if mode == "global_shuffle":
            order = nonidentity_permutation(rng, length)
        elif mode == "block4_shuffle":
            if length % 4:
                raise ValueError("4-block shuffle requires a frame count divisible by four")
            blocks = np.arange(length).reshape(4, length // 4)
            order = blocks[nonidentity_permutation(rng, 4)].reshape(-1)
        elif mode == "within_block_shuffle":
            if length % 4:
                raise ValueError("Within-block shuffle requires a frame count divisible by four")
            pieces = []
            width = length // 4
            for block in range(4):
                local = nonidentity_permutation(rng, width) + block * width
                pieces.append(local)
            order = np.concatenate(pieces)
        else:
            raise ValueError(mode)
        rows.append(order)
    return torch.as_tensor(np.stack(rows), dtype=torch.long, device=device)


def perturb_video(x: torch.Tensor, indices: np.ndarray, mode: str, seed: int) -> torch.Tensor:
    # x is [B,C,T,H,W]. All operations preserve the 16-frame tensor shape.
    if mode == "original":
        return x
    if mode == "reverse":
        return x.flip(2)
    if mode == "repeat_center":
        center = x.shape[2] // 2
        return x[:, :, center : center + 1].expand_as(x).contiguous()
    if mode == "temporal_mean_repeat":
        return x.mean(dim=2, keepdim=True).expand_as(x).contiguous()
    order = sample_permutations(indices, mode, x.shape[2], seed, x.device)
    gather_index = order[:, None, :, None, None].expand(-1, x.shape[1], -1, x.shape[3], x.shape[4])
    return torch.gather(x, 2, gather_index)


def pooled_activation(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported hook output: {type(output)}")
    if output.ndim == 5:  # R3D: [B,C,T,H,W]
        return output.mean(dim=(2, 3, 4))
    if output.ndim == 3:  # MViT block tokens: [B,THW,C]
        return output.mean(dim=1)
    if output.ndim == 2:
        return output
    if output.ndim == 4:
        return output.mean(dim=(1, 2))
    raise ValueError(f"Unsupported activation shape {tuple(output.shape)}")


def register_stage_hooks(encoder, backbone: str):
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def save(_module, _inputs, output):
            captured[name] = pooled_activation(output).detach().float().cpu()

        return save

    if backbone == "tv_r3d18":
        targets = [(f"layer{i}", getattr(encoder.backbone, f"layer{i}")) for i in range(1, 5)]
    else:
        blocks = encoder.backbone.blocks
        selected = (0, len(blocks) // 2, len(blocks) - 1)
        targets = [
            ("early_block_00", blocks[selected[0]]),
            (f"middle_block_{selected[1]:02d}", blocks[selected[1]]),
            (f"late_block_{selected[2]:02d}", blocks[selected[2]]),
        ]
    for name, module in targets:
        handles.append(module.register_forward_hook(hook(name)))
    return captured, handles, [name for name, _ in targets]


@torch.inference_mode()
def extract_perturbed(encoder, loader, backbone: str, device: torch.device, seed: int):
    captured, handles, stage_names = register_stage_hooks(encoder, backbone)
    stores = {name: {mode: [] for mode in PERTURBATIONS} for name in ("backbone", "projection", *stage_names)}
    labels, indices, keys, sample_names = [], [], [], []
    try:
        for batch_number, batch in enumerate(loader, start=1):
            x = batch["rgb"].permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
            batch_indices = batch["global_index"].cpu().numpy()
            for mode in PERTURBATIONS:
                captured.clear()
                altered = perturb_video(x, batch_indices, mode, seed)
                feature = encoder.forward_features(altered)
                projection = encoder.fc(feature)
                stores["backbone"][mode].append(feature.detach().float().cpu())
                stores["projection"][mode].append(projection.detach().float().cpu())
                for stage in stage_names:
                    if stage not in captured:
                        raise RuntimeError(f"Hook {stage} did not fire")
                    stores[stage][mode].append(captured[stage])
            labels.append(batch["tier_ids"]["tier1"].long().cpu())
            indices.append(batch["global_index"].long().cpu())
            keys.extend(str(value) for value in batch["key"])
            sample_names.extend(str(value) for value in batch["sample_name"])
            if batch_number % 20 == 0:
                print(f"  processed {batch_number}/{len(loader)} batches", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    arrays = {
        name: {mode: torch.cat(parts).numpy() for mode, parts in modes.items()}
        for name, modes in stores.items()
    }
    return arrays, torch.cat(labels).numpy(), torch.cat(indices).numpy(), np.asarray(keys), np.asarray(sample_names)


def prediction_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    recall = recall_score(y, prediction, labels=[0, 1], average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "take_recall": float(recall[0]),
        "put_recall": float(recall[1]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_model(spec: dict, arrays: dict, y: np.ndarray, indices: np.ndarray, keys: np.ndarray,
                  sample_names: np.ndarray, train_feature_dir: Path, seed: int):
    from sklearn.linear_model import LogisticRegression

    metric_rows, clip_rows, layer_rows = [], [], []
    for representation in ("backbone", "projection"):
        stored = np.load(train_feature_dir / spec["feature_id"] / f"{representation}_features.npz")
        train_x, train_y = stored["train"], stored["train_y"]
        if not np.array_equal(y, stored["test_y"]):
            raise RuntimeError(f"Held-out labels differ from cached features for {spec['id']}")
        cached_original = stored["test"]
        extracted_original = arrays[representation]["original"]
        original_agreement = np.sum(normalize(cached_original) * normalize(extracted_original), axis=1)
        print(
            f"  {representation} cached/extracted cosine: mean={original_agreement.mean():.9f}, "
            f"min={original_agreement.min():.9f}", flush=True
        )
        classifier = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed
        ).fit(normalize(train_x), train_y)
        predictions = {
            mode: classifier.predict(normalize(arrays[representation][mode])) for mode in PERTURBATIONS
        }
        original_prediction = predictions["original"]
        original_metric = prediction_metrics(y, original_prediction)
        original_z = normalize(arrays[representation]["original"])
        for mode in PERTURBATIONS:
            current_z = normalize(arrays[representation][mode])
            cosine = np.sum(original_z * current_z, axis=1)
            prediction = predictions[mode]
            metric = prediction_metrics(y, prediction)
            flipped = prediction != original_prediction
            take, put = y == 0, y == 1
            row = {
                "model": spec["label"],
                "model_id": spec["id"],
                "backbone": spec["backbone"],
                "initialization": spec["init"],
                "representation": representation,
                "perturbation": mode,
                "cosine_mean": float(cosine.mean()),
                "cosine_std": float(cosine.std(ddof=1)),
                "cosine_take": float(cosine[take].mean()),
                "cosine_put": float(cosine[put].mean()),
                **metric,
                "ba_drop": original_metric["balanced_accuracy"] - metric["balanced_accuracy"],
                "macro_f1_drop": original_metric["macro_f1"] - metric["macro_f1"],
                "take_recall_drop": original_metric["take_recall"] - metric["take_recall"],
                "put_recall_drop": original_metric["put_recall"] - metric["put_recall"],
                "prediction_flip_rate": float(flipped.mean()),
                "prediction_flip_take": float(flipped[take].mean()),
                "prediction_flip_put": float(flipped[put].mean()),
                "cached_extracted_original_cosine": float(original_agreement.mean()),
            }
            metric_rows.append(row)
            for i in range(len(y)):
                clip_rows.append(
                    {
                        "model": spec["label"],
                        "model_id": spec["id"],
                        "representation": representation,
                        "perturbation": mode,
                        "n_manifest_index": int(indices[i]),
                        "key": keys[i],
                        "sample_name": sample_names[i],
                        "label_id": int(y[i]),
                        "label": "take" if y[i] == 0 else "put",
                        "cosine_original_perturbed": float(cosine[i]),
                        "prediction_original": int(original_prediction[i]),
                        "prediction_perturbed": int(prediction[i]),
                        "flipped": int(flipped[i]),
                    }
                )
    for stage in (name for name in arrays if name not in {"backbone", "projection"}):
        original_z = normalize(arrays[stage]["original"])
        for mode in PERTURBATIONS:
            cosine = np.sum(original_z * normalize(arrays[stage][mode]), axis=1)
            take, put = y == 0, y == 1
            layer_rows.append(
                {
                    "model": spec["label"],
                    "model_id": spec["id"],
                    "backbone": spec["backbone"],
                    "initialization": spec["init"],
                    "stage": stage,
                    "perturbation": mode,
                    "cosine_mean": float(cosine.mean()),
                    "cosine_std": float(cosine.std(ddof=1)),
                    "cosine_take": float(cosine[take].mean()),
                    "cosine_put": float(cosine[put].mean()),
                    "cosine_distance": float(1.0 - cosine.mean()),
                }
            )
    return metric_rows, clip_rows, layer_rows


def save_model_arrays(path: Path, arrays: dict, y: np.ndarray, indices: np.ndarray) -> None:
    payload = {"labels": y, "n_manifest_index": indices}
    for representation, modes in arrays.items():
        for mode, values in modes.items():
            payload[f"{representation}__{mode}"] = values
    np.savez_compressed(path, **payload)


def heatmap(axis, matrix, row_labels, col_labels, title, colorbar_label, cmap="viridis", center=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    norm = None
    if center is not None and np.nanmin(matrix) < center < np.nanmax(matrix):
        norm = TwoSlopeNorm(vmin=float(np.nanmin(matrix)), vcenter=center, vmax=float(np.nanmax(matrix)))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axis.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            rgba = image.cmap(image.norm(matrix[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                col,
                row,
                f"{matrix[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if luminance < 0.43 else "black",
            )
    plt.colorbar(image, ax=axis, label=colorbar_label, fraction=0.046, pad=0.04)


def build_plots(output: Path, metric_rows: list[dict], layer_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
    modes = list(PERTURBATIONS[1:])
    models = list(dict.fromkeys(row["model"] for row in metric_rows))
    for field, filename, title, cmap, center in (
        ("cosine_mean", "temporal_feature_cosine.png", "Original vs perturbed feature cosine", "viridis", None),
        ("ba_drop", "temporal_ba_drop.png", "Frozen linear balanced-accuracy drop", "RdBu_r", 0.0),
        ("prediction_flip_rate", "temporal_prediction_flip.png", "Prediction flip rate vs original", "magma", None),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for axis, representation in zip(axes, ("backbone", "projection")):
            matrix = np.asarray([
                [next(row[field] for row in metric_rows if row["model"] == model and row["representation"] == representation and row["perturbation"] == mode) for mode in modes]
                for model in models
            ])
            heatmap(axis, matrix, models, modes, f"{title}: {representation}", field, cmap, center)
        fig.savefig(output / filename, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, representation, class_name, field in (
        (axes[0, 0], "backbone", "Take", "take_recall_drop"),
        (axes[0, 1], "backbone", "Put", "put_recall_drop"),
        (axes[1, 0], "projection", "Take", "take_recall_drop"),
        (axes[1, 1], "projection", "Put", "put_recall_drop"),
    ):
        matrix = np.asarray([
            [next(row[field] for row in metric_rows if row["model"] == model and row["representation"] == representation and row["perturbation"] == mode) for mode in modes]
            for model in models
        ])
        heatmap(axis, matrix, models, modes, f"{representation}: {class_name} recall drop", "recall drop", "RdBu_r", 0.0)
    fig.savefig(output / "temporal_class_recall_drop.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, model in zip(axes.flat, models):
        stages = list(dict.fromkeys(row["stage"] for row in layer_rows if row["model"] == model))
        matrix = np.asarray([
            [next(row["cosine_distance"] for row in layer_rows if row["model"] == model and row["stage"] == stage and row["perturbation"] == mode) for mode in modes]
            for stage in stages
        ])
        heatmap(axis, matrix, stages, modes, f"{model}: layer sensitivity (1 - cosine)", "cosine distance", "magma", None)
    fig.savefig(output / "temporal_layer_sensitivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.package_root = args.package_root.resolve()
    args.label_map = args.package_root / "runtime/manifests/take_put/dev_N/label_map.json"
    manifest = args.package_root / "runtime/manifests/take_put/dev_N/test.jsonl"
    src_root = args.project_root / "codex_script/rgb_mvit_motioncrop_seed1_20260721/src"
    sys.path[:0] = [str(args.project_root), str(src_root), str(args.package_root / "tools")]
    args.output.mkdir(parents=True, exist_ok=True)
    specs = (
        {"id": "r3d_rand", "label": "R3D-random", "backbone": "tv_r3d18", "init": "random", "feature_id": "pretrain_r3d_rand"},
        {"id": "r3d_k400", "label": "R3D-K400", "backbone": "tv_r3d18", "init": "kinetics400", "feature_id": "pretrain_r3d_k400"},
        {"id": "mvit_rand", "label": "MViT-random", "backbone": "mvit_v2_s", "init": "random", "feature_id": "pretrain_mvit_rand"},
        {"id": "mvit_k400", "label": "MViT-K400", "backbone": "mvit_v2_s", "init": "kinetics400", "feature_id": "pretrain_mvit_k400"},
    )
    all_metric_rows, all_clip_rows, all_layer_rows = [], [], []
    for spec in specs:
        checkpoint = args.results_root / "pretrain/take_put/dev_N/takeput_pretrain" / f"{spec['id']}_sup/checkpoint_0200.pth"
        spec = {**spec, "checkpoint": str(checkpoint.resolve())}
        print(f"\n[{spec['label']}] loading {checkpoint}", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = load_encoder(src_root, spec["backbone"], spec["init"], checkpoint, device)
        arrays, y, indices, keys, sample_names = extract_perturbed(
            encoder, build_loader(args, manifest), spec["backbone"], device, args.seed
        )
        save_model_arrays(args.output / f"{spec['id']}_temporal_features.npz", arrays, y, indices)
        metric_rows, clip_rows, layer_rows = analyze_model(
            spec, arrays, y, indices, keys, sample_names,
            args.package_root / "report_assets/umap", args.seed,
        )
        all_metric_rows.extend(metric_rows)
        all_clip_rows.extend(clip_rows)
        all_layer_rows.extend(layer_rows)
        del encoder, arrays
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(args.output / "temporal_diagnostics_summary.csv", all_metric_rows)
    write_csv(args.output / "temporal_diagnostics_per_clip.csv", all_clip_rows)
    write_csv(args.output / "temporal_layer_sensitivity.csv", all_layer_rows)
    build_plots(args.output, all_metric_rows, all_layer_rows)
    metadata = {
        "seed": args.seed,
        "n_frames": 16,
        "held_out_subject": "N",
        "linear_probe_training_subjects": ["M", "MR", "J"],
        "class_map": {"take": 0, "put": 1},
        "perturbations": list(PERTURBATIONS),
        "models": list(specs),
        "notes": {
            "linear_probe": "Fit only on chronological M/MR/J cached features; never refit on N or perturbed data.",
            "shuffle_seed": "Deterministic per N manifest index and shared across all four models.",
            "repeat_center": "Repeat temporal index T//2 (index 8, the ninth sampled frame) sixteen times.",
            "temporal_mean_repeat": "Pixelwise mean of all sixteen normalized frames, repeated sixteen times.",
        },
    }
    (args.output / "temporal_diagnostics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote diagnostics to {args.output}", flush=True)


if __name__ == "__main__":
    main()
