#!/usr/bin/env python3
"""Extract features and assess train-to-held-out-subject geometry.

Unlike the earlier joint embedding utility, PCA/UMAP are fitted on training
features only and the held-out subject is transformed without refitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def effective_rank(x: np.ndarray) -> tuple[float, float]:
    x = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    variance = singular * singular
    probability = variance / np.clip(variance.sum(), 1e-12, None)
    positive = probability[probability > 0]
    return float(np.exp(-(positive * np.log(positive)).sum())), float(probability[:5].sum())


def geometry(x: np.ndarray, y: np.ndarray, persons: np.ndarray) -> dict:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    z = normalize(x)
    classes = np.unique(y)
    centroids = np.stack([normalize(z[y == cls].mean(0, keepdims=True))[0] for cls in classes])
    within = float(np.mean([np.mean(1 - z[y == cls] @ centroids[i]) for i, cls in enumerate(classes)]))
    pairwise = 1 - centroids @ centroids.T
    between = float(pairwise[np.triu_indices_from(pairwise, k=1)].mean())
    rank, top5 = effective_rank(z)
    result = {
        "silhouette_cosine": float(silhouette_score(z, y, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(z, y)),
        "effective_rank": rank,
        "top5_variance_fraction": top5,
        "within_cosine_distance": within,
        "between_centroid_cosine_distance": between,
        "fisher_like_between_within_ratio": between / max(within, 1e-12),
    }
    if len(np.unique(persons)) > 1:
        result["subject_silhouette_cosine"] = float(silhouette_score(z, persons, metric="cosine"))
    return result


def predictive(train_x, train_y, test_x, test_y, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    train_z, test_z = normalize(train_x), normalize(test_x)
    classifier = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed).fit(train_z, train_y)
    prediction = classifier.predict(test_z)
    nearest = train_y[(test_z @ train_z.T).argmax(1)]
    labels = np.unique(np.concatenate((train_y, test_y)))
    return {
        "linear_balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "linear_macro_f1": float(f1_score(test_y, prediction, average="macro", zero_division=0)),
        "knn1_balanced_accuracy": float(balanced_accuracy_score(test_y, nearest)),
        "linear_confusion_matrix": confusion_matrix(test_y, prediction, labels=labels).tolist(),
    }


def read_persons(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        return np.asarray([json.loads(line)["person"] for line in handle if line.strip()])


def build_loader(args, manifest: Path, label_map: dict):
    from utils_.mapstype_dataloader_with_index import (
        PackedMultiModalConfig,
        build_packed_mapstyle_dataset,
        build_packed_mapstyle_loader_from_dataset,
    )

    cfg = PackedMultiModalConfig(
        n_frames=args.n_frames,
        rgb_two_views=False,
        rgb_camera_id=args.rgb_camera_id,
        use_modalities=("rgb",),
        load_labels=True,
        label_map_path=str(args.label_map),
        tier_mode="tier1",
        is_train=False,
        rgb_out_hw=(args.image_size, args.image_size),
        rgb_mean=tuple(args.rgb_mean),
        rgb_std=tuple(args.rgb_std),
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


def state_dict(checkpoint: Path) -> dict:
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in obj.items()}


def load_encoder(args, device):
    from backbone.video_backbone import generate_video_model

    if args.checkpoint_kind == "classifier":
        model = generate_video_model(args.backbone, num_classes=args.num_classes, model_depth=18)
        source = state_dict(args.checkpoint)
        filtered = {key: value for key, value in source.items() if key in model.state_dict() and model.state_dict()[key].shape == value.shape}
        message = model.load_state_dict(filtered, strict=False)
        if len(filtered) == 0:
            raise RuntimeError("No classifier checkpoint tensors matched the model")
        print(f"[classifier load] matched={len(filtered)}, missing={len(message.missing_keys)}")
        return model.to(device).eval()
    from backbone.MoCo_VAR_supcon_wds import MoCo3D

    model = MoCo3D(
        partial(generate_video_model, backbone_name=args.backbone, model_depth=18),
        dim=args.proj_dim,
        K=args.queue_size,
        T=0.07,
        mlp=True,
        exclude_invalid_queue=True,
    )
    source = state_dict(args.checkpoint)
    message = model.load_state_dict(source, strict=False)
    unexpected = [key for key in message.unexpected_keys if not key.startswith("round2_aux_classifier")]
    if unexpected:
        raise RuntimeError(f"Unexpected pretrain keys: {unexpected[:20]}")
    return model.encoder_q.to(device).eval()


@torch.inference_mode()
def extract(encoder, loader, device):
    features, projections, labels = [], [], []
    for batch in loader:
        rgb = batch["rgb"].permute(0, 2, 1, 3, 4).contiguous().to(device)
        feature = encoder.forward_features(rgb)
        projection = encoder.fc(feature)
        features.append(feature.float().cpu())
        projections.append(projection.float().cpu())
        labels.append(batch["tier_ids"]["tier1"].long().cpu())
    return torch.cat(features).numpy(), torch.cat(projections).numpy(), torch.cat(labels).numpy()


def plot_embedding(path: Path, train_xy, test_xy, train_y, test_y, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    classes = np.unique(np.concatenate((train_y, test_y)))
    colors = plt.get_cmap("tab20", len(classes))
    for index, cls in enumerate(classes):
        train_mask, test_mask = train_y == cls, test_y == cls
        axis.scatter(train_xy[train_mask, 0], train_xy[train_mask, 1], s=10, alpha=0.35, color=colors(index))
        axis.scatter(test_xy[test_mask, 0], test_xy[test_mask, 1], s=24, alpha=0.8, marker="x", color=colors(index), label=str(cls))
    axis.set_title(title + " (train=dot, held-out=x)")
    axis.legend(title="class id", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_embedding(output: Path, name: str, train_x, test_x, train_y, test_y, seed: int) -> dict:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed).fit(train_x)
    train_pca, test_pca = pca.transform(train_x), pca.transform(test_x)
    np.savez(output / f"{name}_pca_trainfit.npz", train=train_pca, test=test_pca, train_y=train_y, test_y=test_y)
    plot_embedding(output / f"{name}_pca_trainfit.png", train_pca, test_pca, train_y, test_y, f"{name} PCA")
    note = {"pca_explained_variance": pca.explained_variance_ratio_.tolist()}
    try:
        import umap

        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit(train_x)
        train_umap, test_umap = reducer.transform(train_x), reducer.transform(test_x)
        np.savez(output / f"{name}_umap_trainfit.npz", train=train_umap, test=test_umap, train_y=train_y, test_y=test_y)
        plot_embedding(output / f"{name}_umap_trainfit.png", train_umap, test_umap, train_y, test_y, f"{name} UMAP")
    except Exception as exc:
        note["umap_note"] = f"UMAP unavailable: {type(exc).__name__}: {exc}"
    return note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=["pretrain", "classifier"], required=True)
    parser.add_argument("--backbone", choices=["tv_r3d18", "mvit_v2_s"], required=True)
    parser.add_argument("--backbone-init", choices=["random", "kinetics400"], default="random")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--rgb-camera-id", default="00143")
    parser.add_argument("--rgb-mean", nargs=3, type=float, default=[0.45, 0.45, 0.45])
    parser.add_argument("--rgb-std", nargs=3, type=float, default=[0.225, 0.225, 0.225])
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--queue-size", type=int, default=1088)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.src_root.parents[2]
    sys.path[:0] = [str(project), str(args.src_root.resolve())]
    from codex_script.rgb_supcon_repair_20260806.common.runtime_patch import install

    install(args.src_root, "rgb", "current", args.backbone_init, False)
    from utils_.mapstype_dataloader_with_index import load_label_map_json

    label_map = load_label_map_json(str(args.label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args, device)
    train = extract(encoder, build_loader(args, args.train_manifest, label_map), device)
    test = extract(encoder, build_loader(args, args.test_manifest, label_map), device)
    train_persons, test_persons = read_persons(args.train_manifest), read_persons(args.test_manifest)
    if len(train_persons) != len(train[2]) or len(test_persons) != len(test[2]):
        raise RuntimeError("Manifest order and extracted feature count differ")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "backbone": args.backbone,
        "representations": {},
    }
    for index, name in enumerate(("backbone", "projection")):
        train_x, test_x = train[index], test[index]
        result["representations"][name] = {
            "train_geometry": geometry(train_x, train[2], train_persons),
            "test_geometry": geometry(test_x, test[2], test_persons),
            "held_out_predictive": predictive(train_x, train[2], test_x, test[2], args.seed),
            **save_embedding(args.output, name, train_x, test_x, train[2], test[2], args.seed),
        }
        np.savez_compressed(
            args.output / f"{name}_features.npz",
            train=train_x,
            test=test_x,
            train_y=train[2],
            test_y=test[2],
            train_person=train_persons,
            test_person=test_persons,
        )
    (args.output / "feature_quality.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
