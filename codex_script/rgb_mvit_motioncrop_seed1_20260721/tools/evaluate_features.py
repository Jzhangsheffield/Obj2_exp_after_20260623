#!/usr/bin/env python3
"""Extract frozen SupLoss features and quantify class separation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--tier-mode", default="tier1")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--n-frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rgb-camera-id", required=True)
    p.add_argument("--rgb-mean", nargs=3, type=float, required=True)
    p.add_argument("--rgb-std", nargs=3, type=float, required=True)
    p.add_argument("--rgb-preserve-aspect-pad", action="store_true")
    p.add_argument("--backbone", choices=["resnet3d18", "mvit_v2_s"], required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--queue-size", type=int, default=1088)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def build_loader(args, manifest, label_map):
    from utils_.mapstype_dataloader_with_index import (
        PackedMultiModalConfig, build_packed_mapstyle_dataset,
        build_packed_mapstyle_loader_from_dataset,
    )
    cfg = PackedMultiModalConfig(
        n_frames=args.n_frames, rgb_two_views=False,
        rgb_camera_id=args.rgb_camera_id,
        rgb_preserve_aspect_pad=args.rgb_preserve_aspect_pad,
        use_modalities=("rgb",), load_labels=True,
        label_map_path=str(args.dataset_root / args.label_map),
        tier_mode=args.tier_mode, is_train=False,
        rgb_out_hw=(args.image_size, args.image_size),
        rgb_mean=tuple(args.rgb_mean), rgb_std=tuple(args.rgb_std),
    )
    ds = build_packed_mapstyle_dataset(
        args.dataset_root, manifest, cfg, label_map=label_map, verify_paths_on_init=True
    )
    return build_packed_mapstyle_loader_from_dataset(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False, prefetch_factor=2, pin_memory=True,
    )


def load_encoder(args, device):
    from backbone.MoCo_VAR_supcon_wds import MoCo3D
    from backbone.video_backbone import generate_video_model
    model = MoCo3D(
        partial(generate_video_model, backbone_name=args.backbone, model_depth=18),
        dim=args.proj_dim, K=args.queue_size, T=args.temperature,
        mlp=True, exclude_invalid_queue=True,
    )
    obj = torch.load(args.checkpoint, map_location="cpu")
    state = obj.get("state_dict", obj)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    missing = [k for k in msg.missing_keys if not k.startswith("queue")]
    if missing or msg.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing[:20]}, unexpected={msg.unexpected_keys[:20]}")
    return model.encoder_q.to(device).eval()


@torch.inference_mode()
def extract(encoder, loader, tier_mode, device):
    backbone_batches, projection_batches, label_batches, keys = [], [], [], []
    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = encoder.fc.register_forward_pre_hook(hook)
    try:
        for batch in loader:
            x = batch["rgb"].permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
            captured.clear()
            projection = encoder(x)
            if len(captured) != 1:
                raise RuntimeError("Expected exactly one classifier pre-hook call")
            backbone_batches.append(captured[0].float().cpu())
            projection_batches.append(projection.float().cpu())
            label_batches.append(batch["tier_ids"][tier_mode].long().cpu())
            keys.extend(str(x) for x in batch["key"])
    finally:
        handle.remove()
    return (
        torch.cat(backbone_batches).numpy(), torch.cat(projection_batches).numpy(),
        torch.cat(label_batches).numpy(), keys,
    )


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def metrics_for_feature(train_x, train_y, test_x, test_y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, silhouette_score

    train_n, test_n = normalize(train_x), normalize(test_x)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    clf.fit(train_n, train_y)
    linear_pred = clf.predict(test_n)

    similarities = test_n @ train_n.T
    knn = {}
    for k in (1, 5, 10):
        idx = np.argpartition(-similarities, kth=min(k, similarities.shape[1]) - 1, axis=1)[:, :k]
        pred = []
        for row in idx:
            counts = np.bincount(train_y[row], minlength=int(train_y.max()) + 1)
            pred.append(int(counts.argmax()))
        pred = np.asarray(pred)
        knn[f"knn_k{k}_balanced_acc"] = float(balanced_accuracy_score(test_y, pred))

    classes = np.unique(test_y)
    centroids = np.stack([normalize(test_n[test_y == c].mean(axis=0, keepdims=True))[0] for c in classes])
    centroid_pred = classes[(test_n @ centroids.T).argmax(axis=1)]
    within = np.mean([np.mean(1.0 - test_n[test_y == c] @ centroids[i]) for i, c in enumerate(classes)])
    pairwise = 1.0 - centroids @ centroids.T
    between = pairwise[np.triu_indices_from(pairwise, k=1)].mean()
    return {
        "linear_accuracy": float(accuracy_score(test_y, linear_pred)),
        "linear_balanced_accuracy": float(balanced_accuracy_score(test_y, linear_pred)),
        "linear_macro_f1": float(f1_score(test_y, linear_pred, average="macro", zero_division=0)),
        **knn,
        "true_label_silhouette_cosine": float(silhouette_score(test_n, test_y, metric="cosine")),
        "nearest_centroid_balanced_accuracy": float(balanced_accuracy_score(test_y, centroid_pred)),
        "mean_within_class_cosine_distance": float(within),
        "mean_between_centroid_cosine_distance": float(between),
        "between_within_ratio": float(between / max(within, 1e-12)),
    }


def save_projection(x, labels, keys, out_dir, name, seed):
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = PCA(n_components=2, random_state=seed).fit_transform(normalize(x))
    csv_path = out_dir / f"{name}_pca.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "pca1", "pca2"])
        w.writerows((k, int(y), float(a), float(b)) for k, y, (a, b) in zip(keys, labels, coords))
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=16, cmap="tab20", alpha=0.8)
    ax.set(title=f"{name} PCA by true class", xlabel="PC1", ylabel="PC2")
    fig.colorbar(scatter, ax=ax, label="class id")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_pca.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.src_root.resolve()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from utils_.mapstype_dataloader_with_index import load_label_map_json

    label_map = load_label_map_json(args.dataset_root / args.label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(args, device)
    train = extract(encoder, build_loader(args, args.train_manifest, label_map), args.tier_mode, device)
    test = extract(encoder, build_loader(args, args.test_manifest, label_map), args.tier_mode, device)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, train_x, test_x in (
        ("backbone", train[0], test[0]), ("projection", train[1], test[1])
    ):
        np.save(args.output / f"train_{name}.npy", train_x)
        np.save(args.output / f"test_{name}.npy", test_x)
        results[name] = metrics_for_feature(train_x, train[2], test_x, test[2], args.seed)
        save_projection(test_x, test[2], test[3], args.output, name, args.seed)
    np.save(args.output / "train_labels.npy", train[2])
    np.save(args.output / "test_labels.npy", test[2])
    payload = {"checkpoint": str(args.checkpoint.resolve()), "backbone": args.backbone, "metrics": results}
    (args.output / "feature_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
