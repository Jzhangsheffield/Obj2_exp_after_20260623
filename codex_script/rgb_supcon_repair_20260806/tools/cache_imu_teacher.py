#!/usr/bin/env python3
"""Cache frozen 512-D IMU SupLoss features in exact RGB-train manifest order."""
from __future__ import annotations
import argparse, ast, json, sys
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F

def literal(x):
    if x is None or isinstance(x, (list, tuple)): return x
    try: return ast.literal_eval(x)
    except Exception: return x

def main():
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, required=True); p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--args-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args(); sys.path.insert(0, str(args.project_root)); cfg = json.loads(args.args_json.read_text(encoding="utf-8"))
    from backbone.MoCo_VAR_supcon_wds_1D import MoCo1D
    from backbone.renet1d_my import build_resnet1d
    from utils_.mapstype_dataloader_with_index_mindrove_modified_varlen import PackedMultiModalConfig, build_packed_mapstyle_dataset, build_packed_mapstyle_loader_from_dataset, load_label_map_json
    stats = {}
    for name in ("mindrove_left_imu_mean", "mindrove_left_imu_std", "mindrove_right_imu_mean", "mindrove_right_imu_std"):
        value = literal(cfg.get(name)); stats[name] = None if value is None else tuple(float(v) for v in value)
    label_map_path = args.dataset_root / "label_map_except_take_put.json"
    ds_cfg = PackedMultiModalConfig(n_frames=16, rgb_two_views=False, use_modalities=("mindrove",), missing_policy="skip", load_labels=True,
        label_map_path=str(label_map_path), tier_mode="tier1", is_train=False, mindrove_two_views=False,
        mindrove_target_len=int(cfg.get("mindrove_target_len", 128)), mindrove_imu_target_len=int(cfg.get("mindrove_imu_target_len", 128)),
        mindrove_hands=tuple(cfg.get("mindrove_hands", ["left", "right"])), mindrove_signals=("imu",), mindrove_merge_hands=False,
        mindrove_apply_augmentation=False, mindrove_apply_normalization=True, **stats)
    label_map = load_label_map_json(str(label_map_path))
    ds = build_packed_mapstyle_dataset(str(args.dataset_root), str(args.manifest), ds_cfg, label_map=label_map, verify_paths_on_init=True)
    expected=[json.loads(line)["sample_name"] for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    actual=[str(row["sample_name"]) for row in ds.records]
    if actual != expected:
        raise RuntimeError(f"IMU teacher cache must preserve exact manifest order; expected {len(expected)} rows, loader kept {len(actual)}")
    loader = build_packed_mapstyle_loader_from_dataset(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, pin_memory=True, prefetch_factor=2 if args.num_workers else None)
    base = partial(build_resnet1d, arch=str(cfg.get("ts_arch", "resnet10_1d")), in_channels=12, base_channels=int(cfg.get("ts_base_channels", 64)), stem_kernel_size=int(cfg.get("ts_stem_kernel_size", 7)), stem_stride=int(cfg.get("ts_stem_stride", 2)), use_stem_pool=bool(cfg.get("ts_use_stem_pool", True)), zero_init_residual=False)
    model = MoCo1D(base_encoder=base, dim=int(cfg.get("proj_dim", 128)), K=int(cfg.get("K_queue", 1088)), m=float(cfg.get("momentum", .999)), T=float(cfg.get("temperature", .07)), mlp=True, enable_kcl_loss=False, num_positive=6, exclude_invalid_queue=False)
    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False); state = obj["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}; model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval(); feats=[]; labels=[]; keys=[]
    with torch.inference_mode():
        for batch in loader:
            parts = [batch["mindrove"][f"{hand}_imu"] for hand in ("left", "right")]
            target = max(int(x.shape[-1]) for x in parts); parts = [F.interpolate(x.float(), size=target, mode="linear", align_corners=False) if x.shape[-1] != target else x.float() for x in parts]
            x = torch.cat(parts, dim=1).to(device); feats.append(model.encoder_q.forward_features(x).float().cpu()); labels.append(batch["tier_ids"]["tier1"].long().cpu()); keys.extend(str(x) for x in batch["key"])
    features = torch.cat(feats); y = torch.cat(labels); prototypes = torch.stack([features[y == c].mean(0) for c in range(15)])
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save({"features": features, "labels": y, "keys": keys, "class_prototypes": prototypes, "manifest": str(args.manifest)}, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(features), "feature_dim": features.shape[1], "classes": sorted(y.unique().tolist())}, indent=2))

if __name__ == "__main__": main()
