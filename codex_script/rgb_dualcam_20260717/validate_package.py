#!/usr/bin/env python3
"""Offline checks for the dual-camera configuration and runtime source patches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualcam_pretrain_entry import install_dualcam_loader, normalized_aligned_indices
from inspect_manifest import inspect_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--loader-smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((HERE / "config" / "dualcam_config.json").read_text(encoding="utf-8"))
    assert [int(x["index"]) for x in cfg["experiments"]] == list(range(7))
    assert [int(x["index"]) for x in cfg["finetune_plan"]] == list(range(9))
    assert set(cfg["cameras"]) == {"00143", "00152"}
    for lengths in ((31, 29), (20, 19), (12, 16), (100, 96)):
        a, b = normalized_aligned_indices(*lengths, n_frames=16, span_min=0.85, randomize=False)
        assert len(a) == len(b) == 16
        assert min(a) >= 0 and max(a) < lengths[0]
        assert min(b) >= 0 and max(b) < lengths[1]
        norm_a = [x / max(1, lengths[0] - 1) for x in a]
        norm_b = [x / max(1, lengths[1] - 1) for x in b]
        assert max(abs(x - y) for x, y in zip(norm_a, norm_b)) <= 0.06

    original_pretrain = ROOT / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    original_ft = ROOT / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    subprocess.run([
        sys.executable, str(HERE / "dualcam_pretrain_entry.py"),
        "--dualcam-original-script", str(original_pretrain),
        "--dualcam-view-mode", "cross_random",
        "--dualcam-mean-b", "0.413208", "0.425694", "0.427126",
        "--dualcam-std-b", "0.271570", "0.253219", "0.249541",
        "--dualcam-validate-only",
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "dualcam_finetune_entry.py"),
        "--dualcam-original-script", str(original_ft), "--dualcam-validate-only",
    ], check=True)

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root and dataset_root.is_dir():
        reports = []
        for rel in (
            cfg["task"]["train_manifest_rel"],
            cfg["task"]["val_manifest_rel"],
            cfg["task"]["test_manifest_rel"],
        ):
            report = inspect_manifest(dataset_root, dataset_root / rel, args.check_files)
            reports.append(report)
            assert report["missing_cam00143"] == report["missing_cam00152"] == 0
            assert (report["missing_tensor_files"] or 0) == 0
        print(json.dumps(reports, indent=2))
        if args.loader_smoke:
            import utils_.mapstype_dataloader_with_index_mindrove_modified_varlen as dl
            label_map = dl.load_label_map_json(dataset_root / cfg["task"]["label_map_rel"])
            aug = cfg["augmentation"]
            cam_a, cam_b = cfg["cameras"]["00143"], cfg["cameras"]["00152"]
            for mode in ("same143", "same152", "cross_fixed", "cross_random", "hybrid"):
                install_dualcam_loader(
                    mode, "00143", "00152", cam_b["mean"], cam_b["std"],
                    aug["temporal_span_min"], 0.5,
                )
                ds_cfg = dl.PackedMultiModalConfig(
                    n_frames=cfg["task"]["n_frames"], rgb_two_views=True,
                    rgb_camera_id="00143", use_modalities=("rgb",), missing_policy="skip",
                    load_labels=True, tier_mode=cfg["task"]["tier_mode"], is_train=True,
                    label_map_path=str(dataset_root / cfg["task"]["label_map_rel"]),
                    rgb_mean=tuple(cam_a["mean"]), rgb_std=tuple(cam_a["std"]),
                    rgb_out_hw=(224, 224), rrc_scale=tuple(aug["rrc_scale"]),
                    rrc_ratio=tuple(aug["rrc_ratio"]), rgb_hflip_p=aug["hflip_p"],
                    rgb_vflip_p=aug["vflip_p"], rgb_jitter_p=aug["jitter_p"],
                    rgb_jitter_brightness=aug["jitter_strength"][0],
                    rgb_jitter_contrast=aug["jitter_strength"][1],
                    rgb_jitter_saturation=aug["jitter_strength"][2],
                    rgb_jitter_hue=aug["jitter_strength"][3], rgb_gray_p=aug["gray_p"],
                    rgb_blur_p=aug["blur_p"], rgb_blur_kernel=5,
                )
                dataset = dl.build_packed_mapstyle_dataset(
                    dataset_root, cfg["task"]["train_manifest_rel"], ds_cfg,
                    label_map=label_map, verify_paths_on_init=False,
                )
                view1, view2 = dataset[0]["rgb"]
                assert tuple(view1.shape) == tuple(view2.shape) == (16, 3, 224, 224)
                assert view1.isfinite().all() and view2.isfinite().all()
                print(f"loader {mode}: {tuple(view1.shape)} / {tuple(view2.shape)} OK")
    print("Dual-camera package validation: OK")


if __name__ == "__main__":
    main()
