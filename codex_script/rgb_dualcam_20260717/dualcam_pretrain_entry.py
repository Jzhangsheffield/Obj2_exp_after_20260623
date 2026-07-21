#!/usr/bin/env python3
"""Dual-camera RGB pretraining entry with aligned temporal positive pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import (  # noqa: E402
    _replace_once,
    _value_after,
    patch_training_source,
)


VIEW_MODES = ("same143", "same152", "cross_fixed", "cross_random", "hybrid")


def normalized_aligned_indices(
    length_a: int,
    length_b: int,
    n_frames: int,
    span_min: float,
    randomize: bool = True,
) -> Tuple[List[int], List[int]]:
    """Map identical normalized clip positions onto two camera timelines."""
    if min(length_a, length_b, n_frames) <= 0:
        raise ValueError("Camera lengths and n_frames must be positive")
    if not 0.0 < span_min <= 1.0:
        raise ValueError("span_min must be in (0, 1]")
    if randomize:
        span = span_min + (1.0 - span_min) * float(torch.rand(()).item())
        start = (1.0 - span) * float(torch.rand(()).item())
    else:
        span, start = 1.0, 0.0
    positions = torch.linspace(start, start + span, steps=n_frames, dtype=torch.float64)
    idx_a = torch.round(positions * max(0, length_a - 1)).long().clamp(0, length_a - 1)
    idx_b = torch.round(positions * max(0, length_b - 1)).long().clamp(0, length_b - 1)
    return idx_a.tolist(), idx_b.tolist()


def _build_camera_transform(cfg, mean: Sequence[float], std: Sequence[float]):
    from aug.spatial_augmentation import (
        TemporallyConsistentSpatialAugmentation,
        ValidationAugmentation,
    )

    if not cfg.is_train:
        return ValidationAugmentation(size=cfg.rgb_out_hw, mean=mean, std=std)
    hflip = cfg.rgb_hflip_p if cfg.rgb_apply_spatial_aug else 0.0
    vflip = cfg.rgb_vflip_p if cfg.rgb_apply_spatial_aug else 0.0
    jitter = cfg.rgb_jitter_p if cfg.rgb_apply_spatial_aug else 0.0
    gray = cfg.rgb_gray_p if cfg.rgb_apply_spatial_aug else 0.0
    blur = cfg.rgb_blur_p if cfg.rgb_apply_spatial_aug else 0.0
    return TemporallyConsistentSpatialAugmentation(
        size=cfg.rgb_out_hw,
        crop_scale=cfg.rrc_scale,
        crop_ratio=cfg.rrc_ratio,
        flip_p=hflip,
        vflip_p=vflip,
        jitter_p=jitter,
        jitter_brightness=cfg.rgb_jitter_brightness,
        jitter_contrast=cfg.rgb_jitter_contrast,
        jitter_saturation=cfg.rgb_jitter_saturation,
        jitter_hue=cfg.rgb_jitter_hue,
        gray_p=gray,
        blur_p=blur,
        blur_kernel=cfg.rgb_blur_kernel,
        blur_sigma=cfg.rgb_blur_sigma,
        mean=mean,
        std=std,
    )


def install_dualcam_loader(
    view_mode: str,
    camera_a: str,
    camera_b: str,
    mean_b: Sequence[float],
    std_b: Sequence[float],
    span_min: float,
    hybrid_cross_probability: float,
) -> None:
    """Monkey-patch only the two-view RGB load path; single-view refresh stays camera A."""
    if view_mode not in VIEW_MODES:
        raise ValueError(f"Unsupported view mode: {view_mode}")
    if not 0.0 <= hybrid_cross_probability <= 1.0:
        raise ValueError("hybrid_cross_probability must be in [0, 1]")

    import utils_.mapstype_dataloader_with_index_mindrove_modified_varlen as dl

    cls = dl.PackedRGBDepthMindRoveMapDataset
    if not hasattr(cls, "_dualcam_original_load_rgb"):
        cls._dualcam_original_load_rgb = cls._load_rgb
    original_load_rgb = cls._dualcam_original_load_rgb

    def load_video(self, rec, camera_id: str) -> torch.Tensor:
        rel = dl._get_rgb_rel_from_record(rec, camera_id)
        if rel is None:
            raise FileNotFoundError(
                f"Missing camera {camera_id} for sample {rec.get('sample_name', 'unknown')}"
            )
        path = self.dataset_root / rel
        obj = torch.load(path, map_location="cpu")
        video = obj["frames"] if isinstance(obj, dict) else obj
        if not torch.is_tensor(video) or video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(f"Invalid RGB tensor for {camera_id}: {path} / {getattr(video, 'shape', None)}")
        return video

    def choose_pair() -> Tuple[str, str]:
        if view_mode == "same143":
            return camera_a, camera_a
        if view_mode == "same152":
            return camera_b, camera_b
        if view_mode == "cross_fixed":
            return camera_a, camera_b
        if view_mode == "cross_random":
            return (camera_a, camera_b) if float(torch.rand(()).item()) < 0.5 else (camera_b, camera_a)
        if float(torch.rand(()).item()) < hybrid_cross_probability:
            return (camera_a, camera_b) if float(torch.rand(()).item()) < 0.5 else (camera_b, camera_a)
        same = camera_a if float(torch.rand(()).item()) < 0.5 else camera_b
        return same, same

    def dualcam_load_rgb(self, rec):
        if not self.cfg.rgb_two_views:
            return original_load_rgb(self, rec)
        if self.cfg.rgb_transform is None:
            raise RuntimeError("cfg.rgb_transform is None")
        if not hasattr(self, "_dualcam_transform_b"):
            self._dualcam_transform_b = _build_camera_transform(self.cfg, mean_b, std_b)

        cam1, cam2 = choose_pair()
        video1 = load_video(self, rec, cam1)
        video2 = video1 if cam1 == cam2 else load_video(self, rec, cam2)
        idx1, idx2 = normalized_aligned_indices(
            int(video1.shape[0]), int(video2.shape[0]), self.cfg.n_frames,
            span_min=span_min, randomize=bool(self.cfg.is_train),
        )
        transform1 = self.cfg.rgb_transform if cam1 == camera_a else self._dualcam_transform_b
        transform2 = self.cfg.rgb_transform if cam2 == camera_a else self._dualcam_transform_b
        view1 = torch.as_tensor(transform1(video1[idx1])).contiguous()
        view2 = torch.as_tensor(transform2(video2[idx2])).contiguous()
        return view1, view2

    cls._load_rgb = dualcam_load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dualcam-original-script", required=True)
    parser.add_argument("--dualcam-view-mode", choices=VIEW_MODES, required=True)
    parser.add_argument("--dualcam-camera-a", default="00143")
    parser.add_argument("--dualcam-camera-b", default="00152")
    parser.add_argument("--dualcam-mean-b", nargs=3, type=float, required=True)
    parser.add_argument("--dualcam-std-b", nargs=3, type=float, required=True)
    parser.add_argument("--dualcam-temporal-span-min", type=float, default=0.85)
    parser.add_argument("--dualcam-hybrid-cross-probability", type=float, default=0.5)
    parser.add_argument("--dualcam-aux-ce-weight", type=float, default=0.0)
    parser.add_argument("--dualcam-auto-resume", action="store_true")
    parser.add_argument("--dualcam-validate-only", action="store_true")
    parser.add_argument("--dualcam-parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()

    original = Path(custom.dualcam_original_script).expanduser().resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    root = original.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    source = patch_training_source(original.read_text(encoding="utf-8"))
    compile(source, str(original), "exec")
    if custom.dualcam_validate_only:
        print("Dual-camera pretrain source validation: OK")
        return
    if custom.dualcam_parse_only:
        source = _replace_once(
            source,
            'if __name__ == "__main__":\n    worker(args)',
            'if __name__ == "__main__":\n    print("Dual-camera pretrain command parse: OK")',
            "dualcam-parse-only",
        )

    save_path = _value_after(remaining, "--weight_save_path")
    epochs = int(_value_after(remaining, "--epochs", "200"))
    resume_checkpoint = ""
    if save_path and not custom.dualcam_parse_only:
        output = Path(save_path)
        output.mkdir(parents=True, exist_ok=True)
        final = output / f"checkpoint_{epochs:04d}.pth"
        if final.is_file():
            print(f"[DualCam Skip] completed checkpoint exists: {final}")
            return
        if custom.dualcam_auto_resume:
            checkpoints = sorted(output.glob("checkpoint_*.pth"))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
        record = {
            "view_mode": custom.dualcam_view_mode,
            "camera_a": custom.dualcam_camera_a,
            "camera_b": custom.dualcam_camera_b,
            "mean_b": custom.dualcam_mean_b,
            "std_b": custom.dualcam_std_b,
            "temporal_span_min": custom.dualcam_temporal_span_min,
            "hybrid_cross_probability": custom.dualcam_hybrid_cross_probability,
            "aux_ce_weight": custom.dualcam_aux_ce_weight,
            "resume_checkpoint": resume_checkpoint or None,
            "alignment": "shared normalized clip positions; no timestamps available",
        }
        (output / "dualcam_wrapper_args.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    install_dualcam_loader(
        view_mode=custom.dualcam_view_mode,
        camera_a=custom.dualcam_camera_a,
        camera_b=custom.dualcam_camera_b,
        mean_b=custom.dualcam_mean_b,
        std_b=custom.dualcam_std_b,
        span_min=custom.dualcam_temporal_span_min,
        hybrid_cross_probability=custom.dualcam_hybrid_cross_probability,
    )
    sys.argv = [str(original), *remaining]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(original),
        "ROUND2_AUX_CE_WEIGHT": float(custom.dualcam_aux_ce_weight),
        "ROUND2_RESUME_CHECKPOINT": resume_checkpoint,
    }
    exec(compile(source, str(original), "exec"), globals_dict, globals_dict)


if __name__ == "__main__":
    main()
