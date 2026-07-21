#!/usr/bin/env python3
"""In-memory training entry adding temporal, resume, and proto-mode controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--required-original-script", required=True)
    parser.add_argument("--required-temporal-mode", choices=["independent", "shared", "overlap"], default="shared")
    parser.add_argument("--required-min-temporal-overlap", type=float, default=1.0)
    parser.add_argument("--required-proto-positive-mode", choices=["single", "all", "soft"], default="all")
    parser.add_argument("--required-backbone-temporal-mode", choices=["current", "t3_lfb"], default="current")
    parser.add_argument("--required-auto-resume", action="store_true")
    parser.add_argument("--required-parse-only", action="store_true")
    parser.add_argument("--required-validate-only", action="store_true")
    custom, remaining = parser.parse_known_args()

    original = Path(custom.required_original_script).expanduser().resolve()
    project = original.parents[1]
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import (
        _replace_once,
        _value_after,
        make_temporal_sampler,
        patch_training_source,
    )
    source = patch_training_source(original.read_text(encoding="utf-8"))
    source = _replace_once(
        source,
        "loss_proto = prototype_contrastive_loss_all_positive(",
        "loss_proto = REQUIRED_PROTO_LOSS_FN(",
        "required-proto-positive-mode",
    )
    source = _replace_once(
        source,
        '                    "round2_aux_ce_weight": ROUND2_AUX_CE_WEIGHT,\n',
        '                    "round2_aux_ce_weight": ROUND2_AUX_CE_WEIGHT,\n'
        '                    "required_backbone_temporal_mode": REQUIRED_BACKBONE_TEMPORAL_MODE,\n',
        "checkpoint-temporal-mode",
    )
    if custom.required_validate_only:
        compile(source, str(original), "exec")
        print("Required pretrain source patch validation: OK")
        return
    from common.proto_loss_modes import get_proto_loss
    if custom.required_parse_only:
        source = _replace_once(
            source,
            'if __name__ == "__main__":\n    worker(args)',
            'if __name__ == "__main__":\n    print("Required pretrain command parse: OK")',
            "parse-only",
        )

    output_text = _value_after(remaining, "--weight_save_path")
    epochs = int(_value_after(remaining, "--epochs", "200"))
    resume = ""
    if output_text and not custom.required_parse_only:
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        final_path = output / ("checkpoint_%04d.pth" % epochs)
        if final_path.is_file():
            print("[Skip] completed pretrain exists: %s" % final_path)
            return
        if custom.required_auto_resume:
            candidates = sorted(output.glob("checkpoint_*.pth"))
            if candidates:
                resume = str(candidates[-1])
        (output / "required_wrapper_args.json").write_text(
            json.dumps(
                {
                    "temporal_mode": custom.required_temporal_mode,
                    "min_temporal_overlap": custom.required_min_temporal_overlap,
                    "proto_positive_mode": custom.required_proto_positive_mode,
                    "backbone_temporal_mode": custom.required_backbone_temporal_mode,
                    "resume_checkpoint": resume or None,
                }, indent=2,
            ), encoding="utf-8",
        )

    import aug.temporal_augmentation_adaptive as temporal
    import backbone.resnet as resnet3d
    from common.temporal_backbone import install_generate_model_mode
    temporal.sample_two_views_indices = make_temporal_sampler(
        custom.required_temporal_mode, custom.required_min_temporal_overlap
    )
    install_generate_model_mode(resnet3d, custom.required_backbone_temporal_mode)
    sys.argv = [str(original)] + remaining
    namespace = {
        "__name__": "__main__",
        "__file__": str(original),
        "ROUND2_AUX_CE_WEIGHT": 0.0,
        "ROUND2_RESUME_CHECKPOINT": resume,
        "REQUIRED_PROTO_LOSS_FN": get_proto_loss(custom.required_proto_positive_mode),
        "REQUIRED_BACKBONE_TEMPORAL_MODE": custom.required_backbone_temporal_mode,
    }
    exec(compile(source, str(original), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
