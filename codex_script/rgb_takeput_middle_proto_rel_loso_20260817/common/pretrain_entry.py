#!/usr/bin/env python3
"""Process-local adapter for the old ProtoLoss/RelLoss trainer.

This entry supports torchvision R3D-18 and MViT-v2-S with random or K400
initialization.  It never edits the source trainer and rejects the V2 package.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Patch anchor {label!r} expected once, found {count}")
    return source.replace(old, new, 1)


def value_after(argv: list[str], flag: str, default: str | None = None) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def patch_lr(source: str) -> str:
    old = '''    lr = args.learning_rate
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
'''
    new = '''    base_lr = args.learning_rate
    warmup = int(EXP_LR_WARMUP_EPOCHS)
    if warmup > 0 and epoch < warmup:
        lr = base_lr * float(epoch + 1) / float(warmup)
    elif args.cos:
        progress = float(epoch - warmup) / float(max(1, args.epochs - warmup - 1))
        progress = min(1.0, max(0.0, progress))
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        lr = base_lr
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
'''
    return replace_once(source, old, new, "warmup learning rate")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--package-source", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--exp-backbone", choices=["tv_r3d18", "mvit_v2_s"], required=True)
    parser.add_argument("--exp-backbone-init", choices=["random", "kinetics400"], required=True)
    parser.add_argument("--proto-positive-mode", choices=["single", "all", "soft"], default="all")
    parser.add_argument("--lr-warmup-epochs", type=int, default=10)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()

    source_path = Path(custom.package_source).resolve()
    project = Path(custom.project_root).resolve()
    package_root = Path(__file__).resolve().parents[1]
    for path in (project, package_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    forbidden = (project / "codex_script" / "rgb_proto_rel_v2_20260804").resolve()
    if source_path == forbidden or forbidden in source_path.parents:
        raise RuntimeError("V2 source is forbidden in this experiment package")

    from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import (
        make_temporal_sampler,
        patch_training_source,
    )

    source = patch_training_source(source_path.read_text(encoding="utf-8"))
    source = patch_lr(source)
    source = replace_once(
        source,
        "loss_proto = prototype_contrastive_loss_all_positive(",
        "loss_proto = OLD_PROTO_LOSS_FN(",
        "old prototype positive mode",
    )
    if custom.parse_only:
        source = replace_once(
            source,
            'if __name__ == "__main__":\n    worker(args)',
            'if __name__ == "__main__":\n    print("old Proto/Rel command parse: OK")',
            "parse only",
        )

    output_text = value_after(remaining, "--weight_save_path")
    epochs = int(value_after(remaining, "--epochs", "200"))
    resume = ""
    if output_text and not custom.parse_only:
        output = Path(output_text)
        output.mkdir(parents=True, exist_ok=True)
        if (output / f"checkpoint_{epochs:04d}.pth").is_file():
            print(f"[Skip] completed: {output}")
            return
        candidates = sorted(output.glob("checkpoint_*.pth")) if custom.auto_resume else []
        resume = str(candidates[-1]) if candidates else ""
        (output / "old_loss_wrapper_args.json").write_text(
            json.dumps(vars(custom) | {"resume_checkpoint": resume or None}, indent=2),
            encoding="utf-8",
        )

    from codex_script.rgb_supcon_repair_20260806.common.runtime_patch import _make_model_factory
    import aug.temporal_augmentation_adaptive as temporal_module
    import backbone.resnet as resnet_module

    temporal_module.sample_two_views_indices = make_temporal_sampler("shared", 1.0)
    factory = _make_model_factory(resnet_module, "current", custom.exp_backbone_init, False)
    resnet_module.generate_model = lambda model_depth, num_classes: factory(
        custom.exp_backbone, num_classes, model_depth, False
    )

    from common.proto_loss_modes import get_proto_loss

    sys.argv = [str(source_path), *remaining]
    namespace = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "ROUND2_AUX_CE_WEIGHT": 0.0,
        "ROUND2_RESUME_CHECKPOINT": resume,
        "EXP_LR_WARMUP_EPOCHS": int(custom.lr_warmup_epochs),
        "OLD_PROTO_LOSS_FN": get_proto_loss(custom.proto_positive_mode),
    }
    exec(compile(source, str(source_path), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
