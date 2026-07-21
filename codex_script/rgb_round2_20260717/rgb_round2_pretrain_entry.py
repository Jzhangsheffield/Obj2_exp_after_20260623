#!/usr/bin/env python3
"""Round-2 RGB pretraining entry.

This wrapper leaves the project training script untouched.  It installs an
action-preserving temporal sampler, optionally adds a CE head on the 512-D
backbone feature, and injects true checkpoint resume support into the source
in memory before executing it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import runpy
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def _value_after(argv: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Patch anchor {label!r} expected once, found {count}")
    return source.replace(old, new, 1)


def make_temporal_sampler(mode: str, min_overlap: float) -> Callable:
    import aug.temporal_augmentation_adaptive as temporal

    original = temporal.sample_two_views_indices
    mode = mode.lower()
    if mode == "independent":
        return original
    if mode not in {"shared", "overlap"}:
        raise ValueError(f"Unsupported temporal mode: {mode}")
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be in [0, 1]")

    def sample(T: int, n: int, rng=None) -> Tuple[List[int], List[int]]:
        temporal._check_inputs(T, n)
        rng = random if rng is None else rng
        if T <= n:
            base = temporal.sample_indices_strict(T, n)
        else:
            base = temporal._sample_one_adaptive_view(T, n, rng)
        if mode == "shared":
            return list(base), list(base)

        # Keep at least ceil(min_overlap*n) positions unchanged.  The remaining
        # positions receive a small local time jitter, so the two clips stay in
        # the same action phase while still being non-identical.
        keep = min(n, max(0, int(math.ceil(min_overlap * n))))
        change_count = n - keep
        changed_positions = set(rng.sample(range(n), change_count)) if change_count else set()
        view2 = list(base)
        for i in changed_positions:
            candidates = [x for x in (base[i] - 2, base[i] - 1, base[i] + 1, base[i] + 2) if 0 <= x < T]
            if candidates:
                view2[i] = rng.choice(candidates)
        view2 = temporal._clip_and_sort(view2, T)
        return list(base), view2

    return sample


def patch_training_source(source: str) -> str:
    source = _replace_once(
        source,
        "        ).to(device)\n\n        if (\n",
        """        ).to(device)

        # Round-2 optional CE head.  A pre-hook captures the input to the
        # projection head, i.e. the 512-D pooled backbone representation.
        if ROUND2_AUX_CE_WEIGHT > 0.0:
            projection = model.encoder_q.fc
            if not isinstance(projection, nn.Sequential) or not hasattr(projection[0], "in_features"):
                raise RuntimeError("Round-2 CE expects encoder_q.fc to be an MLP projection head")
            feature_dim = int(projection[0].in_features)
            model.round2_aux_classifier = nn.Linear(feature_dim, num_classes).to(device)

            def _round2_capture_backbone_feature(_module, inputs):
                model._round2_last_backbone_feature = inputs[0]

            projection.register_forward_pre_hook(_round2_capture_backbone_feature)

        if (
""",
        "aux-head",
    )
    source = _replace_once(
        source,
        "            features, target, loss_kcl, q, _ = model(im_q=view1, im_k=view2, labels=labels)\n            supcon_queue_anchor_stats = _compute_supcon_queue_anchor_stats(\n",
        """            features, target, loss_kcl, q, _ = model(im_q=view1, im_k=view2, labels=labels)
            loss_aux_ce = torch.zeros((), device=device, dtype=q.dtype)
            if ROUND2_AUX_CE_WEIGHT > 0.0:
                raw_model = model.module if hasattr(model, "module") else model
                backbone_feature = getattr(raw_model, "_round2_last_backbone_feature", None)
                if backbone_feature is None:
                    raise RuntimeError("Round-2 CE hook did not capture the 512-D backbone feature")
                logits_aux = raw_model.round2_aux_classifier(backbone_feature)
                loss_aux_ce = torch.nn.functional.cross_entropy(logits_aux, labels)
            supcon_queue_anchor_stats = _compute_supcon_queue_anchor_stats(
""",
        "aux-loss-compute",
    )
    source = _replace_once(
        source,
        "                if use_rel_loss:\n                    loss = loss + lambda_rel * loss_rel\n\n        # -----------------------------\n",
        """                if use_rel_loss:
                    loss = loss + lambda_rel * loss_rel

            if ROUND2_AUX_CE_WEIGHT > 0.0:
                loss = loss + ROUND2_AUX_CE_WEIGHT * loss_aux_ce

        # -----------------------------
""",
        "aux-loss-add",
    )
    source = _replace_once(
        source,
        "    losses_rel = AverageMeter(\"loss_rel\")\n",
        "    losses_rel = AverageMeter(\"loss_rel\")\n    losses_aux_ce = AverageMeter(\"loss_aux_ce\")\n",
        "aux-meter",
    )
    source = _replace_once(
        source,
        "        losses_rel.update(loss_rel.item(), n=bs)\n",
        "        losses_rel.update(loss_rel.item(), n=bs)\n        losses_aux_ce.update(loss_aux_ce.item(), n=bs)\n",
        "aux-meter-update",
    )
    source = _replace_once(
        source,
        "                f\"rel_loss={loss_rel.item():.4f} \"\n                f\"(avg_loss={losses.average:.4f}) \"\n",
        """                f"rel_loss={loss_rel.item():.4f} "
                f"aux_ce_loss={loss_aux_ce.item():.4f} "
                f"(avg_loss={losses.average:.4f}) "
""",
        "aux-log-current",
    )
    source = _replace_once(
        source,
        "                f\"(avg_rel={losses_rel.average:.4f})\"\n",
        "                f\"(avg_rel={losses_rel.average:.4f}) \"\n                f\"(avg_aux_ce={losses_aux_ce.average:.4f})\"\n",
        "aux-log-average",
    )
    source = _replace_once(
        source,
        "        proto_state = None\n\n        for epoch in range(args.start_epoch, args.epochs):\n",
        """        proto_state = None

        if ROUND2_RESUME_CHECKPOINT:
            try:
                resume_obj = torch.load(ROUND2_RESUME_CHECKPOINT, map_location=device, weights_only=False)
            except TypeError:
                resume_obj = torch.load(ROUND2_RESUME_CHECKPOINT, map_location=device)
            _unwrap_model(model).load_state_dict(resume_obj["state_dict"], strict=True)
            optimizer.load_state_dict(resume_obj["optimizer"])
            args.start_epoch = int(resume_obj.get("epoch", 0))
            if resume_obj.get("prototype_bank") is not None:
                proto_state = {}
                for key in (
                    "prototype_bank", "class_num_prototypes", "proto_rel_temperature_bank",
                    "sample_to_proto", "sample_to_class", "valid_sample_mask"
                ):
                    value = resume_obj.get(key)
                    proto_state[key] = value.to(device) if torch.is_tensor(value) else value
                proto_state["enable_prototype_temperature_scaling"] = bool(
                    resume_obj.get("enable_prototype_temperature_scaling", False)
                )
            log(f"[Round2 Resume] checkpoint={ROUND2_RESUME_CHECKPOINT} start_epoch={args.start_epoch}")

        for epoch in range(args.start_epoch, args.epochs):
""",
        "resume",
    )
    source = _replace_once(
        source,
        '                    "ablation_mode": args.ablation_mode,\n',
        '                    "ablation_mode": args.ablation_mode,\n                    "round2_aux_ce_weight": ROUND2_AUX_CE_WEIGHT,\n',
        "checkpoint-metadata",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--round2-original-script", required=True)
    parser.add_argument("--round2-temporal-mode", choices=["independent", "shared", "overlap"], default="independent")
    parser.add_argument("--round2-min-temporal-overlap", type=float, default=0.75)
    parser.add_argument("--round2-aux-ce-weight", type=float, default=0.0)
    parser.add_argument("--round2-auto-resume", action="store_true")
    parser.add_argument("--round2-validate-only", action="store_true")
    parser.add_argument("--round2-parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()

    original_script = Path(custom.round2_original_script).expanduser().resolve()
    if not original_script.is_file():
        raise FileNotFoundError(original_script)
    project_root = original_script.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if custom.round2_aux_ce_weight < 0:
        raise ValueError("round2 auxiliary CE weight must be non-negative")

    source = original_script.read_text(encoding="utf-8")
    patched = patch_training_source(source)
    if custom.round2_validate_only:
        compile(patched, str(original_script), "exec")
        print("Round-2 pretrain source patch validation: OK")
        return
    if custom.round2_parse_only:
        patched = _replace_once(
            patched,
            'if __name__ == "__main__":\n    worker(args)',
            'if __name__ == "__main__":\n    print("Round-2 pretrain command parse: OK")',
            "parse-only",
        )

    weight_save = _value_after(remaining, "--weight_save_path")
    epochs = int(_value_after(remaining, "--epochs", "200"))
    resume_checkpoint = ""
    if weight_save and not custom.round2_parse_only:
        output_dir = Path(weight_save)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = output_dir / f"checkpoint_{epochs:04d}.pth"
        if final_ckpt.is_file():
            print(f"[Round2 Skip] completed checkpoint exists: {final_ckpt}")
            return
        if custom.round2_auto_resume:
            candidates = sorted(output_dir.glob("checkpoint_*.pth"))
            if candidates:
                resume_checkpoint = str(candidates[-1])
        wrapper_record = {
            "original_script": str(original_script),
            "temporal_mode": custom.round2_temporal_mode,
            "min_temporal_overlap": custom.round2_min_temporal_overlap,
            "aux_ce_weight": custom.round2_aux_ce_weight,
            "auto_resume": bool(custom.round2_auto_resume),
            "resume_checkpoint": resume_checkpoint or None,
        }
        (output_dir / "round2_wrapper_args.json").write_text(
            json.dumps(wrapper_record, indent=2), encoding="utf-8"
        )

    import aug.temporal_augmentation_adaptive as temporal
    temporal.sample_two_views_indices = make_temporal_sampler(
        custom.round2_temporal_mode, custom.round2_min_temporal_overlap
    )

    sys.argv = [str(original_script), *remaining]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(original_script),
        "ROUND2_AUX_CE_WEIGHT": float(custom.round2_aux_ce_weight),
        "ROUND2_RESUME_CHECKPOINT": resume_checkpoint,
    }
    exec(compile(patched, str(original_script), "exec"), globals_dict, globals_dict)


if __name__ == "__main__":
    main()
