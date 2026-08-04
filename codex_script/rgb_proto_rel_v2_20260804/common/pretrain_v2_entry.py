#!/usr/bin/env python3
"""Runtime wrapper that injects V2 losses without modifying legacy sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _after(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"V2 source patch {label!r} expected one match, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v2-original-script", required=True)
    p.add_argument("--v2-temporal-mode", choices=["independent", "shared", "overlap"], default="shared")
    p.add_argument("--v2-min-temporal-overlap", type=float, default=1.0)
    p.add_argument("--v2-assignment-mode", choices=["same_view_soft", "teacher_soft", "teacher_balanced"], default="teacher_balanced")
    p.add_argument("--v2-assignment-temperature", type=float, default=0.05)
    p.add_argument("--v2-prediction-temperature", type=float, default=0.07)
    p.add_argument("--v2-sinkhorn-iterations", type=int, default=3)
    p.add_argument("--v2-balance-weight", type=float, default=0.2)
    p.add_argument("--v2-diversity-weight", type=float, default=0.1)
    p.add_argument("--v2-diversity-margin", type=float, default=0.85)
    p.add_argument("--v2-preview-momentum", type=float, default=0.5)
    p.add_argument("--v2-bank-momentum", type=float, default=0.99)
    p.add_argument("--v2-rel-mode", choices=["rank", "rank_direction"], default="rank_direction")
    p.add_argument("--v2-rel-topk", type=int, default=3)
    p.add_argument("--v2-rel-margin", type=float, default=0.05)
    p.add_argument("--v2-rel-temperature", type=float, default=0.05)
    p.add_argument("--v2-direction-weight", type=float, default=0.25)
    p.add_argument("--v2-direction-delta", type=float, default=0.005)
    p.add_argument("--v2-diagnostic-interval", type=int, default=50)
    p.add_argument("--v2-auto-resume", action="store_true")
    p.add_argument("--v2-parse-only", action="store_true")
    custom, remaining = p.parse_known_args()

    original = Path(custom.v2_original_script).resolve()
    project = original.parents[1]
    package_root = Path(__file__).resolve().parents[1]
    for path in (str(package_root), str(project)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import make_temporal_sampler, patch_training_source

    source = patch_training_source(original.read_text(encoding="utf-8"))
    source = _replace_once(source, "features, target, loss_kcl, q, _ = model(im_q=view1, im_k=view2, labels=labels)", "features, target, loss_kcl, q, k_teacher = model(im_q=view1, im_k=view2, labels=labels)", "teacher-feature")
    marker = "                # 1) prototype contrastive loss"
    start = source.rfind("                # ------------------------------------------------------------", 0, source.index(marker))
    end_marker = "                # 最终总损失"
    end = source.rfind("                # ------------------------------------------------------------", start, source.index(end_marker, start))
    aux = '''                # V2 auxiliary losses.
                v2_out = V2_CONTROLLER.compute(
                    q=q, teacher=k_teacher, labels=labels,
                    prototype_bank=prototype_bank,
                    class_num_prototypes=class_num_prototypes,
                    use_proto_loss=use_proto_loss, use_rel_loss=use_rel_loss,
                    epoch=epoch, step=step_idx,
                )
                loss_proto = v2_out["loss_proto"]
                loss_rel = v2_out["loss_rel"]
                proto_ids = v2_out["hard_ids"]

'''
    source = source[:start] + aux + source[end:]
    old_update = '''            ema_update_prototype_bank_(
                prototype_bank=proto_state["prototype_bank"],
                q=q.detach(),
                labels=labels.detach(),
                proto_ids=proto_ids.detach(),
                bank_ema_momentum=proto_ema_momentum,
                class_num_prototypes=proto_state["class_num_prototypes"],
            )'''
    new_update = '''            V2_CONTROLLER.update_bank_(
                prototype_bank=proto_state["prototype_bank"],
                teacher=k_teacher.detach(), labels=labels.detach(),
                responsibilities=v2_out["responsibilities"].detach(),
                class_num_prototypes=proto_state["class_num_prototypes"],
            )'''
    source = _replace_once(source, old_update, new_update, "teacher-ema-update")
    if custom.v2_parse_only:
        compile(source, str(original), "exec")
        print("Proto/Rel V2 source patch validation: OK")
        return

    from common.v2_losses import V2Config, V2Controller

    output = _after(remaining, "--weight_save_path")
    resume_checkpoint = ""
    if output and custom.v2_auto_resume and not custom.v2_parse_only:
        out_path = Path(output)
        epochs = int(_after(remaining, "--epochs", "200"))
        final_checkpoint = out_path / f"checkpoint_{epochs:04d}.pth"
        if final_checkpoint.is_file():
            print(f"[V2] Final checkpoint already exists; nothing to do: {final_checkpoint}")
            return
        candidates = []
        for candidate in out_path.glob("checkpoint_*.pth"):
            match = re.fullmatch(r"checkpoint_(\d+)\.pth", candidate.name)
            if match and int(match.group(1)) < epochs:
                candidates.append((int(match.group(1)), candidate))
        if candidates:
            resume_checkpoint = str(max(candidates, key=lambda item: item[0])[1])
            print(f"[V2] Auto-resume from: {resume_checkpoint}")
    diagnostic = str(Path(output) / "v2_diagnostics.jsonl") if output and not custom.v2_parse_only else None
    cfg = V2Config(
        assignment_mode=custom.v2_assignment_mode, assignment_temperature=custom.v2_assignment_temperature,
        prediction_temperature=custom.v2_prediction_temperature, sinkhorn_iterations=custom.v2_sinkhorn_iterations,
        balance_weight=custom.v2_balance_weight, diversity_weight=custom.v2_diversity_weight,
        diversity_cos_margin=custom.v2_diversity_margin, preview_momentum=custom.v2_preview_momentum,
        bank_momentum=custom.v2_bank_momentum, rel_mode=custom.v2_rel_mode,
        rel_topk_classes=custom.v2_rel_topk, rel_margin=custom.v2_rel_margin,
        rel_temperature=custom.v2_rel_temperature, direction_weight=custom.v2_direction_weight,
        direction_delta=custom.v2_direction_delta, diagnostic_interval=custom.v2_diagnostic_interval,
        diagnostic_path=diagnostic,
    )
    if output and not custom.v2_parse_only:
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        (out / "v2_effective_config.json").write_text(json.dumps(vars(cfg), indent=2), encoding="utf-8")

    import aug.temporal_augmentation_adaptive as temporal
    temporal.sample_two_views_indices = make_temporal_sampler(custom.v2_temporal_mode, custom.v2_min_temporal_overlap)
    sys.argv = [str(original)] + remaining
    namespace = {"__name__": "__main__", "__file__": str(original), "ROUND2_AUX_CE_WEIGHT": 0.0, "ROUND2_RESUME_CHECKPOINT": resume_checkpoint, "V2_CONTROLLER": V2Controller(cfg)}
    exec(compile(source, str(original), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
