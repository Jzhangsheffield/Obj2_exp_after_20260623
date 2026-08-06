#!/usr/bin/env python3
"""Run the isolated SupLoss trainer with repair variants and optional IMU guidance."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1: raise RuntimeError(f"Patch anchor {label!r} expected once, found {count}")
    return source.replace(old, new, 1)
def value_after(argv, flag, default=None):
    try: return argv[argv.index(flag) + 1]
    except (ValueError, IndexError): return default

def patch_lr(source):
    old = '''    lr = args.learning_rate
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            if epoch >= milestone:
                lr *= 0.1
'''
    new = '''    base_lr = args.learning_rate
    warmup = int(REPAIR_LR_WARMUP_EPOCHS)
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
    return replace_once(source, old, new, "learning-rate-warmup")

def patch_teacher(source):
    source = replace_once(source,
        "            projection.register_forward_pre_hook(_round2_capture_backbone_feature)\n\n        if (\n",
        '''            projection.register_forward_pre_hook(_round2_capture_backbone_feature)
            if REPAIR_XMODAL_WEIGHT > 0.0 or REPAIR_XREL_WEIGHT > 0.0:
                model.repair_xmodal_adapter = nn.Linear(feature_dim, REPAIR_TEACHER_DIM).to(device)

        if (
''', "teacher-adapter")
    source = replace_once(source,
        "                loss_aux_ce = torch.nn.functional.cross_entropy(logits_aux, labels)\n            supcon_queue_anchor_stats = _compute_supcon_queue_anchor_stats(\n",
        '''                loss_aux_ce = torch.nn.functional.cross_entropy(logits_aux, labels)
            loss_xmodal = torch.zeros((), device=device, dtype=q.dtype)
            loss_xrel = torch.zeros((), device=device, dtype=q.dtype)
            if REPAIR_XMODAL_WEIGHT > 0.0 or REPAIR_XREL_WEIGHT > 0.0:
                raw_model = model.module if hasattr(model, "module") else model
                backbone_feature = getattr(raw_model, "_round2_last_backbone_feature", None)
                teacher_feature = REPAIR_TEACHER_FEATURES.index_select(0, global_index.detach().cpu()).to(device=device, dtype=backbone_feature.dtype)
                mapped = torch.nn.functional.normalize(raw_model.repair_xmodal_adapter(backbone_feature), dim=1)
                teacher_norm = torch.nn.functional.normalize(teacher_feature, dim=1)
                if REPAIR_XMODAL_WEIGHT > 0.0:
                    loss_xmodal = (1.0 - (mapped * teacher_norm).sum(dim=1)).mean()
                if REPAIR_XREL_WEIGHT > 0.0:
                    prototypes = torch.nn.functional.normalize(REPAIR_TEACHER_PROTOTYPES.to(device=device, dtype=mapped.dtype), dim=1)
                    student_logp = torch.nn.functional.log_softmax((mapped @ prototypes.T) / REPAIR_XREL_TEMPERATURE, dim=1)
                    teacher_prob = torch.nn.functional.softmax((teacher_norm @ prototypes.T) / REPAIR_XREL_TEMPERATURE, dim=1)
                    loss_xrel = torch.nn.functional.kl_div(student_logp, teacher_prob, reduction="batchmean")
            supcon_queue_anchor_stats = _compute_supcon_queue_anchor_stats(
''', "teacher-loss")
    source = replace_once(source,
        "            if ROUND2_AUX_CE_WEIGHT > 0.0:\n                loss = loss + ROUND2_AUX_CE_WEIGHT * loss_aux_ce\n\n        # -----------------------------\n",
        '''            if ROUND2_AUX_CE_WEIGHT > 0.0:
                loss = loss + ROUND2_AUX_CE_WEIGHT * loss_aux_ce
            if REPAIR_XMODAL_WEIGHT > 0.0:
                loss = loss + REPAIR_XMODAL_WEIGHT * loss_xmodal
            if REPAIR_XREL_WEIGHT > 0.0:
                loss = loss + REPAIR_XREL_WEIGHT * loss_xrel

        # -----------------------------
''', "teacher-loss-add")
    return source

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repair-source", required=True); p.add_argument("--repair-src-root", required=True)
    p.add_argument("--repair-representation", choices=["rgb", "absdiff", "rgb_absdiff"], default="rgb")
    p.add_argument("--repair-temporal-mode", choices=["current", "t3_lfb"], default="current")
    p.add_argument("--repair-lr-warmup-epochs", type=int, default=0)
    p.add_argument("--repair-aux-ce-weight", type=float, default=0.0)
    p.add_argument("--repair-xmodal-weight", type=float, default=0.0); p.add_argument("--repair-xrel-weight", type=float, default=0.0)
    p.add_argument("--repair-xrel-temperature", type=float, default=0.1); p.add_argument("--repair-teacher-cache")
    p.add_argument("--repair-auto-resume", action="store_true"); p.add_argument("--repair-parse-only", action="store_true")
    custom, remaining = p.parse_known_args()
    source_path = Path(custom.repair_source).resolve(); src_root = Path(custom.repair_src_root).resolve()
    project = source_path.parents[4]; package_root = Path(__file__).resolve().parents[1]
    for path in (package_root, project, src_root):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from common.runtime_patch import install
    install(src_root, custom.repair_representation, custom.repair_temporal_mode)
    from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import patch_training_source
    source = patch_teacher(patch_lr(patch_training_source(source_path.read_text(encoding="utf-8"))))
    if custom.repair_parse_only:
        source = replace_once(source, 'if __name__ == "__main__":\n    worker(args)', 'if __name__ == "__main__":\n    print("repair pretrain command parse: OK")', "parse-only")
    output_text = value_after(remaining, "--weight_save_path"); epochs = int(value_after(remaining, "--epochs", "200")); resume = ""
    if output_text and not custom.repair_parse_only:
        out = Path(output_text); out.mkdir(parents=True, exist_ok=True)
        if (out / f"checkpoint_{epochs:04d}.pth").is_file(): print(f"[Skip] completed: {out}"); return
        candidates = sorted(out.glob("checkpoint_*.pth")) if custom.repair_auto_resume else []
        resume = str(candidates[-1]) if candidates else ""
        (out / "repair_wrapper_args.json").write_text(json.dumps(vars(custom) | {"resume_checkpoint": resume or None}, indent=2), encoding="utf-8")
    teacher_needed = custom.repair_xmodal_weight > 0 or custom.repair_xrel_weight > 0
    if teacher_needed and custom.repair_aux_ce_weight <= 0: raise ValueError("IMU guidance requires aux_ce_weight > 0 for the shared feature hook")
    teacher_features = teacher_prototypes = None; teacher_dim = 512
    if teacher_needed:
        if not custom.repair_teacher_cache: raise ValueError("--repair-teacher-cache is required for IMU guidance")
        import torch
        obj = torch.load(custom.repair_teacher_cache, map_location="cpu", weights_only=False)
        teacher_features = obj["features"].float(); teacher_prototypes = obj["class_prototypes"].float(); teacher_dim = int(teacher_features.shape[1])
    sys.argv = [str(source_path), *remaining]
    namespace = {"__name__": "__main__", "__file__": str(source_path), "ROUND2_AUX_CE_WEIGHT": float(custom.repair_aux_ce_weight),
        "ROUND2_RESUME_CHECKPOINT": resume, "REPAIR_LR_WARMUP_EPOCHS": int(custom.repair_lr_warmup_epochs),
        "REPAIR_XMODAL_WEIGHT": float(custom.repair_xmodal_weight), "REPAIR_XREL_WEIGHT": float(custom.repair_xrel_weight),
        "REPAIR_XREL_TEMPERATURE": float(custom.repair_xrel_temperature), "REPAIR_TEACHER_FEATURES": teacher_features,
        "REPAIR_TEACHER_PROTOTYPES": teacher_prototypes, "REPAIR_TEACHER_DIM": teacher_dim}
    exec(compile(source, str(source_path), "exec"), namespace, namespace)

if __name__ == "__main__": main()
