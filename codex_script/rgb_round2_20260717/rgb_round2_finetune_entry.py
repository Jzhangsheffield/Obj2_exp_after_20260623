#!/usr/bin/env python3
"""Execute the existing fine-tuning source with in-memory resume support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Fine-tune patch anchor {label!r} expected once, found {count}")
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = replace_once(
        source,
        "    scaler_obj = GradScaler(enabled=(device.type == \"cuda\" and args.enable_amp and (not use_bf16)))\n\n    config_payload = {\n",
        """    scaler_obj = GradScaler(enabled=(device.type == "cuda" and args.enable_amp and (not use_bf16)))

    round2_ft_start_epoch = 0
    if ROUND2_FT_AUTO_RESUME:
        periodic = sorted(Path(run_dir).glob("epoch_*.pth"))
        if periodic:
            resume_path = periodic[-1]
            try:
                resume_obj = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                resume_obj = torch.load(resume_path, map_location=device)
            model.load_state_dict(resume_obj["model_state_dict"], strict=True)
            if resume_obj.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_obj["optimizer_state_dict"])
            if scaler_obj.is_enabled() and resume_obj.get("scaler_state_dict") is not None:
                scaler_obj.load_state_dict(resume_obj["scaler_state_dict"])
            round2_ft_start_epoch = int(resume_obj.get("epoch", 0))
            print(f"[Round2 FT Resume] {resume_path} -> epoch {round2_ft_start_epoch}")

    config_payload = {
""",
        "resume-load",
    )
    source = replace_once(
        source,
        '    with open(log_file, "w", encoding="utf-8") as f:\n',
        '    round2_log_mode = "a" if round2_ft_start_epoch > 0 else "w"\n    with open(log_file, round2_log_mode, encoding="utf-8") as f:\n',
        "resume-log-mode",
    )
    source = replace_once(
        source,
        "    final_train_acc = None\n",
        """    if round2_ft_start_epoch > 0:
        round2_best_specs = (
            ("best_val.pth", "acc"),
            ("best_val_macro_f1.pth", "macro"),
            ("best_val_balanced.pth", "balanced"),
        )
        for round2_name, round2_kind in round2_best_specs:
            round2_path = Path(run_dir) / round2_name
            if not round2_path.is_file():
                continue
            try:
                round2_obj = torch.load(round2_path, map_location="cpu", weights_only=False)
            except TypeError:
                round2_obj = torch.load(round2_path, map_location="cpu")
            round2_info = round2_obj.get("extra_info", {})
            round2_value = float(round2_info.get("selection_metric_value", -1.0))
            round2_loss = round2_info.get("val_loss")
            round2_epoch = int(round2_obj.get("epoch", -1))
            if round2_kind == "acc":
                best_val_acc, best_val_acc_loss, best_val_acc_epoch = round2_value, round2_loss, round2_epoch
            elif round2_kind == "macro":
                best_val_macro_f1, best_val_macro_f1_loss, best_val_macro_f1_epoch = round2_value, round2_loss, round2_epoch
            else:
                best_val_balanced_acc, best_val_balanced_loss, best_val_balanced_epoch = round2_value, round2_loss, round2_epoch

    final_train_acc = None
""",
        "resume-best-metrics",
    )
    source = replace_once(
        source,
        "        for epoch in range(args.epochs):\n",
        "        for epoch in range(round2_ft_start_epoch, args.epochs):\n",
        "resume-loop",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--round2-original-script", required=True)
    parser.add_argument("--round2-auto-resume", action="store_true")
    parser.add_argument("--round2-validate-only", action="store_true")
    parser.add_argument("--round2-parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()
    original = Path(custom.round2_original_script).expanduser().resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    project_root = original.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    source = patch_source(original.read_text(encoding="utf-8"))
    compile(source, str(original), "exec")
    if custom.round2_validate_only:
        print("Round-2 fine-tune source patch validation: OK")
        return
    if custom.round2_parse_only:
        source = replace_once(
            source,
            "args = parser.parse_args()\n",
            "args = parser.parse_args()\n"
            "if globals().get('ROUND2_FT_PARSE_ONLY', False):\n"
            "    print('Round-2 fine-tune command parse: OK')\n"
            "    raise SystemExit(0)\n",
            "parse-only",
        )
    sys.argv = [str(original), *remaining]
    g = {
        "__name__": "__main__",
        "__file__": str(original),
        "ROUND2_FT_AUTO_RESUME": bool(custom.round2_auto_resume),
        "ROUND2_FT_PARSE_ONLY": bool(custom.round2_parse_only),
    }
    exec(compile(source, str(original), "exec"), g, g)


if __name__ == "__main__":
    main()
