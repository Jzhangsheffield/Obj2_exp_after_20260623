#!/usr/bin/env python3
"""Add full logits/probabilities to test output without editing project source."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def value_after(argv: list[str], flag: str) -> str:
    if flag not in argv or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"Missing required argument {flag}")
    return argv[argv.index(flag) + 1]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Detailed test patch anchor {label!r} matched {count} times")
    return source.replace(old, new, 1)


def main() -> None:
    argv = sys.argv[1:]
    patch_only = "--detail-patch-only" in argv
    if patch_only:
        argv.remove("--detail-patch-only")
    source_path = Path(value_after(argv, "--repair-source")).resolve()
    save_path = Path(value_after(argv, "--save_path")).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        probs_cpu = probs.detach().cpu()\n        loss_cpu = per_sample_loss.detach().cpu()\n",
        "        probs_cpu = probs.detach().cpu()\n        logits_cpu = logits.detach().cpu()\n        loss_cpu = per_sample_loss.detach().cpu()\n",
        "logit tensor",
    )
    source = replace_once(
        source,
        '                "sample_loss": float(loss_cpu[i].item()),\n                "correct": int(is_correct),\n',
        '                "sample_loss": float(loss_cpu[i].item()),\n'
        '                "logits_json": json.dumps([float(x) for x in logits_cpu[i].tolist()]),\n'
        '                "probabilities_json": json.dumps([float(x) for x in probs_cpu[i].tolist()]),\n'
        '                "correct": int(is_correct),\n',
        "detailed row",
    )
    source = replace_once(
        source,
        '            "sample_loss",\n            "correct",\n',
        '            "sample_loss",\n            "logits_json",\n            "probabilities_json",\n            "correct",\n',
        "detailed columns",
    )
    source = replace_once(source, "        weight_dir = str(weight_path_obj.parent)\n", "        weight_dir = str(args.save_path)\n", "test output directory")
    patched_source = save_path / "_runtime_detailed_classifier.py"
    patched_source.write_text(source, encoding="utf-8")
    compile(source, str(patched_source), "exec")
    if patch_only:
        print(f"Detailed classifier patch: OK ({patched_source})")
        return

    codex_script_root = Path(__file__).resolve().parents[3]
    repair_entry = codex_script_root / "rgb_supcon_repair_20260806" / "common" / "classifier_entry.py"
    if not repair_entry.is_file():
        raise FileNotFoundError(repair_entry)
    forwarded = list(argv)
    forwarded[forwarded.index("--repair-source") + 1] = str(patched_source)
    subprocess.run([sys.executable, "-u", str(repair_entry), *forwarded], check=True)


if __name__ == "__main__":
    main()
