#!/usr/bin/env python3
"""Offline validation for config, temporal samplers, and source patch anchors."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rgb_round2_pretrain_entry import make_temporal_sampler


def main() -> None:
    cfg = json.loads((HERE / "config" / "round2_config.json").read_text(encoding="utf-8"))
    indices = [int(x["index"]) for x in cfg["experiments"]]
    assert indices == list(range(9)), indices
    ft_indices = [int(x["index"]) for x in cfg["finetune_plan"]]
    assert ft_indices == list(range(9)), ft_indices
    for overlap in (0.75, 1.0):
        mode = "shared" if overlap == 1.0 else "overlap"
        sampler = make_temporal_sampler(mode, overlap)
        for T in (12, 16, 40, 100):
            a, b = sampler(T, 16, random.Random(7))
            assert len(a) == len(b) == 16
            if mode == "shared":
                assert a == b
            else:
                # Temporal views are sorted, and short clips can contain
                # repeated upsampled frames.  Multiset intersection therefore
                # measures shared source frames more faithfully than comparing
                # array positions after sorting.
                shared = sum((Counter(a) & Counter(b)).values()) / 16
                assert shared >= 0.75, (T, shared, a, b)

    pretrain_original = ROOT / "train" / "MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
    finetune_original = ROOT / "ft_and_test" / "train_mapstyle_finetune_and_test.py"
    subprocess.run([
        sys.executable, str(HERE / "rgb_round2_pretrain_entry.py"),
        "--round2-original-script", str(pretrain_original), "--round2-validate-only"
    ], check=True)
    subprocess.run([
        sys.executable, str(HERE / "rgb_round2_finetune_entry.py"),
        "--round2-original-script", str(finetune_original), "--round2-validate-only"
    ], check=True)
    print("Round-2 package validation: OK")


if __name__ == "__main__":
    main()
