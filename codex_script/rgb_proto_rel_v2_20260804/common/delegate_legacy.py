#!/usr/bin/env python3
"""Delegate fine-tune/test commands to the verified legacy package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"finetune", "test", "summarize_test"}:
        raise SystemExit("Usage: delegate_legacy.py {finetune|test|summarize_test} [arguments]")
    action = sys.argv[1]
    project = Path(__file__).resolve().parents[3]
    names = {"finetune": "run_finetune.py", "test": "run_test.py", "summarize_test": "summarize_results.py"}
    target = project / "codex_script" / "rgb_required_5stages_20260719" / "common" / names[action]
    subprocess.run([sys.executable, "-u", str(target), *sys.argv[2:]], cwd=str(project), check=True)


if __name__ == "__main__":
    main()

