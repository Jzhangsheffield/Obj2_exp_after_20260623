#!/usr/bin/env python3
"""Fine-tune/test wrapper with resume and camera-specific test artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codex_script.rgb_round2_20260717.rgb_round2_finetune_entry import (  # noqa: E402
    patch_source as patch_resume_source,
    replace_once,
)


def patch_test_artifacts(source: str) -> str:
    source = replace_once(
        source,
        '            f"{weight_stem}_per_sample_test.csv"\n',
        '            f"{weight_stem}_{DUALCAM_TEST_TAG}_per_sample_test.csv"\n',
        "dualcam-per-sample-name",
    )
    source = replace_once(
        source,
        '            f"{weight_stem}_test_metrics.json"\n',
        '            f"{weight_stem}_{DUALCAM_TEST_TAG}_test_metrics.json"\n',
        "dualcam-metrics-name",
    )
    source = replace_once(
        source,
        '            "use_modality": args.use_modality,\n            "num_samples": num_samples,\n',
        '            "use_modality": args.use_modality,\n'
        '            "rgb_camera_id": args.rgb_camera_id,\n'
        '            "num_samples": num_samples,\n',
        "dualcam-summary-camera",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dualcam-original-script", required=True)
    parser.add_argument("--dualcam-auto-resume", action="store_true")
    parser.add_argument("--dualcam-test-tag", default="eval")
    parser.add_argument("--dualcam-validate-only", action="store_true")
    parser.add_argument("--dualcam-parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", custom.dualcam_test_tag).strip("_") or "eval"

    original = Path(custom.dualcam_original_script).expanduser().resolve()
    if not original.is_file():
        raise FileNotFoundError(original)
    root = original.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    source = patch_test_artifacts(patch_resume_source(original.read_text(encoding="utf-8")))
    compile(source, str(original), "exec")
    if custom.dualcam_validate_only:
        print("Dual-camera fine-tune/test source validation: OK")
        return
    if custom.dualcam_parse_only:
        source = replace_once(
            source,
            "args = parser.parse_args()\n",
            "args = parser.parse_args()\n"
            "if globals().get('ROUND2_FT_PARSE_ONLY', False):\n"
            "    print('Dual-camera fine-tune/test command parse: OK')\n"
            "    raise SystemExit(0)\n",
            "dualcam-parse-only",
        )
    sys.argv = [str(original), *remaining]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(original),
        "ROUND2_FT_AUTO_RESUME": bool(custom.dualcam_auto_resume),
        "ROUND2_FT_PARSE_ONLY": bool(custom.dualcam_parse_only),
        "DUALCAM_TEST_TAG": tag,
    }
    exec(compile(source, str(original), "exec"), globals_dict, globals_dict)


if __name__ == "__main__":
    main()
