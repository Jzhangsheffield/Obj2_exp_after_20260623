#!/usr/bin/env python3
"""Fine-tune/test entry adding resume plus the selected temporal backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--required-original-script", required=True)
    parser.add_argument("--required-backbone-temporal-mode", choices=["current", "t3_lfb"], default="current")
    parser.add_argument("--required-auto-resume", action="store_true")
    parser.add_argument("--required-validate-only", action="store_true")
    parser.add_argument("--required-parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()
    original = Path(custom.required_original_script).expanduser().resolve()
    project = original.parents[1]
    package_root = Path(__file__).resolve().parents[1]
    for path in (project, package_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from codex_script.rgb_round2_20260717.rgb_round2_finetune_entry import patch_source, replace_once
    source = patch_source(original.read_text(encoding="utf-8"))
    source = replace_once(
        source,
        "    config_payload = {\n",
        "    config_payload = {\n"
        "        'required_backbone_temporal_mode': REQUIRED_BACKBONE_TEMPORAL_MODE,\n",
        "config-temporal-mode",
    )
    compile(source, str(original), "exec")
    if custom.required_validate_only:
        print("Required fine-tune source patch validation: OK")
        return
    if custom.required_parse_only:
        source = replace_once(
            source,
            "args = parser.parse_args()\n",
            "args = parser.parse_args()\n"
            "if globals().get('ROUND2_FT_PARSE_ONLY', False):\n"
            "    print('Required fine-tune command parse: OK')\n"
            "    raise SystemExit(0)\n",
            "parse-only",
        )

    import backbone.resnet as resnet3d
    from common.temporal_backbone import install_generate_model_mode
    install_generate_model_mode(resnet3d, custom.required_backbone_temporal_mode)
    sys.argv = [str(original)] + remaining
    namespace = {
        "__name__": "__main__",
        "__file__": str(original),
        "ROUND2_FT_AUTO_RESUME": bool(custom.required_auto_resume),
        "ROUND2_FT_PARSE_ONLY": bool(custom.required_parse_only),
        "REQUIRED_BACKBONE_TEMPORAL_MODE": custom.required_backbone_temporal_mode,
    }
    exec(compile(source, str(original), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
