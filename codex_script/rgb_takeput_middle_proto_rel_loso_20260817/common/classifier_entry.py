#!/usr/bin/env python3
"""Process-local R3D/MViT classifier adapter with an RGB-only import shim."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--package-source", required=True)
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--backbone-init", choices=["random", "kinetics400"], required=True)
    parser.add_argument("--parse-only", action="store_true")
    custom, remaining = parser.parse_known_args()
    source_path = Path(custom.package_source).resolve()
    src_root = Path(custom.src_root).resolve()
    project = src_root.parents[2]
    package_root = Path(__file__).resolve().parents[1]
    for path in (project, package_root, src_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from codex_script.rgb_supcon_repair_20260806.common.runtime_patch import install

    install(src_root, "rgb", "current", custom.backbone_init, False)
    optional_module = "aug.mindrove_augmentation_tensor_varlen"
    if optional_module not in sys.modules:
        try:
            __import__(optional_module)
        except ModuleNotFoundError as exc:
            if exc.name != optional_module:
                raise
            shim = types.ModuleType(optional_module)

            def unavailable(*_args, **_kwargs):
                raise RuntimeError("MindRove shim was called in an RGB-only run")

            shim.apply_mindrove_augmentation = unavailable
            sys.modules[optional_module] = shim

    source = source_path.read_text(encoding="utf-8")
    if custom.parse_only:
        anchor = 'if __name__ == "__main__":\n    main(args)'
        if source.count(anchor) != 1:
            raise RuntimeError("classifier parse-only anchor mismatch")
        source = source.replace(anchor, 'if __name__ == "__main__":\n    print("classifier command parse: OK")', 1)
    elif "--save_path" in remaining:
        output = Path(remaining[remaining.index("--save_path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "classifier_wrapper_args.json").write_text(
            json.dumps(vars(custom), indent=2), encoding="utf-8"
        )

    sys.argv = [str(source_path), *remaining]
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
