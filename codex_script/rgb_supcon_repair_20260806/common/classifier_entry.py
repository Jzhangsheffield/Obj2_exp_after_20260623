#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repair-source", required=True); p.add_argument("--repair-src-root", required=True)
    p.add_argument("--repair-representation", choices=["rgb", "absdiff", "rgb_absdiff"], default="rgb")
    p.add_argument("--repair-temporal-mode", choices=["current", "t3_lfb"], default="current")
    p.add_argument("--repair-parse-only", action="store_true")
    custom, remaining = p.parse_known_args()
    source_path = Path(custom.repair_source).resolve(); src_root = Path(custom.repair_src_root).resolve()
    package_root = Path(__file__).resolve().parents[1]
    for path in (package_root, src_root):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from common.runtime_patch import install
    install(src_root, custom.repair_representation, custom.repair_temporal_mode)
    source = source_path.read_text(encoding="utf-8")
    if custom.repair_parse_only:
        anchor = 'if __name__ == "__main__":\n    main(args)'
        if source.count(anchor) != 1: raise RuntimeError("classifier parse-only anchor mismatch")
        source = source.replace(anchor, 'if __name__ == "__main__":\n    print("repair classifier command parse: OK")', 1)
    sys.argv = [str(source_path), *remaining]
    exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})

if __name__ == "__main__": main()
