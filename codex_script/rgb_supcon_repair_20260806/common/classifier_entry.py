#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repair-source", required=True); p.add_argument("--repair-src-root", required=True)
    p.add_argument("--repair-representation", choices=["rgb", "absdiff", "rgb_absdiff"], default="rgb")
    p.add_argument("--repair-temporal-mode", choices=["current", "t3_lfb"], default="current")
    p.add_argument("--repair-backbone-init", choices=["random", "kinetics400"], default="random")
    p.add_argument("--repair-freeze-patch-embed", action="store_true")
    p.add_argument("--repair-finetune-policy", choices=["native", "head", "partial", "full"], default="native")
    p.add_argument("--repair-parse-only", action="store_true")
    custom, remaining = p.parse_known_args()
    source_path = Path(custom.repair_source).resolve(); src_root = Path(custom.repair_src_root).resolve()
    package_root = Path(__file__).resolve().parents[1]
    for path in (package_root, src_root):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    from common.runtime_patch import configure_partial_finetune, install
    install(src_root, custom.repair_representation, custom.repair_temporal_mode,
            custom.repair_backbone_init, custom.repair_freeze_patch_embed)
    source = source_path.read_text(encoding="utf-8")
    if custom.repair_finetune_policy != "native":
        anchor = "    configure_finetune_mode(model, args.finetune_mode)\n"
        if source.count(anchor) != 1: raise RuntimeError("classifier finetune-policy anchor mismatch")
        source = source.replace(anchor, anchor + "    REPAIR_CONFIGURE_FINETUNE(model)\n", 1)
        train_anchor = "    else:\n        model.train()\n\n    total_seen = 0\n"
        if source.count(train_anchor) != 1: raise RuntimeError("classifier train-mode anchor mismatch")
        source = source.replace(train_anchor, "    else:\n        model.train()\n    REPAIR_AFTER_MODEL_TRAIN(model)\n\n    total_seen = 0\n", 1)
    if custom.repair_parse_only:
        anchor = 'if __name__ == "__main__":\n    main(args)'
        if source.count(anchor) != 1: raise RuntimeError("classifier parse-only anchor mismatch")
        source = source.replace(anchor, 'if __name__ == "__main__":\n    print("repair classifier command parse: OK")', 1)
    if not custom.repair_parse_only and "--save_path" in remaining:
        output = Path(remaining[remaining.index("--save_path") + 1]); output.mkdir(parents=True, exist_ok=True)
        (output / "repair_wrapper_args.json").write_text(json.dumps(vars(custom), indent=2), encoding="utf-8")
    sys.argv = [str(source_path), *remaining]
    def repair_configure(model):
        if custom.repair_finetune_policy == "partial": configure_partial_finetune(model)
        if custom.repair_freeze_patch_embed and custom.repair_finetune_policy in {"partial", "full"}:
            if hasattr(model.backbone, "conv_proj"): target = model.backbone.conv_proj
            elif hasattr(model.backbone, "patch_embed"): target = model.backbone.patch_embed.proj
            else: raise ValueError("patch freezing requested for a non-transformer backbone")
            for parameter in target.parameters(): parameter.requires_grad = False
    def repair_after_model_train(model):
        if custom.repair_finetune_policy != "partial": return
        for module in model.modules():
            parameters=list(module.parameters(recurse=True))
            if parameters and not any(parameter.requires_grad for parameter in parameters): module.eval()
        model.fc.train()
    namespace = {"__name__": "__main__", "__file__": str(source_path), "REPAIR_CONFIGURE_FINETUNE": repair_configure,
                 "REPAIR_AFTER_MODEL_TRAIN": repair_after_model_train}
    exec(compile(source, str(source_path), "exec"), namespace, namespace)

if __name__ == "__main__": main()
