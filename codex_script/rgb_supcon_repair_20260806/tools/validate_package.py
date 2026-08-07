#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,required=True);p.add_argument("--dataset-root",type=Path,required=True);args=p.parse_args();root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/"config"/"experiment_plan.json").read_text(encoding="utf-8"));errors=[]
    for rel in plan["source_manifests"].values():
        if not (args.dataset_root/rel).is_file():errors.append(f"missing dataset file: {args.dataset_root/rel}")
    base=args.project_root/plan["base_source_package_rel"]
    for rel in ("src/train/pretrain_supcon.py","src/ft_and_test/train_classifier.py","src/backbone/video_backbone.py"):
        if not (base/rel).is_file():errors.append(f"missing base source: {base/rel}")
    for path in root.rglob("*.py"):
        try:compile(path.read_text(encoding="utf-8"),str(path),"exec")
        except Exception as e:errors.append(f"compile {path}: {e}")
    try:
        sys.path[:0] = [str(root), str(args.project_root)]
        from common.pretrain_entry import patch_lr, patch_teacher
        from codex_script.rgb_round2_20260717.rgb_round2_pretrain_entry import patch_training_source
        pretrain_source = (base/"src/train/pretrain_supcon.py").read_text(encoding="utf-8")
        compile(patch_teacher(patch_lr(patch_training_source(pretrain_source))), str(base/"src/train/pretrain_supcon.py"), "exec")
        classifier_source = (base/"src/ft_and_test/train_classifier.py").read_text(encoding="utf-8")
        if classifier_source.count('if __name__ == "__main__":\n    main(args)') != 1:
            errors.append("classifier parse-only anchor mismatch")
        if classifier_source.count("    configure_finetune_mode(model, args.finetune_mode)\n") != 1:
            errors.append("classifier Stage 6 finetune-policy anchor mismatch")
        if classifier_source.count("    else:\n        model.train()\n\n    total_seen = 0\n") != 1:
            errors.append("classifier Stage 6 train-mode anchor mismatch")
    except Exception as e:
        errors.append(f"source patch validation: {e}")
    for stage,rows in plan["stages"].items():
        ids=[x["id"] for x in rows];inds=[x["index"] for x in rows]
        if len(ids)!=len(set(ids)) or inds!=list(range(len(rows))):errors.append(f"bad ids/indices in {stage}")
        for row in rows:
            bs=int(row.get("batch_size",plan["pretrain_common"]["batch_size"]));q=int(plan["pretrain_common"]["queue_size"])
            if q%bs:errors.append(f"queue {q} not divisible by batch {bs}: {row['id']}")
            if row["representation"]=="rgb_absdiff" and not row["backbone"].startswith("dual_"):errors.append(f"six-channel representation needs dual backbone: {row['id']}")
    stage6=plan.get("stage6",{}); rows=stage6.get("stage6a",[])
    if [x.get("index") for x in rows] != list(range(5)): errors.append("Stage 6A must contain contiguous indices 0-4")
    allowed={"resnet3d10","tv_r3d18","r2plus1d18","mvit_v2_s","swin3d_t"}
    for row in rows:
        if row.get("backbone") not in allowed: errors.append(f"unsupported Stage 6 backbone: {row.get('backbone')}")
        if row.get("backbone_init") not in {"random","kinetics400"}: errors.append(f"bad Stage 6 init: {row.get('id')}")
    try:
        selection=json.loads((root/"config"/"stage6_selection.json").read_text(encoding="utf-8"));known={x["id"] for x in rows}
        for exp_id in selection.get("stage6b_candidates",[]):
            if exp_id not in known: errors.append(f"unknown Stage 6 candidate: {exp_id}")
        if selection.get("stage6b_winner") not in known: errors.append("unknown Stage 6B winner")
        if selection.get("stage6b_policy") not in {"head","partial","full"}: errors.append("bad Stage 6B policy")
    except Exception as e: errors.append(f"Stage 6 selection validation: {e}")
    if errors:raise RuntimeError("Validation failed:\n- "+"\n- ".join(errors))
    counts={k:len(v) for k,v in plan["stages"].items()};counts["stage6a"]=len(rows)
    print(json.dumps({"status":"OK","python":sys.version,"package":str(root),"experiments":counts},indent=2))
if __name__=="__main__":main()
