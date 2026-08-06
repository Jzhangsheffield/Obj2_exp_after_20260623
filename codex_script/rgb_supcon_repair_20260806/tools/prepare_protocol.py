#!/usr/bin/env python3
"""Create person-held-out manifests and audit run leakage/path coverage."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

def rows(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def run_key(row):
    key = str(row.get("original_key", ""))
    return re.sub(r"_clip_.*$", "", key)
def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")
def summary(values):
    return {"samples": len(values), "people": dict(Counter(str(x.get("person")) for x in values)), "classes": dict(Counter(str(x.get("tier1")) for x in values)), "runs": len({run_key(x) for x in values})}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--train", required=True); p.add_argument("--val", required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); train = rows(args.dataset_root / args.train); val = rows(args.dataset_root / args.val); all_rows = train + val
    by_name = {}
    for row in all_rows:
        by_name[str(row["sample_name"])] = row
    all_rows = list(by_name.values())
    outputs = {
        "train_MJ.jsonl": [x for x in all_rows if x.get("person") in {"M", "J"}],
        "val_MR.jsonl": [x for x in all_rows if x.get("person") == "MR"],
        "fold_M_train_JMR.jsonl": [x for x in all_rows if x.get("person") != "M"], "fold_M_val_M.jsonl": [x for x in all_rows if x.get("person") == "M"],
        "fold_J_train_MMR.jsonl": [x for x in all_rows if x.get("person") != "J"], "fold_J_val_J.jsonl": [x for x in all_rows if x.get("person") == "J"],
        "fold_MR_train_MJ.jsonl": [x for x in all_rows if x.get("person") != "MR"], "fold_MR_val_MR.jsonl": [x for x in all_rows if x.get("person") == "MR"],
    }
    for name, values in outputs.items(): write_jsonl(args.output / name, values)
    old_overlap = sorted({run_key(x) for x in train} & {run_key(x) for x in val})
    path_audit = {}
    for key in ("rgb_cam_00143", "rgb_cam_00143_motion_crop_m32", "mindrove"):
        missing = [x["sample_name"] for x in all_rows if not x.get(key) or not (args.dataset_root / x.get(key, "")).is_file()]
        path_audit[key] = {"missing": len(missing), "first_missing": missing[:10]}
    report = {"source": {"train": summary(train), "val": summary(val), "run_overlap_count": len(old_overlap), "run_overlap_first": old_overlap[:20]},
              "derived": {name: summary(values) for name, values in outputs.items()}, "paths": path_audit}
    (args.output / "protocol_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
