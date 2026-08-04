#!/usr/bin/env python3
"""Summarize V2 pretraining diagnostics without touching the locked test set."""

from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from config_utils import load_stage, roots


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--project-root")
    a = p.parse_args(); master, stage, _ = load_stage(Path(a.config).resolve()); project, _ = roots(master, a.project_root, None)
    rows = []
    for exp in stage["pretrain_experiments"]:
        path = project / stage["pretrain_output_rel"] / exp["id"] / "v2_diagnostics.jsonl"
        records = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try: records.append(json.loads(line))
                except json.JSONDecodeError: pass
        tail = records[-20:]
        row = {"experiment_id": exp["id"], "seed": exp.get("seed", 1), "records": len(records), "status": "complete" if records else "missing"}
        for key in ["assignment_entropy", "assignment_soft_mass_min", "assignment_soft_mass_max", "dead_prototypes_in_batch", "same_class_proto_cos_mean", "same_class_proto_cos_max", "bank_update_mean", "bank_update_max", "proto_assign", "proto_balance", "proto_diversity", "rel_rank", "rel_direction", "hard_negative_similarity", "margin_violation"]:
            vals = [float(x[key]) for x in tail if key in x]
            row["tail_" + key] = sum(vals) / len(vals) if vals else ""
        rows.append(row)
    outdir = project / stage["pretrain_output_rel"] / "analysis"; outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "v2_pretrain_diagnostics.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False)); print(f"Wrote {out}")


if __name__ == "__main__": main()
