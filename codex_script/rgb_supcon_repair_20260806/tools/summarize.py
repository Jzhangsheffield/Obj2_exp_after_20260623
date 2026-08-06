#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--results-root",type=Path,required=True);args=p.parse_args();out=args.results_root/"summary";out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((args.results_root/"diagnostics").glob("*/*/diagnostics.json")) if (args.results_root/"diagnostics").is_dir() else []:
        d=json.loads(path.read_text(encoding="utf-8"));b=d["features"]["backbone"];normal=b["normal_predictive"];r={"stage":path.parent.parent.name,"experiment":path.parent.name,"backbone":d["backbone"],"representation":d["representation"],"n_frames":d["n_frames"],"temporal_mode":d["temporal_mode"],"train_silhouette":b["train_geometry"]["silhouette_cosine"],"val_silhouette":b["val_geometry"]["silhouette_cosine"],"effective_rank":b["val_geometry"]["effective_rank"],"top5_variance":b["val_geometry"]["top5_variance_fraction"],"frozen_linear_ba":normal["linear_balanced_accuracy"]}
        base=normal["linear_balanced_accuracy"]
        for mode,v in b["perturbations"].items():r[f"{mode}_cosine"]=v["original_perturbed_mean_cosine"];r[f"{mode}_ba_drop"]=base-v["linear_balanced_accuracy"]
        rows.append(r)
    if rows:
        fields=list(rows[0]);
        with (out/"diagnostic_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        ranked=sorted(rows,key=lambda x:(x["frozen_linear_ba"],x["effective_rank"]),reverse=True);(out/"diagnostic_ranking.json").write_text(json.dumps(ranked,indent=2,ensure_ascii=False),encoding="utf-8")
    crossval=[]
    cv_root=args.results_root/"diagnostics_crossval"
    for path in sorted(cv_root.glob("*/*/*/diagnostics.json")) if cv_root.is_dir() else []:
        d=json.loads(path.read_text(encoding="utf-8"));b=d["features"]["backbone"];normal=b["normal_predictive"]
        crossval.append({"profile":path.parents[2].name,"stage":path.parents[1].name,"experiment":path.parent.name,"frozen_linear_ba":normal["linear_balanced_accuracy"],"val_silhouette":b["val_geometry"]["silhouette_cosine"],"effective_rank":b["val_geometry"]["effective_rank"]})
    if crossval:
        with (out/"crossval_diagnostic_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(crossval[0]));w.writeheader();w.writerows(crossval)
    test_rows=[]
    for path in sorted((args.results_root/"test"/"stage5").glob("*/test_results.csv")) if (args.results_root/"test"/"stage5").is_dir() else []:
        with path.open(encoding="utf-8-sig") as f:data=list(csv.DictReader(f))
        if data:test_rows.append({"experiment":path.parent.name,**data[-1]})
    if test_rows:
        fields=sorted({k for r in test_rows for k in r});
        with (out/"locked_test_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(test_rows)
    print(json.dumps({"diagnostic_runs":len(rows),"crossval_runs":len(crossval),"test_runs":len(test_rows),"output":str(out)},indent=2))
if __name__=="__main__":main()
