#!/usr/bin/env python3
"""Extract reproducible analysis tables from the downloaded Stage 1/2A/3A/4 results."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def avg(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def med(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(values) if values else None


def maximum(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return max(values) if values else None


def parse_classifier_log(path: Path) -> dict[int, dict]:
    epoch_pattern = re.compile(
        r"^\[(\d+)\].*?train loss: ([0-9.eE+-]+), train_acc: ([0-9.eE+-]+), "
        r"train_balanced_acc: ([0-9.eE+-]+), train_macro_f1: ([0-9.eE+-]+).*?"
        r"val loss: ([0-9.eE+-]+), val_acc: ([0-9.eE+-]+), "
        r"val_balanced_acc: ([0-9.eE+-]+), val_macro_f1: ([0-9.eE+-]+)"
    )
    result: dict[int, dict] = {}
    current = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = epoch_pattern.match(line)
        if match:
            values = list(map(float, match.groups()[1:]))
            current = int(match.group(1))
            result[current] = dict(zip(
                ["train_loss", "train_acc", "train_ba", "train_f1", "val_loss", "val_acc", "val_ba", "val_f1"],
                values,
            ))
        elif current is not None and line.strip().startswith("val_per_class_acc:"):
            result[current]["val_per_class_acc"] = json.loads(line.split(":", 1)[1].strip())
        elif current is not None and line.strip().startswith("val_per_class_support:"):
            result[current]["val_per_class_support"] = json.loads(line.split(":", 1)[1].strip())
    return result


def classifier_tables(results: Path) -> tuple[list[dict], list[dict]]:
    rows, per_class = [], []
    for path in sorted(results.glob("classifier/fold_*/*/*/*/summary.json")):
        data = read_json(path)
        rel = path.relative_to(results).parts
        stage, experiment = rel[2], rel[3]
        logs = parse_classifier_log(path.with_name("train_logs.txt"))
        best_epoch = int(data["best_val_balanced_epoch"])
        best_log = logs.get(best_epoch, {})
        row = {
            "fold": rel[1].replace("fold_", ""), "stage": stage, "experiment": experiment,
            "best_ba": data.get("best_val_balanced_acc"), "best_f1": data.get("best_val_macro_f1"),
            "best_acc": data.get("best_val_acc"), "best_ba_epoch": best_epoch,
            "final_train_acc": data.get("final_train_acc"), "final_val_acc": data.get("final_val_acc"),
            "final_val_ba": data.get("final_val_balanced_acc"), "final_val_f1": data.get("final_val_macro_f1"),
            "best_to_final_ba_drop": data.get("best_val_balanced_acc", 0) - data.get("final_val_balanced_acc", 0),
            "final_generalization_gap_acc": data.get("final_train_acc", 0) - data.get("final_val_acc", 0),
            "best_epoch_val_loss": best_log.get("val_loss"), "summary": str(path),
        }
        rows.append(row)
        recalls = best_log.get("val_per_class_acc", {})
        support = best_log.get("val_per_class_support", {})
        for name, recall in recalls.items():
            per_class.append({"stage": stage, "experiment": experiment, "best_ba_epoch": best_epoch,
                              "class": name, "recall": recall, "support": support.get(name)})
    baselines = {row["experiment"]: row for row in rows}
    for row in rows:
        for key, exp in (("delta_vs_sup_ba", "s0_sup"), ("delta_vs_direct_ba", "d0_k400_direct")):
            row[key] = row["best_ba"] - baselines[exp]["best_ba"]
    return rows, per_class


def load_debug(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def records_summary(records: list[dict], key: str, active_after: int | None = None) -> dict:
    chosen = records if active_after is None else [x for x in records if int(x.get("epoch", 0)) >= active_after]
    values = [x.get(key) for x in chosen]
    return {f"{key}_mean": avg(values), f"{key}_median": med(values), f"{key}_max": maximum(values), f"{key}_n": len(chosen)}


def pretrain_table(results: Path, plan: dict) -> list[dict]:
    catalog = {row["id"]: row for stage in plan["stages"].values() for row in stage}
    output = []
    for path in sorted(results.glob("pretrain/fold_*/*/*/debug_train_log.jsonl")):
        rel = path.relative_to(results).parts
        stage, experiment = rel[2], rel[3]
        cfg = catalog.get(experiment, {})
        records = load_debug(path)
        proto_start = int(cfg.get("proto_start", 10**9))
        rel_start = int(cfg.get("rel_start", 10**9))
        proto_records = [x for x in records if int(x.get("epoch", 0)) >= proto_start and float(cfg.get("lambda_proto", 0)) > 0]
        rel_records = [x for x in records if int(x.get("epoch", 0)) >= rel_start and float(cfg.get("lambda_rel", 0)) > 0]
        row = {
            "stage": stage, "experiment": experiment, "records": len(records),
            "proto_start": None if proto_start > 200 else proto_start,
            "rel_start": None if rel_start > 200 else rel_start,
            "nonfinite_records": sum(bool(x.get("nonfinite_check", {}).get("has_nonfinite")) for x in records),
            "final10_supcon_mean": avg(x.get("loss_supcon") for x in records if int(x.get("epoch", 0)) >= 191),
            "final10_grad_norm_mean": avg(x.get("grad_stats", {}).get("total_grad_norm") for x in records if int(x.get("epoch", 0)) >= 191),
            "final10_feature_std_mean": avg(x.get("feature_stats", {}).get("q_feature_std_mean") for x in records if int(x.get("epoch", 0)) >= 191),
            "proto_active_records": len(proto_records), "rel_active_records": len(rel_records),
            "proto_loss_mean_active": avg(x.get("loss_proto") for x in proto_records),
            "weighted_proto_mean_active": avg(x.get("weighted_proto_contrib") for x in proto_records),
            "weighted_proto_max_active": maximum(x.get("weighted_proto_contrib") for x in proto_records),
            "weighted_proto_to_sup_mean": avg(abs(float(x.get("weighted_proto_contrib", 0))) / max(abs(float(x.get("loss_supcon", 0))), 1e-12) for x in proto_records),
            "rel_loss_mean_active": avg(x.get("loss_rel") for x in rel_records),
            "weighted_rel_mean_active": avg(x.get("weighted_rel_contrib") for x in rel_records),
            "weighted_rel_max_active": maximum(x.get("weighted_rel_contrib") for x in rel_records),
            "weighted_rel_to_sup_mean": avg(abs(float(x.get("weighted_rel_contrib", 0))) / max(abs(float(x.get("loss_supcon", 0))), 1e-12) for x in rel_records),
        }
        if rel_start <= 200:
            before = [x for x in records if rel_start - 5 <= int(x.get("epoch", 0)) < rel_start]
            after = [x for x in records if rel_start <= int(x.get("epoch", 0)) <= rel_start + 5]
            for name, subset in (("rel_before5", before), ("rel_after5", after)):
                row[f"{name}_grad_mean"] = avg(x.get("grad_stats", {}).get("total_grad_norm") for x in subset)
                row[f"{name}_supcon_mean"] = avg(x.get("loss_supcon") for x in subset)
                row[f"{name}_feature_std_mean"] = avg(x.get("feature_stats", {}).get("q_feature_std_mean") for x in subset)
        output.append(row)
    return output


def tensor_health(checkpoint: Path) -> dict:
    import torch
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    bank = state.get("prototype_bank")
    class_counts = state.get("class_num_prototypes")
    assignment = state.get("sample_to_proto")
    sample_class = state.get("sample_to_class")
    valid = state.get("valid_sample_mask")
    if bank is None or class_counts is None or assignment is None or sample_class is None:
        del state
        return {"prototype_state": False}
    bank = bank.float()
    class_counts = class_counts.long()
    assignment = assignment.long()
    sample_class = sample_class.long()
    valid = valid.bool() if valid is not None else assignment.ge(0)
    within_cos, assignment_cv, min_fractions, max_fractions = [], [], [], []
    dead = 0
    active_vectors, active_classes = [], []
    assignment_counts = []
    for class_id in range(bank.shape[0]):
        count = int(class_counts[class_id])
        vectors = torch.nn.functional.normalize(bank[class_id, :count], dim=1)
        active_vectors.append(vectors)
        active_classes.extend([class_id] * count)
        if count > 1:
            cos = vectors @ vectors.T
            mask = ~torch.eye(count, dtype=torch.bool)
            within_cos.extend(cos[mask].tolist())
        counts = [int(((sample_class == class_id) & valid & (assignment == p)).sum()) for p in range(count)]
        assignment_counts.append(counts)
        dead += sum(x == 0 for x in counts)
        total = sum(counts)
        if total and count > 1:
            assignment_cv.append(statistics.pstdev(counts) / statistics.mean(counts))
            min_fractions.append(min(counts) / total)
            max_fractions.append(max(counts) / total)
    all_vectors = torch.cat(active_vectors)
    classes = torch.tensor(active_classes)
    cos = all_vectors @ all_vectors.T
    diff = classes[:, None].ne(classes[None, :])
    nearest_diff = cos.masked_fill(~diff, -float("inf")).max(dim=1).values
    result = {
        "prototype_state": True, "active_prototypes": int(class_counts.sum()), "dead_prototypes": dead,
        "valid_assignment_fraction": float(valid.float().mean()),
        "assignment_cv_mean": avg(assignment_cv), "assignment_cv_max": maximum(assignment_cv),
        "assignment_min_fraction_mean": avg(min_fractions), "assignment_min_fraction_min": min(min_fractions) if min_fractions else None,
        "assignment_max_fraction_mean": avg(max_fractions),
        "within_class_cos_mean": avg(within_cos), "within_class_cos_max": maximum(within_cos),
        "nearest_different_cos_mean": float(nearest_diff.mean()), "nearest_different_cos_max": float(nearest_diff.max()),
        "assignment_counts": json.dumps(assignment_counts),
    }
    del state, bank, assignment, sample_class, valid, all_vectors, cos
    return result


def prototype_health_table(results: Path) -> list[dict]:
    rows = []
    for checkpoint in sorted(results.glob("pretrain/fold_*/*/*/checkpoint_0200.pth")):
        rel = checkpoint.relative_to(results).parts
        row = {"fold": rel[1].replace("fold_", ""), "stage": rel[2], "experiment": rel[3]}
        row.update(tensor_health(checkpoint))
        rows.append(row)
        print(f"prototype health: {row['stage']}/{row['experiment']}", flush=True)
    return rows


def purity(assignments, environments) -> float:
    buckets = defaultdict(lambda: defaultdict(int))
    for proto, env in zip(assignments, environments):
        buckets[proto][env] += 1
    total = len(assignments)
    return sum(max(counts.values()) for counts in buckets.values()) / total if total else float("nan")


def environment_table(results: Path, permutations: int = 200) -> list[dict]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rows = []
    for summary_path in sorted(results.glob("prototype_environment/fold_*/*/*/environment_prototype_summary.json")):
        rel = summary_path.relative_to(results).parts
        summaries = read_json(summary_path)
        final = summaries[-1]
        contingency_path = summary_path.with_name("final_environment_contingency.csv")
        grouped = defaultdict(list)
        with contingency_path.open(encoding="utf-8-sig") as handle:
            for item in csv.DictReader(handle):
                grouped[item["class"]].append(item)
        class_arrays = []
        for items in grouped.values():
            assignments, environments = [], []
            for item in items:
                count = int(item["count"])
                assignments.extend([int(item["prototype"])] * count)
                environments.extend([item["lighting"]] * count)
            class_arrays.append((assignments, environments))
        rng = random.Random(20260812)
        null_nmi, null_ari, null_purity = [], [], []
        for _ in range(permutations):
            nmis, aris, purs = [], [], []
            for assignments, environments in class_arrays:
                shuffled = list(environments)
                rng.shuffle(shuffled)
                nmis.append(normalized_mutual_info_score(shuffled, assignments))
                aris.append(adjusted_rand_score(shuffled, assignments))
                purs.append(purity(assignments, shuffled))
            null_nmi.append(avg(nmis)); null_ari.append(avg(aris)); null_purity.append(avg(purs))
        row = {
            "fold": rel[1].replace("fold_", ""), "stage": rel[2], "experiment": rel[3],
            "epochs_analyzed": len(summaries),
            "nmi_final": final["mean_class_nmi"], "ari_final": final["mean_class_ari"],
            "purity_final": final["mean_class_purity"], "assignment_ids_used": final["assignment_ids_used"],
            "nmi_epoch_mean": avg(x["mean_class_nmi"] for x in summaries),
            "ari_epoch_mean": avg(x["mean_class_ari"] for x in summaries),
            "purity_epoch_mean": avg(x["mean_class_purity"] for x in summaries),
            "null_nmi_mean": avg(null_nmi), "null_ari_mean": avg(null_ari), "null_purity_mean": avg(null_purity),
            "nmi_permutation_p": (1 + sum(x >= final["mean_class_nmi"] for x in null_nmi)) / (permutations + 1),
            "ari_permutation_p": (1 + sum(x >= final["mean_class_ari"] for x in null_ari)) / (permutations + 1),
            "purity_permutation_p": (1 + sum(x >= final["mean_class_purity"] for x in null_purity)) / (permutations + 1),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.plan)
    classifier, per_class = classifier_tables(args.results)
    pretrain = pretrain_table(args.results, plan)
    environment = environment_table(args.results)
    health = prototype_health_table(args.results)
    for name, rows in (("classifier_metrics.csv", classifier), ("best_epoch_per_class.csv", per_class),
                       ("pretrain_diagnostics.csv", pretrain), ("environment_alignment.csv", environment),
                       ("prototype_health_final.csv", health)):
        write_csv(args.output / name, rows)
    payload = {"classifier": classifier, "pretrain": pretrain, "environment": environment, "prototype_health": health}
    (args.output / "analysis_data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"classifier": len(classifier), "pretrain": len(pretrain), "environment": len(environment), "health": len(health)}, indent=2))


if __name__ == "__main__":
    main()
