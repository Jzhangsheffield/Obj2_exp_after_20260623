#!/usr/bin/env python3
"""Collect per-run classifier summaries and per-epoch curves into tables."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"^\[(?P<epoch>\d+)\].*?train loss:\s*(?P<train_loss>[0-9.eE+-]+).*?"
    r"val loss:\s*(?P<val_loss>[0-9.eE+-]+).*?val_acc:\s*(?P<val_acc>[0-9.eE+-]+).*?"
    r"val_balanced_acc:\s*(?P<val_ba>[0-9.eE+-]+).*?val_macro_f1:\s*(?P<val_f1>[0-9.eE+-]+)"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results_root / "analysis" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    curves = []
    for summary_path in sorted(args.results_root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        row = {"summary_path": str(summary_path), "run_dir": str(summary_path.parent)}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
        runs.append(row)
        log_path = summary_path.parent / "train_logs.txt"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = EPOCH_PATTERN.search(line)
                if match:
                    item = {"run_dir": str(summary_path.parent)}
                    item.update({key: float(value) for key, value in match.groupdict().items() if key != "epoch"})
                    item["epoch"] = int(match.group("epoch")) + 1
                    curves.append(item)
    write_csv(output / "classifier_runs.csv", runs)
    write_csv(output / "classifier_epoch_curves.csv", curves)
    best_rows = []
    for run_dir in sorted({row["run_dir"] for row in curves}):
        current = [row for row in curves if row["run_dir"] == run_dir]
        if not current:
            continue
        best = max(current, key=lambda row: row["val_ba"])
        final = max(current, key=lambda row: row["epoch"])
        best_rows.append(
            {
                "run_dir": run_dir,
                "best_epoch_by_balanced_accuracy": best["epoch"],
                "best_val_balanced_accuracy": best["val_ba"],
                "best_val_macro_f1": best["val_f1"],
                "final_epoch": final["epoch"],
                "final_val_balanced_accuracy": final["val_ba"],
                "overfit_drop_balanced_accuracy": best["val_ba"] - final["val_ba"],
            }
        )
    write_csv(output / "classifier_best_and_overfit.csv", best_rows)
    report = [
        "# 自动汇总报告",
        "",
        f"- 结果根目录：`{args.results_root.resolve()}`",
        f"- 找到分类运行：{len(runs)}",
        f"- 找到逐 epoch 记录：{len(curves)}",
        f"- 可计算最佳 epoch 的运行：{len(best_rows)}",
        "",
        "详细数据见 `classifier_runs.csv`、`classifier_epoch_curves.csv` 和 `classifier_best_and_overfit.csv`。",
    ]
    (output / "SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "epoch_records": len(curves), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
