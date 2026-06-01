#!/usr/bin/env python3
"""Summarize EndoPremiseBench main6000 metrics into reusable tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


MODEL_NAMES = {
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "qwen3vl4b": "Qwen3-VL-4B",
    "internvl25": "InternVL2.5-4B",
    "minicpmv4": "MiniCPM-V-4",
}

METRIC_FIELDS = ["n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "ambiguous_rate", "HPS"]


def model_name_from_path(path: Path) -> str:
    stem = path.name.replace("_premise_main6000_v2_metrics.json", "")
    for key, value in MODEL_NAMES.items():
        if stem.startswith(key):
            return value
    return stem


def load_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_premise_main6000_v2_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row: Dict[str, Any] = {"model": model_name_from_path(path), "metrics_path": str(path)}
        for field in METRIC_FIELDS:
            row[field] = metrics.get(field)
        true = (metrics.get("premise_breakdown") or {}).get("true") or {}
        false = (metrics.get("premise_breakdown") or {}).get("false") or {}
        row["true_n"] = true.get("n")
        row["false_n"] = false.get("n")
        row["true_accuracy"] = true.get("accuracy")
        row["false_accuracy"] = false.get("accuracy")
        rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", *METRIC_FIELDS, "true_n", "false_n", "true_accuracy", "false_accuracy", "metrics_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EndoPremiseBench Main Results v2",
        "",
        "| Model | n | Acc_TP | Acc_FP | SR | ORR | PFR | ambiguous | HPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    fmt(row.get("n")),
                    fmt(row.get("Acc_TP")),
                    fmt(row.get("Acc_FP")),
                    fmt(row.get("SR")),
                    fmt(row.get("ORR")),
                    fmt(row.get("PFR")),
                    fmt(row.get("ambiguous_rate")),
                    fmt(row.get("HPS")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- This table is generated from scored `*_premise_main6000_v2_metrics.json` files.",
            "- Do not promote these numbers to paper claims before experiment audit and result-to-claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--table-output", type=Path, default=Path("tables/main_results_v2.md"))
    parser.add_argument("--csv-output", type=Path, default=Path("tables/main_results_v2.csv"))
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    write_markdown(rows, args.table_output)
    write_csv(rows, args.csv_output)
    print(json.dumps({"models": len(rows), "table": str(args.table_output), "csv": str(args.csv_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

