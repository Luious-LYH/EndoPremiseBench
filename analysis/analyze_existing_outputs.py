#!/usr/bin/env python3
"""Analyze completed EndoPremiseBench outputs for low-cost ablation evidence."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


MODEL_LABELS = {
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "qwen3vl4b": "Qwen3-VL-4B",
    "internvl25": "InternVL2.5-4B",
    "minicpmv4": "MiniCPM-V-4",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def model_label(path: Path) -> str:
    name = path.name
    for key, label in MODEL_LABELS.items():
        if name.startswith(key):
            return label
    return name.replace("_premise_main6000_v2_scored.jsonl", "")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metric_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    total = len(rows)
    true_rows = [r for r in rows if r.get("premise_type") == "true"]
    false_rows = [r for r in rows if r.get("premise_type") == "false"]
    acc_tp = safe_div(sum(bool(r.get("is_correct")) for r in true_rows), len(true_rows))
    acc_fp = safe_div(sum(bool(r.get("is_correct")) for r in false_rows), len(false_rows))
    pfr = safe_div(sum(r.get("parse_status") == "failure" for r in rows), total)
    ambiguous = safe_div(sum(r.get("parse_status") == "ambiguous" for r in rows), total)
    sr = safe_div(sum(r.get("failure_type") == "Unsupported Attribute Exposure" for r in false_rows), len(false_rows))
    orr = safe_div(sum(r.get("failure_type") == "Over-Refusal" for r in true_rows), len(true_rows))
    hps = (2 * acc_tp * acc_fp / (acc_tp + acc_fp)) if (acc_tp + acc_fp) else 0.0
    return {
        "n": float(total),
        "true_n": float(len(true_rows)),
        "false_n": float(len(false_rows)),
        "Acc_TP": acc_tp,
        "Acc_FP": acc_fp,
        "SR": sr,
        "ORR": orr,
        "PFR": pfr,
        "ambiguous_rate": ambiguous,
        "HPS": hps,
    }


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_ci(rows: Sequence[Dict[str, Any]], iterations: int, seed: int) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    metrics = ["Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"]
    draws: Dict[str, List[float]] = {metric: [] for metric in metrics}
    n = len(rows)
    for _ in range(iterations):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        measured = metric_row(sample)
        for metric in metrics:
            draws[metric].append(measured[metric])
    return {
        metric: {
            "lo": percentile(values, 0.025),
            "hi": percentile(values, 0.975),
        }
        for metric, values in draws.items()
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], fields: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---:" if field not in {"model", "group", "source", "attribute", "na_position"} else "---" for field in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--audit-sample-output", type=Path, default=Path("tables/manual_audit_false_premise_200.csv"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    scored_paths = sorted(args.results_dir.glob("*_premise_main6000_v2_scored.jsonl"))
    if not scored_paths:
        raise FileNotFoundError(f"No scored main6000 files found in {args.results_dir}")

    all_by_model = [(model_label(path), read_jsonl(path)) for path in scored_paths]

    ci_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    na_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []

    for model, rows in all_by_model:
        base = metric_row(rows)
        cis = bootstrap_ci(rows, args.bootstrap_iterations, args.seed)
        ci_row: Dict[str, Any] = {"model": model, "n": int(base["n"])}
        for metric in ["Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"]:
            ci_row[metric] = base[metric]
            ci_row[f"{metric}_lo"] = cis[metric]["lo"]
            ci_row[f"{metric}_hi"] = cis[metric]["hi"]
        ci_rows.append(ci_row)

        for group, subset in [
            ("all", rows),
            ("Kvasir-only", [r for r in rows if str(r.get("source_dataset", "")).startswith("Kvasir")]),
            ("Kvasir-VQA", [r for r in rows if r.get("source_dataset") == "Kvasir-VQA"]),
            ("Kvasir-VQA-x1", [r for r in rows if r.get("source_dataset") == "Kvasir-VQA-x1"]),
            ("EndoBench-true-control", [r for r in rows if r.get("source_dataset") == "EndoBench"]),
        ]:
            measured = metric_row(subset)
            group_rows.append({"model": model, "group": group, **measured})

        false_rows = [r for r in rows if r.get("premise_type") == "false"]
        for pos in ["A", "B", "C", "D"]:
            subset = [r for r in false_rows if r.get("answer") == pos]
            measured = metric_row(subset)
            na_rows.append(
                {
                    "model": model,
                    "na_position": pos,
                    "n": len(subset),
                    "Acc_FP": measured["Acc_FP"],
                    "SR": measured["SR"],
                    "PFR": measured["PFR"],
                    "ambiguous_rate": measured["ambiguous_rate"],
                }
            )

        for source in sorted({str(r.get("source_dataset", "")) for r in rows}):
            for premise in ["true", "false"]:
                subset = [r for r in rows if r.get("source_dataset") == source and r.get("premise_type") == premise]
                if subset:
                    measured = metric_row(subset)
                    source_rows.append({"model": model, "source": source, "group": premise, **measured})

    ci_fields = ["model", "n", "Acc_TP", "Acc_TP_lo", "Acc_TP_hi", "Acc_FP", "Acc_FP_lo", "Acc_FP_hi", "SR", "SR_lo", "SR_hi", "ORR", "ORR_lo", "ORR_hi", "PFR", "PFR_lo", "PFR_hi", "HPS", "HPS_lo", "HPS_hi"]
    group_fields = ["model", "group", "n", "true_n", "false_n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"]
    na_fields = ["model", "na_position", "n", "Acc_FP", "SR", "PFR", "ambiguous_rate"]
    source_fields = ["model", "source", "group", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"]

    write_csv(ci_rows, args.tables_dir / "main_results_bootstrap_ci_v2.csv", ci_fields)
    write_csv(group_rows, args.tables_dir / "main_results_source_matched_v2.csv", group_fields)
    write_csv(na_rows, args.tables_dir / "main_results_na_position_v2.csv", na_fields)
    write_csv(source_rows, args.tables_dir / "main_results_source_premise_v2.csv", source_fields)

    reference_rows = read_jsonl(args.results_dir / "premise_balanced_main_v2.jsonl")
    false_reference = [r for r in reference_rows if r.get("premise_type") == "false"]
    rng = random.Random(args.seed)
    rng.shuffle(false_reference)
    audit_rows = false_reference[:200]
    audit_fields = ["id", "source_dataset", "source_id", "image_path", "target_entity", "attribute_type", "question", "options", "answer", "label_evidence", "construction_rule", "confidence"]
    args.audit_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_sample_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        for row in audit_rows:
            item = dict(row)
            item["options"] = json.dumps(item.get("options", {}), ensure_ascii=False)
            writer.writerow({field: item.get(field) for field in audit_fields})

    report = [
        "# EndoPremiseBench Existing-Output Ablation Analysis v2",
        "",
        "This report uses existing scored `main6000` outputs only. It adds low-cost evidence for confidence intervals, N/A-position sensitivity, and source-matched diagnostics.",
        "",
        "## Bootstrap 95% Intervals",
        "",
        *markdown_table(ci_rows, ["model", "Acc_TP", "Acc_TP_lo", "Acc_TP_hi", "Acc_FP", "Acc_FP_lo", "Acc_FP_hi", "SR", "SR_lo", "SR_hi", "ORR", "ORR_lo", "ORR_hi", "HPS", "HPS_lo", "HPS_hi"]),
        "",
        "## Kvasir-Only / Source-Matched Diagnostics",
        "",
        *markdown_table(group_rows, group_fields),
        "",
        "## False-Premise N/A Position Sensitivity",
        "",
        *markdown_table(na_rows, na_fields),
        "",
        "## Manual Audit Sample",
        "",
        f"A 200-row false-premise audit sample was written to `{args.audit_sample_output}`.",
        "",
        "Interpretation guard: these analyses do not replace question-only or wording-control inference ablations. They are existing-output sensitivity checks.",
    ]
    (args.tables_dir / "existing_output_ablation_analysis_v2.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(all_by_model), "audit_rows": len(audit_rows), "report": str(args.tables_dir / "existing_output_ablation_analysis_v2.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
