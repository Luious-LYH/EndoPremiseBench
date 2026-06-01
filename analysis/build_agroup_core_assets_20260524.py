"""Build A-group paper assets from completed and current partial artifacts.

The compiled paper may show current partial runs as such, but it should never
silently promote expected placeholders to evidence. Draft-only expected values
are written to paper/draft_placeholders for planning.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
PLACEHOLDERS = PAPER / "draft_placeholders"
AGROUP = ROOT / "results" / "a_group_supplement_20260524"
DATE = "20260524"
MAIN_N = 6000

MODELS = {
    "qwen3_vl_8b": {
        "name": "Qwen3-VL-8B",
        "type": "open general",
        "status": "Full",
        "metric": AGROUP / "metrics" / "qwen3_vl_8b_premise_main6000_metrics.json",
        "scored": AGROUP / "scored" / "qwen3_vl_8b_premise_main6000_scored.jsonl",
    },
    "qwen25_vl_7b": {
        "name": "Qwen2.5-VL-7B",
        "type": "open general",
        "status": "Full",
        "metric": AGROUP / "metrics" / "qwen25_vl_7b_premise_main6000_metrics.json",
        "scored": AGROUP / "scored" / "qwen25_vl_7b_premise_main6000_scored.jsonl",
    },
    "internvl25_8b": {
        "name": "InternVL2.5-8B",
        "type": "open general",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "internvl25_8b_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "internvl25_8b_premise_main6000_scored_partial.jsonl",
    },
    "medgemma_4b": {
        "name": "MedGemma-4B",
        "type": "open medical",
        "status": "Full",
        "metric": AGROUP / "metrics" / "medgemma_4b_premise_main6000_metrics.json",
        "scored": AGROUP / "scored" / "medgemma_4b_premise_main6000_scored.jsonl",
    },
    "llava_med_v15": {
        "name": "LLaVA-Med-v1.5-Mistral-7B",
        "type": "open medical",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "llava_med_v15_mistral_7b_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "llava_med_v15_mistral_7b_premise_main6000_scored_partial.jsonl",
    },
    "simula_medgemma": {
        "name": "MedGemma-KvasirVQA-x1-ft",
        "type": "open endoscopy-ft",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "simula_medgemma_kvasir_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "simula_medgemma_kvasir_premise_main6000_scored_partial.jsonl",
    },
    "claude_opus_4_7": {
        "name": "Claude-Opus-4.7",
        "type": "closed API",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "claude_opus_4_7_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "claude_opus_4_7_premise_main6000_scored_partial.jsonl",
    },
    "gpt_5_5": {
        "name": "GPT-5.5",
        "type": "closed API",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "gpt_5_5_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "gpt_5_5_premise_main6000_scored_partial.jsonl",
    },
    "gemini_3_1": {
        "name": "Gemini-3.1-Pro-High",
        "type": "closed API",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "gemini_3_1_pro_high_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "gemini_3_1_pro_high_premise_main6000_scored_partial.jsonl",
    },
    "grok_4_20": {
        "name": "Grok-4.20-MA-xhigh",
        "type": "closed API",
        "status": "Partial",
        "metric": AGROUP / "analysis" / "current_partial_scoring_20260524" / "metrics" / "grok_4_20_multi_agent_xhigh_premise_main6000_metrics_partial.json",
        "scored": AGROUP / "analysis" / "current_partial_scoring_20260524" / "scored" / "grok_4_20_multi_agent_xhigh_premise_main6000_scored_partial.jsonl",
    },
    "lingshu_7b": {
        "name": "Lingshu-7B",
        "type": "open medical",
        "status": "Pending",
        "metric": None,
        "scored": None,
    },
    "simula_qwen25": {
        "name": "Qwen2.5-VL-KvasirVQA-x1-ft",
        "type": "open endoscopy-ft",
        "status": "Pending",
        "metric": None,
        "scored": None,
    },
}

PLACEHOLDER_ROWS = [
    ("Claude-Opus-4.7", "closed API", 6000, 56.0, 70.0, 30.0, 10.0, 62.2, "expected; replace after full scoring"),
    ("GPT-5.5", "closed API", 6000, 59.0, 74.0, 24.0, 8.0, 65.7, "expected; replace after full scoring"),
    ("Gemini-3.1-Pro-High", "closed API", 6000, 53.0, 65.0, 34.0, 12.0, 58.4, "expected; replace after full scoring"),
    ("Grok-4.20-MA-xhigh", "closed API", 6000, 50.0, 60.0, 39.0, 13.0, 54.5, "expected; replace after full scoring"),
    ("InternVL2.5-8B", "open general", 6000, 41.5, 33.0, 66.0, 14.0, 36.7, "expected; replace after full scoring"),
    ("LLaVA-Med-v1.5-Mistral-7B", "open medical", 6000, 35.0, 18.0, 81.0, 9.0, 23.7, "expected; replace after full scoring"),
    ("Lingshu-7B", "open medical", 6000, 37.0, 25.0, 73.0, 8.0, 29.8, "expected; replace after full scoring"),
    ("MedGemma-KvasirVQA-x1-ft", "open endoscopy-ft", 6000, 42.0, 21.0, 78.0, 6.0, 28.0, "expected; replace after full scoring"),
    ("Qwen2.5-VL-KvasirVQA-x1-ft", "open endoscopy-ft", 6000, 45.0, 32.0, 66.0, 7.0, 37.4, "expected; replace after full scoring"),
]


def pct(x: float) -> float:
    return round(float(x) * 100.0, 1)


def pct_raw(x: float) -> str:
    return f"{float(x):.1f}"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_table(path: Path, label: str, caption: str, colspec: str, header: list[str], rows: list[list[str]], star: bool = False) -> None:
    env = "table*" if star else "table"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", rf"\end{{{env}}}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_available(meta: dict) -> bool:
    path = meta.get("metric")
    return isinstance(path, Path) and path.exists()


def format_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return pct_raw(value)


def baseline_summary() -> dict[str, dict]:
    src = AGROUP / "analysis" / "trivial_baselines_20260524" / "summary.csv"
    if not src.exists():
        return {}
    with src.open("r", encoding="utf-8") as f:
        return {row["baseline"]: row for row in csv.DictReader(f)}


def is_primary_row(key: str, metrics: dict[str, dict]) -> bool:
    meta = MODELS[key]
    m = metrics.get(key)
    if m is None:
        return False
    return meta["status"] == "Full" and int(m.get("n", 0)) == MAIN_N


def build_main_results(metrics: dict[str, dict]) -> None:
    rows = []
    csv_rows = []
    for key in MODELS:
        meta = MODELS[key]
        if not is_primary_row(key, metrics):
            continue
        m = metrics.get(key)
        if m is None:
            row = {
                "model": meta["name"],
                "type": meta["type"],
                "status": meta["status"],
                "n": 0,
                "Acc_TP": None,
                "Acc_FP": None,
                "SR": None,
                "ORR": None,
                "PFR": None,
                "Ambig": None,
                "HPS": None,
            }
        else:
            row = {
                "model": meta["name"],
                "type": meta["type"],
                "status": meta["status"],
                "n": int(m["n"]),
                "Acc_TP": pct(m["Acc_TP"]),
                "Acc_FP": pct(m["Acc_FP"]),
                "SR": pct(m["SR"]),
                "ORR": pct(m["ORR"]),
                "PFR": pct(m["PFR"]),
                "Ambig": pct(m.get("ambiguous_rate", 0.0)),
                "HPS": pct(m["HPS"]),
            }
        csv_rows.append(row)
        rows.append(
            [
                row["model"],
                row["type"],
                row["status"],
                str(row["n"]) if row["n"] else "--",
                format_metric(row["Acc_TP"]),
                format_metric(row["Acc_FP"]),
                format_metric(row["SR"]),
                format_metric(row["ORR"]),
                format_metric(row["PFR"]),
                format_metric(row["Ambig"]),
                format_metric(row["HPS"]),
            ]
        )
    baselines = baseline_summary()
    reference_names = {
        "always_NA": "Always N/A",
        "never_NA_random": "Never N/A random concrete",
        "premise_oracle_random": "Premise oracle + random attribute",
    }
    for key in ["always_NA", "never_NA_random", "premise_oracle_random"]:
        if key not in baselines:
            continue
        b = baselines[key]
        row = {
            "model": reference_names[key],
            "type": "reference policy",
            "status": "manifest",
            "n": int(b["n"]),
            "Acc_TP": pct(b["Acc_TP"]),
            "Acc_FP": pct(b["Acc_FP"]),
            "SR": pct(b["SR"]),
            "ORR": pct(b["ORR"]),
            "PFR": pct(b["PFR"]),
            "Ambig": pct(b.get("ambiguous_rate", 0.0)),
            "HPS": pct(b["HPS"]),
        }
        csv_rows.append(row)
        rows.append(
            [
                row["model"],
                row["type"],
                row["status"],
                str(row["n"]),
                format_metric(row["Acc_TP"]),
                format_metric(row["Acc_FP"]),
                format_metric(row["SR"]),
                format_metric(row["ORR"]),
                format_metric(row["PFR"]),
                format_metric(row["Ambig"]),
                format_metric(row["HPS"]),
            ]
        )
    write_csv(
        TABLES / f"agroup_main_results_{DATE}.csv",
        csv_rows,
        ["model", "type", "status", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "Ambig", "HPS"],
    )
    write_table(
        TABLES / f"agroup_main_results_{DATE}.tex",
        "tab:main_results",
        "Primary results on the balanced evaluation split. Model rows are complete 6,000-item evaluations under the same prompt and parser. Reference-policy rows are deterministic policies computed from the manifest/options and are included to calibrate the utility--restraint trade-off. Values are percentages except $n$.",
        "lllrrrrrrrr",
        ["Model", "Family", "Coverage", "$n$", r"\accTP{}", r"\accFP{}", r"\sr{}", r"\orr{}", "PFR", "Ambig.", r"\hps{}"],
        rows,
        star=True,
    )


def build_current_status_table(metrics: dict[str, dict]) -> None:
    rows = []
    csv_rows = []
    for key, meta in MODELS.items():
        m = metrics.get(key)
        n = int(m["n"]) if m else 0
        coverage = 100.0 * n / 6000.0 if n else 0.0
        row = {
            "model": meta["name"],
            "family": meta["type"],
            "status": meta["status"],
            "n": n,
            "coverage": round(coverage, 1),
            "paper_use": "primary" if is_primary_row(key, metrics) else ("metadata only" if m else "not reported"),
        }
        csv_rows.append(row)
        rows.append(
            [
                row["model"],
                row["family"],
                row["status"],
                str(row["n"]) if row["n"] else "--",
                f"{row['coverage']:.1f}" if row["n"] else "--",
                row["paper_use"],
            ]
        )
    write_csv(
        TABLES / f"agroup_current_status_{DATE}.csv",
        csv_rows,
        ["model", "family", "status", "n", "coverage", "paper_use"],
    )
    write_table(
        TABLES / f"agroup_current_status_{DATE}.tex",
        "tab:agroup_status",
        "Model evaluation inventory. This table reports coverage only for incomplete or unavailable evaluations; their scores are intentionally omitted because they are not comparable to the complete primary evidence in \\cref{tab:main_results}.",
        "lllrrp{0.30\\linewidth}",
        ["Model", "Family", "Coverage", "$n$", "Cov.", "Paper use"],
        rows,
        star=True,
    )


def build_completed_results(metrics: dict[str, dict]) -> None:
    rows = []
    csv_rows = []
    for key in MODELS:
        meta = MODELS[key]
        if meta["status"] != "Full":
            continue
        m = metrics[key]
        row = {
            "model": meta["name"],
            "type": meta["type"],
            "n": int(m["n"]),
            "Acc_TP": pct(m["Acc_TP"]),
            "Acc_FP": pct(m["Acc_FP"]),
            "SR": pct(m["SR"]),
            "ORR": pct(m["ORR"]),
            "PFR": pct(m["PFR"]),
            "Ambig": pct(m.get("ambiguous_rate", 0.0)),
            "HPS": pct(m["HPS"]),
        }
        csv_rows.append(row)
        rows.append(
            [
                row["model"],
                row["type"],
                str(row["n"]),
                pct_raw(row["Acc_TP"]),
                pct_raw(row["Acc_FP"]),
                pct_raw(row["SR"]),
                pct_raw(row["ORR"]),
                pct_raw(row["PFR"]),
                pct_raw(row["Ambig"]),
                pct_raw(row["HPS"]),
            ]
        )
    write_csv(TABLES / f"agroup_completed_results_{DATE}.csv", csv_rows, list(csv_rows[0]))
    write_table(
        TABLES / f"agroup_completed_results_{DATE}.tex",
        "tab:completed_results",
        "Completed model-axis results on the 6,000-item source-balanced subset. Values are percentages except $n$. Only fully scored runs are reported as evidence.",
        "llrrrrrrrr",
        ["Model", "Type", "$n$", r"\accTP{}", r"\accFP{}", r"\sr{}", r"\orr{}", "PFR", "Ambig.", r"\hps{}"],
        rows,
        star=True,
    )


def build_axis_table(metrics: dict[str, dict]) -> None:
    groups: dict[str, list[tuple[str, dict, str]]] = {}
    for key, meta in MODELS.items():
        m = metrics.get(key)
        if m is None or not is_primary_row(key, metrics):
            continue
        groups.setdefault(meta["type"], []).append((meta["name"], m, meta["status"]))

    rows = []
    csv_rows = []
    for group in ["open general", "open medical", "open endoscopy-ft", "closed API"]:
        items = groups.get(group, [])
        if not items:
            continue
        hps_values = [pct(m["HPS"]) for _, m, _ in items]
        sr_values = [pct(m["SR"]) for _, m, _ in items]
        accfp_values = [pct(m["Acc_FP"]) for _, m, _ in items]
        best_name, best_m, best_status = max(items, key=lambda item: item[1]["HPS"])
        coverage = ", ".join(f"{name} ({status}, n={int(m['n'])})" for name, m, status in items)
        takeaway = {
            "open general": "complete rows remain below the desired frontier",
            "open medical": "single complete medical-model case remains exposure-prone",
            "open endoscopy-ft": "no complete current row available",
            "closed API": "no complete current row available",
        }[group]
        row = [
            group,
            str(len(items)),
            best_name,
            f"{pct(best_m['HPS']):.1f}",
            f"{min(hps_values):.1f}--{max(hps_values):.1f}",
            f"{min(accfp_values):.1f}--{max(accfp_values):.1f}",
            f"{min(sr_values):.1f}--{max(sr_values):.1f}",
            takeaway,
        ]
        rows.append(row)
        csv_rows.append(
            {
                "family": group,
                "observed_models": len(items),
                "best_model": best_name,
                "best_status": best_status,
                "best_HPS": f"{pct(best_m['HPS']):.1f}",
                "HPS_range": row[4],
                "Acc_FP_range": row[5],
                "SR_range": row[6],
                "takeaway": takeaway,
                "coverage": coverage,
            }
        )
    write_csv(
        TABLES / f"agroup_model_axis_deltas_{DATE}.csv",
        csv_rows,
        ["family", "observed_models", "best_model", "best_status", "best_HPS", "HPS_range", "Acc_FP_range", "SR_range", "takeaway", "coverage"],
    )
    write_table(
        TABLES / f"agroup_model_axis_deltas_{DATE}.tex",
        "tab:model_axis",
        "Model-axis summary restricted to complete 6,000-item A-group artifacts. Incomplete rows are intentionally excluded from this table and appear only in the coverage inventory.",
        "lllrrrrl",
        ["Family", "Obs.", "Best current model", r"Best \hps{}", r"\hps{} range", r"\accFP{} range", r"\sr{} range", "Diagnostic reading"],
        rows,
        star=True,
    )


def build_source_table() -> None:
    rows = []
    csv_rows = []
    for key in MODELS:
        name = MODELS[key]["name"]
        if MODELS[key]["status"] != "Full":
            continue
        scored = read_jsonl(MODELS[key]["scored"])
        for source in ["Kvasir-VQA", "Kvasir-VQA-x1"]:
            true_rows = [r for r in scored if r.get("premise_type") == "true" and r.get("source_dataset") == source]
            false_rows = [r for r in scored if r.get("premise_type") == "false" and r.get("source_dataset") == source]
            n_true = len(true_rows)
            n_false = len(false_rows)
            acc_tp = 100.0 * sum(1 for r in true_rows if r.get("is_correct")) / max(1, n_true)
            acc_fp = 100.0 * sum(1 for r in false_rows if r.get("is_correct")) / max(1, n_false)
            sr = 100.0 * sum(1 for r in false_rows if r.get("failure_type") == "Unsupported Attribute Exposure") / max(1, n_false)
            rows.append([name, source, str(n_true), pct_raw(acc_tp), str(n_false), pct_raw(acc_fp), pct_raw(sr)])
            csv_rows.append({"model": name, "source": source, "n_true": n_true, "Acc_TP": pct_raw(acc_tp), "n_false": n_false, "Acc_FP": pct_raw(acc_fp), "SR": pct_raw(sr)})
    write_csv(TABLES / f"paper_source_premise_false_{DATE}.csv", csv_rows, ["model", "source", "n_true", "Acc_TP", "n_false", "Acc_FP", "SR"])
    write_table(
        TABLES / f"paper_source_premise_false_{DATE}.tex",
        "tab:source_false",
        r"Kvasir-only premise diagnostics for completed runs. Reporting true- and false-premise rows within each Kvasir source makes the source--premise entanglement visible rather than hiding it in the aggregate.",
        "llrrrrr",
        ["Model", "Source", "$n_T$", r"\accTP{}", "$n_F$", r"\accFP{}", r"\sr{}"],
        rows,
        star=True,
    )


def build_attribute_table(metrics: dict[str, dict]) -> None:
    rows = []
    csv_rows = []
    attrs = ["color", "count", "location", "morphology_type", "removal_status"]
    for key in MODELS:
        name = MODELS[key]["name"]
        if MODELS[key]["status"] != "Full":
            continue
        scored = read_jsonl(MODELS[key]["scored"])
        for attr in attrs:
            true_rows = [r for r in scored if r.get("attribute_type") == attr and r.get("premise_type") == "true"]
            false_rows = [r for r in scored if r.get("attribute_type") == attr and r.get("premise_type") == "false"]
            n_true = len(true_rows)
            n_false = len(false_rows)
            acc_tp = 100.0 * sum(1 for r in true_rows if r.get("is_correct")) / max(1, n_true)
            orr = 100.0 * sum(1 for r in true_rows if r.get("failure_type") == "Over-Refusal") / max(1, n_true)
            acc_fp = 100.0 * sum(1 for r in false_rows if r.get("is_correct")) / max(1, n_false)
            sr = 100.0 * sum(1 for r in false_rows if r.get("failure_type") == "Unsupported Attribute Exposure") / max(1, n_false)
            rows.append([name, attr.replace("_", "/"), str(n_true), pct_raw(acc_tp), pct_raw(orr), str(n_false), pct_raw(acc_fp), pct_raw(sr)])
            csv_rows.append(
                {
                    "model": name,
                    "attribute": attr,
                    "n_true": n_true,
                    "Acc_TP": pct_raw(acc_tp),
                    "ORR": pct_raw(orr),
                    "n_false": n_false,
                    "Acc_FP": pct_raw(acc_fp),
                    "SR": pct_raw(sr),
                }
            )
    write_csv(TABLES / f"paper_attribute_breakdown_{DATE}.csv", csv_rows, ["model", "attribute", "n_true", "Acc_TP", "ORR", "n_false", "Acc_FP", "SR"])
    write_table(
        TABLES / f"paper_attribute_breakdown_{DATE}.tex",
        "tab:attribute_breakdown",
        "Attribute-level diagnostics for completed model-axis runs. True-premise and false-premise denominators are separated because exposure and over-refusal are defined on different row sets.",
        "llrrrrrr",
        ["Model", "Attribute", "$n_T$", r"\accTP{}", r"\orr{}", "$n_F$", r"\accFP{}", r"\sr{}"],
        rows,
        star=True,
    )


def build_source_premise_counts() -> None:
    scored = read_jsonl(next(iter(MODELS.values()))["scored"])
    sources = ["EndoBench", "Kvasir-VQA", "Kvasir-VQA-x1"]
    rows = []
    for source in sources:
        true_count = sum(1 for r in scored if r.get("source_dataset") == source and r.get("premise_type") == "true")
        false_count = sum(1 for r in scored if r.get("source_dataset") == source and r.get("premise_type") == "false")
        note = "valid source questions" if false_count == 0 else "paired valid and absence-derived rows"
        rows.append([source, str(true_count), str(false_count), note])
    write_table(
        TABLES / f"paper_source_premise_counts_{DATE}.tex",
        "tab:source_premise_counts",
        "Source and premise composition of the main subset. The table makes the source--premise design boundary explicit.",
        "lrrl",
        ["Source", "True premise", "False premise", "Construction role"],
        rows,
    )


def metric_from_rows(true_rows: list[dict], false_rows: list[dict]) -> dict[str, float]:
    acc_tp = sum(1 for r in true_rows if r.get("is_correct")) / max(1, len(true_rows))
    acc_fp = sum(1 for r in false_rows if r.get("is_correct")) / max(1, len(false_rows))
    sr = sum(1 for r in false_rows if r.get("parse_status") == "ok" and not r.get("is_correct")) / max(1, len(false_rows))
    orr = sum(1 for r in true_rows if r.get("failure_type") == "Over-Refusal") / max(1, len(true_rows))
    hps = 0.0 if acc_tp + acc_fp == 0 else 2 * acc_tp * acc_fp / (acc_tp + acc_fp)
    return {"Acc_TP": acc_tp, "Acc_FP": acc_fp, "SR": sr, "ORR": orr, "HPS": hps}


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def build_ci_table(metrics: dict[str, dict]) -> None:
    random.seed(20260524)
    rows = []
    csv_rows = []
    for key in MODELS:
        if MODELS[key]["status"] != "Full" or key not in metrics:
            continue
        scored = read_jsonl(MODELS[key]["scored"])
        true_rows = [r for r in scored if r.get("premise_type") == "true"]
        false_rows = [r for r in scored if r.get("premise_type") == "false"]
        point = metric_from_rows(true_rows, false_rows)
        samples = {k: [] for k in point}
        for _ in range(1000):
            bt = [true_rows[random.randrange(len(true_rows))] for _ in true_rows]
            bf = [false_rows[random.randrange(len(false_rows))] for _ in false_rows]
            m = metric_from_rows(bt, bf)
            for metric, value in m.items():
                samples[metric].append(value)
        rows.append(
            [
                MODELS[key]["name"],
                f'{pct(point["Acc_TP"]):.1f} [{pct(percentile(samples["Acc_TP"], 0.025)):.1f}, {pct(percentile(samples["Acc_TP"], 0.975)):.1f}]',
                f'{pct(point["Acc_FP"]):.1f} [{pct(percentile(samples["Acc_FP"], 0.025)):.1f}, {pct(percentile(samples["Acc_FP"], 0.975)):.1f}]',
                f'{pct(point["SR"]):.1f} [{pct(percentile(samples["SR"], 0.025)):.1f}, {pct(percentile(samples["SR"], 0.975)):.1f}]',
                f'{pct(point["HPS"]):.1f} [{pct(percentile(samples["HPS"], 0.025)):.1f}, {pct(percentile(samples["HPS"], 0.975)):.1f}]',
            ]
        )
        csv_row = {"model": MODELS[key]["name"]}
        for metric in ["Acc_TP", "Acc_FP", "SR", "ORR", "HPS"]:
            csv_row[metric] = point[metric]
            csv_row[f"{metric}_lo"] = percentile(samples[metric], 0.025)
            csv_row[f"{metric}_hi"] = percentile(samples[metric], 0.975)
        csv_rows.append(csv_row)
    write_csv(
        TABLES / f"agroup_bootstrap_ci_{DATE}.csv",
        csv_rows,
        [
            "model",
            "Acc_TP",
            "Acc_TP_lo",
            "Acc_TP_hi",
            "Acc_FP",
            "Acc_FP_lo",
            "Acc_FP_hi",
            "SR",
            "SR_lo",
            "SR_hi",
            "ORR",
            "ORR_lo",
            "ORR_hi",
            "HPS",
            "HPS_lo",
            "HPS_hi",
        ],
    )
    write_table(
        TABLES / f"agroup_bootstrap_ci_{DATE}.tex",
        "tab:bootstrap_ci",
        r"Bootstrap 95\% confidence intervals for completed main6000 runs. Values are percentages.",
        "lrrrr",
        ["Model", r"\accTP{}", r"\accFP{}", r"\sr{}", r"\hps{}"],
        rows,
        star=True,
    )


def build_figures(metrics: dict[str, dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    plotted = [(k, MODELS[k], metrics[k]) for k in MODELS if is_primary_row(k, metrics)]
    color_map = {
        "open general": "#2f6f9f",
        "open medical": "#c75643",
        "open endoscopy-ft": "#8d5a9e",
        "closed API": "#2f8f4e",
    }
    marker_map = {"Full": "o", "Partial": "s"}

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for key, meta, m in plotted:
        x = pct(m["Acc_TP"])
        y = pct(m["Acc_FP"])
        size = 70 + 5 * pct(m["HPS"])
        face = color_map[meta["type"]] if meta["status"] == "Full" else "white"
        ax.scatter(
            [x],
            [y],
            s=size,
            marker=marker_map.get(meta["status"], "o"),
            facecolor=face,
            edgecolor=color_map[meta["type"]],
            linewidth=1.4,
            zorder=3,
        )
        label = meta["name"].replace("-Mistral-7B", "").replace("-High", "")
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7.2)
    oracle_x = 33.9
    oracle_y = 100.0
    ax.scatter([oracle_x], [oracle_y], s=120, marker="D", facecolor="#f2c14e", edgecolor="#6b5600", linewidth=1.0, zorder=4)
    ax.annotate("premise oracle", (oracle_x, oracle_y), xytext=(6, -14), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Useful true-premise answering $\mathrm{Acc}_{TP}$ (%)")
    ax.set_ylabel(r"False-premise rejection $\mathrm{Acc}_{FP}$ (%)")
    ax.grid(True, linewidth=0.35, color="#d6d6d6")
    ax.set_title("Primary utility-restraint frontier remains below a simple premise oracle")
    handles = []
    for family, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=color, label=family, markersize=6))
    handles.append(plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#f2c14e", markeredgecolor="#6b5600", label="Premise oracle", markersize=6))
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_agroup_frontier_{DATE}.pdf")
    fig.savefig(FIGURES / f"fig_agroup_frontier_{DATE}.png", dpi=240)
    plt.close(fig)

    names = [meta["name"].replace("-Mistral-7B", "").replace("-High", "") for _, meta, _ in plotted]
    sr = [pct(m["SR"]) for _, _, m in plotted]
    colors = [color_map[meta["type"]] for _, meta, _ in plotted]
    hatches = ["//" if meta["status"] == "Partial" else "" for _, meta, _ in plotted]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    bars = ax.bar(range(len(names)), sr, color=colors, width=0.62, edgecolor="#222222", linewidth=0.3)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(50, linestyle="--", color="#555555", linewidth=0.9)
    ax.set_ylabel(r"Unsupported attribute exposure $\mathrm{SR}$ (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", linewidth=0.35, color="#d6d6d6")
    for bar, value in zip(bars, sr):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.6, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("False-premise exposure remains high in completed open-model runs")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_agroup_exposure_{DATE}.pdf")
    fig.savefig(FIGURES / f"fig_agroup_exposure_{DATE}.png", dpi=240)
    plt.close(fig)


def build_placeholder_package(metrics: dict[str, dict]) -> None:
    PLACEHOLDERS.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in MODELS:
        if key not in metrics:
            continue
        m = metrics[key]
        rows.append(
            {
                "model": MODELS[key]["name"],
                "type": MODELS[key]["type"],
                "n": int(m["n"]),
                "Acc_TP": pct(m["Acc_TP"]),
                "Acc_FP": pct(m["Acc_FP"]),
                "SR": pct(m["SR"]),
                "ORR": pct(m["ORR"]),
                "HPS": pct(m["HPS"]),
                "status": "real",
            }
        )
    for model, group, n, acc_tp, acc_fp, sr, orr, hps, note in PLACEHOLDER_ROWS:
        rows.append(
            {
                "model": model,
                "type": group,
                "n": n,
                "Acc_TP": acc_tp,
                "Acc_FP": acc_fp,
                "SR": sr,
                "ORR": orr,
                "HPS": hps,
                "status": note,
            }
        )
    fields = ["model", "type", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "HPS", "status"]
    write_csv(PLACEHOLDERS / f"agroup_expected_results_placeholder_{DATE}.csv", rows, fields)
    md_lines = [
        "# Draft-only A-group Expected Results Placeholder",
        "",
        "These values are narrative planning placeholders and are not included in the compiled paper as evidence.",
        "Replace every row marked `expected` with full raw/scored/metrics artifacts before submission.",
        "",
        "| Model | Type | n | Acc_TP | Acc_FP | SR | ORR | HPS | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['type']} | {row['n']} | {row['Acc_TP']} | {row['Acc_FP']} | {row['SR']} | {row['ORR']} | {row['HPS']} | {row['status']} |"
        )
    (PLACEHOLDERS / f"agroup_expected_results_placeholder_{DATE}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = {key: read_json(meta["metric"]) for key, meta in MODELS.items() if metric_available(meta)}
    build_main_results(metrics)
    build_current_status_table(metrics)
    build_completed_results(metrics)
    build_axis_table(metrics)
    build_source_table()
    build_attribute_table(metrics)
    build_source_premise_counts()
    build_ci_table(metrics)
    build_figures(metrics)
    build_placeholder_package(metrics)
    status = {
        "scored_models": [MODELS[k]["name"] for k in MODELS if k in metrics],
        "full_models": [MODELS[k]["name"] for k in MODELS if MODELS[k]["status"] == "Full" and k in metrics],
        "partial_models": [MODELS[k]["name"] for k in MODELS if MODELS[k]["status"] == "Partial" and k in metrics],
        "pending_models": [MODELS[k]["name"] for k in MODELS if MODELS[k]["status"] == "Pending"],
        "placeholder_models": [row[0] for row in PLACEHOLDER_ROWS],
        "failure_types": dict(Counter(row[-1] for row in PLACEHOLDER_ROWS)),
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
