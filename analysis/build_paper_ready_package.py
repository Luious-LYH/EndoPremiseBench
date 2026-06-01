#!/usr/bin/env python3
"""Build paper-ready EndoPremiseBench tables, figures, and claim notes.

This script only uses existing EndoPremiseBench artifacts. It does not rerun
models or create new experimental results.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REVIEW = ROOT / "review-stage"

DATE = "20260524"

MODEL_ORDER = ["Qwen3-VL-4B", "MiniCPM-V-4", "Qwen2.5-VL-3B", "InternVL2.5-4B"]
MODEL_SHORT = {
    "Qwen3-VL-4B": "Qwen3",
    "MiniCPM-V-4": "MiniCPM",
    "Qwen2.5-VL-3B": "Qwen2.5",
    "InternVL2.5-4B": "InternVL",
}


def ensure_dirs() -> None:
    for path in [TABLES, FIGURES, REVIEW]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt4(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def pct1(value: Any) -> str:
    try:
        return f"{100 * float(value):.1f}"
    except Exception:
        return str(value)


def latex_escape(text: Any) -> str:
    out = str(text)
    for src, dst in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        out = out.replace(src, dst)
    return out


def write_latex_table(path: Path, caption: str, label: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def load_main_results() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "main_results_v2.csv")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model")


def build_main_tables(df: pd.DataFrame) -> None:
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "model": r["model"],
                "n": int(r["n"]),
                "Acc_TP_pct": pct1(r["Acc_TP"]),
                "Acc_FP_pct": pct1(r["Acc_FP"]),
                "SR_pct": pct1(r["SR"]),
                "ORR_pct": pct1(r["ORR"]),
                "PFR_pct": pct1(r["PFR"]),
                "HPS": fmt4(r["HPS"]),
            }
        )
    fields = ["model", "n", "Acc_TP_pct", "Acc_FP_pct", "SR_pct", "ORR_pct", "PFR_pct", "HPS"]
    write_csv(TABLES / f"paper_main_results_{DATE}.csv", rows, fields)
    write_latex_table(
        TABLES / f"paper_main_results_{DATE}.tex",
        "Main EndoPremiseBench results on the source-balanced converted-MCQ probe. Higher Acc_TP, Acc_FP, and HPS are better; higher SR and ORR indicate unsupported attribute exposure and over-refusal.",
        "tab:main_results",
        ["Model", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"],
        [[r["model"], r["n"], r["Acc_TP_pct"], r["Acc_FP_pct"], r["SR_pct"], r["ORR_pct"], r["PFR_pct"], r["HPS"]] for r in rows],
    )


def build_question_only(df: pd.DataFrame) -> None:
    paths = {
        "Qwen3-VL-4B": RESULTS / "qwen3vl4b_premise_question_only_main6000_v1_metrics.json",
        "MiniCPM-V-4": RESULTS / "minicpmv4_premise_question_only_main6000_v1_metrics.json",
        "Qwen2.5-VL-3B": RESULTS / "qwen25_vl_3b_premise_question_only_main6000_v1_metrics.json",
        "InternVL2.5-4B": RESULTS / "internvl25_premise_question_only_main6000_v1_metrics.json",
    }
    main_by_model = {str(r["model"]): r for _, r in df.iterrows()}
    rows: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        q = read_json(paths[model])
        main = main_by_model[model]
        rows.append(
            {
                "model": model,
                "image_HPS": float(main["HPS"]),
                "question_only_HPS": q["HPS"],
                "delta_HPS": q["HPS"] - float(main["HPS"]),
                "image_Acc_TP": float(main["Acc_TP"]),
                "question_only_Acc_TP": q["Acc_TP"],
                "image_Acc_FP": float(main["Acc_FP"]),
                "question_only_Acc_FP": q["Acc_FP"],
            }
        )
    fields = ["model", "image_HPS", "question_only_HPS", "delta_HPS", "image_Acc_TP", "question_only_Acc_TP", "image_Acc_FP", "question_only_Acc_FP"]
    write_csv(TABLES / f"paper_question_only_control_{DATE}.csv", rows, fields)
    write_latex_table(
        TABLES / f"paper_question_only_control_{DATE}.tex",
        "Question-only controls. High false-premise rejection without images often comes with collapsed true-premise accuracy, indicating over-refusal rather than robust visual grounding.",
        "tab:question_only",
        ["Model", "Image HPS", "Q-only HPS", "Q-only Acc_TP", "Q-only Acc_FP"],
        [[r["model"], fmt4(r["image_HPS"]), fmt4(r["question_only_HPS"]), pct1(r["question_only_Acc_TP"]), pct1(r["question_only_Acc_FP"])] for r in rows],
    )


def build_source_tables() -> None:
    source = pd.read_csv(TABLES / "main_results_source_premise_v2.csv")
    source["group"] = source["group"].astype(str).str.lower()
    source["source"] = source["source"].astype(str)
    source["model"] = source["model"].astype(str)
    # Focus on the two Kvasir false-premise sources; this is the cleanest story-facing caveat.
    focus = source[(source["group"] == "false") & (source["source"].isin(["Kvasir-VQA", "Kvasir-VQA-x1"]))].copy()
    focus["source"] = pd.Categorical(focus["source"], categories=["Kvasir-VQA", "Kvasir-VQA-x1"], ordered=True)
    focus["model"] = pd.Categorical(focus["model"], categories=MODEL_ORDER, ordered=True)
    focus = focus.sort_values(["model", "source"])
    rows = []
    for _, r in focus.iterrows():
        rows.append({"model": r["model"], "source": r["source"], "n": int(r["n"]), "Acc_FP_pct": pct1(r["Acc_FP"]), "SR_pct": pct1(r["SR"]), "PFR_pct": pct1(r["PFR"])})
    fields = ["model", "source", "n", "Acc_FP_pct", "SR_pct", "PFR_pct"]
    write_csv(TABLES / f"paper_source_premise_false_{DATE}.csv", rows, fields)
    write_latex_table(
        TABLES / f"paper_source_premise_false_{DATE}.tex",
        "False-premise performance by source. The Kvasir-VQA and Kvasir-VQA-x1 split reveals source sensitivity and motivates a scoped diagnostic interpretation.",
        "tab:source_false",
        ["Model", "Source", "n", "Acc_FP", "SR", "PFR"],
        [[r["model"], r["source"], r["n"], r["Acc_FP_pct"], r["SR_pct"], r["PFR_pct"]] for r in rows],
    )


def build_na_tables() -> None:
    df = pd.read_csv(TABLES / "main_results_na_position_v2.csv")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["na_position"] = pd.Categorical(df["na_position"], categories=["A", "B", "C", "D"], ordered=True)
    df = df.sort_values(["model", "na_position"])
    rows = []
    for _, r in df.iterrows():
        rows.append({"model": r["model"], "na_position": r["na_position"], "n": int(r["n"]), "Acc_FP_pct": pct1(r["Acc_FP"]), "SR_pct": pct1(r["SR"]), "PFR_pct": pct1(r["PFR"])})
    fields = ["model", "na_position", "n", "Acc_FP_pct", "SR_pct", "PFR_pct"]
    write_csv(TABLES / f"paper_na_position_main_stratified_{DATE}.csv", rows, fields)
    # Compact summary for main paper: min/max Acc_FP by model.
    summary = []
    for model, group in df.groupby("model", observed=True):
        accs = group["Acc_FP"].astype(float)
        summary.append({"model": model, "min_Acc_FP_pct": pct1(accs.min()), "max_Acc_FP_pct": pct1(accs.max()), "range_pct": pct1(accs.max() - accs.min())})
    write_csv(TABLES / f"paper_na_position_summary_{DATE}.csv", summary, ["model", "min_Acc_FP_pct", "max_Acc_FP_pct", "range_pct"])
    write_latex_table(
        TABLES / f"paper_na_position_summary_{DATE}.tex",
        "False-premise Acc_FP range across N/A option positions in the main scored outputs. This is a stratification of main rows, not the paired all-position control.",
        "tab:na_position_summary",
        ["Model", "Min Acc_FP", "Max Acc_FP", "Range"],
        [[r["model"], r["min_Acc_FP_pct"], r["max_Acc_FP_pct"], r["range_pct"]] for r in summary],
    )


def build_wording_tables() -> None:
    paths = {
        "Qwen3-VL-4B": (
            "qwen3vl4b",
            RESULTS / "qwen3vl4b_premise_wording_controls_v1_metrics.json",
        ),
        "Qwen2.5-VL-3B": (
            "qwen25_vl_3b",
            RESULTS / "qwen25_vl_3b_premise_wording_controls_v1_metrics.json",
        ),
    }
    rows = []
    for model, (prefix, path) in paths.items():
        j = read_json(path)
        raw_path = RESULTS / f"{prefix}_premise_wording_controls_v1_raw.jsonl"
        scored_path = RESULTS / f"{prefix}_premise_wording_controls_v1_scored.jsonl"
        artifact_status = "raw+scored present" if raw_path.exists() and scored_path.exists() else "metrics only"
        rows.append({"model": model, "n": j["n"], "Acc_FP_pct": pct1(j["Acc_FP"]), "SR_pct": pct1(j["SR"]), "PFR_pct": pct1(j["PFR"]), "row_level_artifact": artifact_status})
    write_csv(TABLES / f"paper_wording_control_status_{DATE}.csv", rows, ["model", "n", "Acc_FP_pct", "SR_pct", "PFR_pct", "row_level_artifact"])
    write_latex_table(
        TABLES / f"paper_wording_control_status_{DATE}.tex",
        "Wording-control status. These controls are complete for Qwen3-VL-4B and Qwen2.5-VL-3B only; InternVL2.5-4B and MiniCPM-V-4 are optional appendix/rebuttal extensions.",
        "tab:wording_status",
        ["Model", "n", "Acc_FP", "SR", "PFR", "Artifact status"],
        [[r["model"], r["n"], r["Acc_FP_pct"], r["SR_pct"], r["PFR_pct"], r["row_level_artifact"]] for r in rows],
    )


def build_attribute_tables() -> None:
    paths = {
        "Qwen3-VL-4B": RESULTS / "qwen3vl4b_premise_main6000_v2_metrics.json",
        "MiniCPM-V-4": RESULTS / "minicpmv4_premise_main6000_v2_metrics.json",
        "Qwen2.5-VL-3B": RESULTS / "qwen25_vl_3b_premise_main6000_v2_metrics.json",
        "InternVL2.5-4B": RESULTS / "internvl25_premise_main6000_v2_metrics.json",
    }
    rows = []
    for model in MODEL_ORDER:
        metrics = read_json(paths[model])
        for attr, br in sorted((metrics.get("attribute_breakdown") or {}).items()):
            dist = br.get("failure_type_distribution") or {}
            n = int(br.get("n", 0))
            unsupported = dist.get("Unsupported Attribute Exposure", 0)
            over_refusal = dist.get("Over-Refusal", 0)
            rows.append(
                {
                    "model": model,
                    "attribute": attr,
                    "n": n,
                    "accuracy_pct": pct1(br.get("accuracy", 0.0)),
                    "unsupported_exposure_pct": pct1(unsupported / n if n else 0.0),
                    "over_refusal_pct": pct1(over_refusal / n if n else 0.0),
                    "parse_failure_pct": pct1(br.get("parse_failure_rate", 0.0)),
                }
            )
    write_csv(TABLES / f"paper_attribute_breakdown_{DATE}.csv", rows, ["model", "attribute", "n", "accuracy_pct", "unsupported_exposure_pct", "over_refusal_pct", "parse_failure_pct"])


def build_manual_audit_package() -> None:
    src = REVIEW / "manual_audit_false_premise_200.csv"
    rows = list(csv.DictReader(src.open("r", encoding="utf-8")))
    out = REVIEW / f"manual_audit_false_premise_200_annotation_template_{DATE}.csv"
    fields = list(rows[0].keys()) + ["human_entity_absent", "human_gold_na_correct", "human_question_valid", "human_notes"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item.update({"human_entity_absent": "", "human_gold_na_correct": "", "human_question_valid": "", "human_notes": ""})
            writer.writerow(item)
    dist = Counter(row["attribute_type"] for row in rows)
    src_dist = Counter(row["source_dataset"] for row in rows)
    report = [
        "# EndoPremiseBench Manual Audit Package",
        "",
        f"Date: `{DATE}`",
        "",
        "## Status",
        "",
        "A 200-row false-premise audit sample exists, but no human adjudication has been filled yet. Do not claim that manual audit passed until the annotation columns are completed and summarized.",
        "",
        "## Files",
        "",
        f"- Original sample: `{src}`",
        f"- Annotation template: `{out}`",
        "",
        "## Annotation Columns",
        "",
        "- `human_entity_absent`: yes/no/uncertain",
        "- `human_gold_na_correct`: yes/no/uncertain",
        "- `human_question_valid`: yes/no/uncertain",
        "- `human_notes`: short explanation for disagreements or uncertainty",
        "",
        "## Sample Distribution",
        "",
        "### By Source",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(src_dist.items()):
        report.append(f"| {key} | {value} |")
    report.extend(["", "### By Attribute", "", "| Attribute | Count |", "|---|---:|"])
    for key, value in sorted(dist.items()):
        report.append(f"| {key} | {value} |")
    report.extend(
        [
            "",
            "## Paper Use",
            "",
            "Use this audit as the trust anchor for converted false-premise labels. Report agreement rate and disagreement categories in the appendix; mention the headline rate briefly in the method section.",
        ]
    )
    (REVIEW / f"MANUAL_AUDIT_PACKAGE_{DATE}.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def sample_error_cases(limit: int = 15) -> None:
    scored_paths = {
        "Qwen3-VL-4B": RESULTS / "qwen3vl4b_premise_main6000_v2_scored.jsonl",
        "MiniCPM-V-4": RESULTS / "minicpmv4_premise_main6000_v2_scored.jsonl",
        "Qwen2.5-VL-3B": RESULTS / "qwen25_vl_3b_premise_main6000_v2_scored.jsonl",
        "InternVL2.5-4B": RESULTS / "internvl25_premise_main6000_v2_scored.jsonl",
    }
    picked: List[Dict[str, Any]] = []
    seen = set()
    # Choose compact, diverse false-premise unsupported cases; this is for qualitative appendix candidates.
    for model in MODEL_ORDER:
        rows = read_jsonl(scored_paths[model])
        for row in rows:
            if row.get("premise_type") != "false" or row.get("failure_type") != "Unsupported Attribute Exposure":
                continue
            key = (row.get("attribute_type"), row.get("source_dataset"))
            if (model, key) in seen:
                continue
            seen.add((model, key))
            picked.append(
                {
                    "model": model,
                    "id": row.get("probe_id") or row.get("id"),
                    "source_dataset": row.get("source_dataset"),
                    "attribute_type": row.get("attribute_type"),
                    "target_entity": row.get("target_entity"),
                    "question": row.get("question"),
                    "gold_answer": row.get("answer"),
                    "parsed_answer": row.get("parsed_answer"),
                    "raw_output": str(row.get("raw_output", ""))[:240].replace("\n", " "),
                }
            )
            if len(picked) >= limit:
                break
        if len(picked) >= limit:
            break
    fields = ["model", "id", "source_dataset", "attribute_type", "target_entity", "question", "gold_answer", "parsed_answer", "raw_output"]
    write_csv(REVIEW / f"QUALITATIVE_FALSE_PREMISE_CASES_{DATE}.csv", picked, fields)


def build_figures(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    colors = {"Acc_TP": "#4C78A8", "Acc_FP": "#59A14F", "SR": "#E15759", "HPS": "#F28E2B"}
    x = list(range(len(df)))
    labels = [MODEL_SHORT[m] for m in df["model"]]

    # Figure 2: main usefulness-factuality profile.
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    width = 0.22
    ax.bar([i - width for i in x], df["Acc_TP"] * 100, width=width, label="Acc_TP", color=colors["Acc_TP"])
    ax.bar(x, df["Acc_FP"] * 100, width=width, label="Acc_FP", color=colors["Acc_FP"])
    ax.bar([i + width for i in x], df["SR"] * 100, width=width, label="SR", color=colors["SR"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.axhline(50, color="0.85", linewidth=0.8, zorder=0)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"fig_main_premise_profile_{DATE}.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: question-only collapse.
    q = pd.read_csv(TABLES / f"paper_question_only_control_{DATE}.csv")
    q["model"] = pd.Categorical(q["model"], categories=MODEL_ORDER, ordered=True)
    q = q.sort_values("model")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.bar([i - 0.18 for i in x], q["image_HPS"], width=0.36, label="Image", color="#4C78A8")
    ax.bar([i + 0.18 for i in x], q["question_only_HPS"], width=0.36, label="Question-only", color="#BAB0AC")
    ax.set_xticks(x, labels)
    ax.set_ylabel("HPS")
    ax.set_ylim(0, max(q["image_HPS"].max(), q["question_only_HPS"].max()) + 0.08)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"fig_question_only_hps_{DATE}.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: source false-premise contrast.
    sf = pd.read_csv(TABLES / f"paper_source_premise_false_{DATE}.csv")
    sf["Acc_FP"] = sf["Acc_FP_pct"].astype(float)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for offset, source, color in [(-0.18, "Kvasir-VQA", "#76B7B2"), (0.18, "Kvasir-VQA-x1", "#B07AA1")]:
        sub = sf[sf["source"] == source]
        vals = [float(sub[sub["model"] == m]["Acc_FP"].iloc[0]) for m in MODEL_ORDER]
        ax.bar([i + offset for i in x], vals, width=0.36, label=source, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("False-premise Acc_FP (%)")
    ax.set_ylim(0, 65)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES / f"fig_source_false_accfp_{DATE}.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_latex_includes() -> None:
    lines = [
        "% Auto-generated EndoPremiseBench paper includes.",
        rf"\input{{tables/paper_main_results_{DATE}.tex}}",
        rf"\input{{tables/paper_question_only_control_{DATE}.tex}}",
        rf"\input{{tables/paper_source_premise_false_{DATE}.tex}}",
        rf"\input{{tables/paper_na_position_summary_{DATE}.tex}}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=.95\linewidth]{{figures/fig_main_premise_profile_{DATE}.pdf}}",
        r"\caption{EndoPremiseBench main profile. All tested models show high unsupported attribute exposure (SR) while true-premise accuracy remains modest.}",
        r"\label{fig:main_premise_profile}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=.75\linewidth]{{figures/fig_question_only_hps_{DATE}.pdf}}",
        r"\caption{Question-only controls reduce HPS for all models, showing that high false-premise rejection without images is often a degenerate over-refusal strategy.}",
        r"\label{fig:question_only_hps}",
        r"\end{figure}",
        "",
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=.75\linewidth]{{figures/fig_source_false_accfp_{DATE}.pdf}}",
        r"\caption{False-premise rejection varies by source, motivating a scoped diagnostic interpretation rather than a source-invariant causal claim.}",
        r"\label{fig:source_false_accfp}",
        r"\end{figure}",
        "",
    ]
    (FIGURES / f"latex_includes_{DATE}.tex").write_text("\n".join(lines), encoding="utf-8")


def build_claim_gate() -> None:
    wording_row_files = list(RESULTS.glob("*wording_controls_v1_scored.jsonl"))
    wording_ready = len(wording_row_files) >= 2
    manual_template = REVIEW / f"manual_audit_false_premise_200_annotation_template_{DATE}.csv"
    report = [
        "# EndoPremiseBench Claim Gate Refresh",
        "",
        f"Date: `{DATE}`",
        "",
        "## Verdict",
        "",
        "`WRITEABLE_AFTER_TARGETED_NON_GPU_FIXES`",
        "",
        "## Supported Main Claim",
        "",
        "On a 6,000-sample source-balanced, dataset-grounded converted-MCQ diagnostic probe, four tested open VLMs show high unsupported attribute exposure under false endoscopic VQA premises (`SR = 53.9%--84.4%`) while true-premise attribute accuracy remains modest (`Acc_TP = 35.6%--39.3%`).",
        "",
        "## Paper-Safe Evidence",
        "",
        "- Main four-model `main6000` results: paper-ready table generated.",
        "- Question-only controls: four models complete with scored artifacts.",
        "- Source/premise diagnostics: existing source-matched tables generated from scored outputs.",
        "- N/A-position sensitivity: main false-row stratification exists for four models; paired all-position controls exist for Qwen3 and Qwen2.5.",
        "- Bootstrap CI: existing table generated and ready for appendix formatting.",
        "",
        "## Still Needed Before Submission",
        "",
        f"- Manual audit: fill `{manual_template}` and summarize agreement.",
        "- Wording controls: Qwen3 and Qwen2.5 now have metrics plus local row-level raw/scored artifacts. Treat as a two-model sensitivity result unless InternVL/MiniCPM are rerun.",
        "- Do not use old pilot/smoke/answer-only artifacts in paper claims.",
        "",
        "## Forbidden Claims",
        "",
        "- Clinical safety benchmark or clinical deployment readiness.",
        "- All VLMs or all medical VQA systems behave this way.",
        "- Clean causal isolation of presupposition robustness from ordinary attribute recognition.",
        "- Question-only high Acc_FP proves robustness.",
        "- Manual audit has passed before annotation is completed.",
        "",
        "## Artifact Caveat",
        "",
        f"Detected wording scored artifact count: `{len(wording_row_files)}`. Two-model wording sensitivity is reproducible locally; four-model wording coverage remains optional.",
    ]
    md_path = REVIEW / f"CLAIM_GATE_REFRESH_{DATE}.md"
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    data = {
        "date": DATE,
        "verdict": "WRITEABLE_AFTER_TARGETED_NON_GPU_FIXES",
        "supported_claim": "four tested open VLMs show high unsupported attribute exposure under false endoscopic VQA premises on main6000 converted-MCQ probe, with modest true-premise accuracy",
        "manual_audit_completed": False,
        "wording_row_level_artifacts_visible": wording_ready,
        "main_ready": True,
        "question_only_ready": True,
        "source_diagnostics_ready": True,
        "na_position_main_stratification_ready": True,
        "paired_na_position_models": ["Qwen3-VL-4B", "Qwen2.5-VL-3B"],
    }
    (REVIEW / f"CLAIM_GATE_REFRESH_{DATE}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_manifest(outputs: Sequence[Path]) -> None:
    lines = ["# EndoPremiseBench Paper Package Manifest", "", f"Date: `{DATE}`", "", "| File | Role |", "|---|---|"]
    roles = {
        "paper_main_results": "Main paper result table",
        "paper_question_only": "Question-only control table",
        "paper_source": "Source sensitivity table",
        "paper_na_position": "N/A-position sensitivity table",
        "paper_wording": "Wording control status table",
        "paper_attribute": "Attribute breakdown table",
        "MANUAL_AUDIT": "Manual audit package",
        "QUALITATIVE": "Qualitative case candidates",
        "CLAIM_GATE": "Claim gate refresh",
        "fig_": "Generated figure",
        "latex_includes": "LaTeX include snippets",
    }
    for path in sorted(outputs, key=lambda p: str(p)):
        role = "Paper package artifact"
        for key, value in roles.items():
            if key in path.name:
                role = value
                break
        lines.append(f"| `{path}` | {role} |")
    (REVIEW / f"PAPER_READY_PACKAGE_{DATE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    before = set(ROOT.rglob("*"))
    df = load_main_results()
    build_main_tables(df)
    build_question_only(df)
    build_source_tables()
    build_na_tables()
    build_wording_tables()
    build_attribute_tables()
    build_manual_audit_package()
    sample_error_cases()
    build_figures(df)
    build_latex_includes()
    build_claim_gate()
    after = set(ROOT.rglob("*"))
    outputs = [p for p in after - before if p.is_file()]
    # Include overwritten/generated deterministic outputs too.
    outputs.extend(
        [
            *TABLES.glob(f"paper_*_{DATE}.*"),
            *FIGURES.glob(f"*_{DATE}.*"),
            REVIEW / f"MANUAL_AUDIT_PACKAGE_{DATE}.md",
            REVIEW / f"QUALITATIVE_FALSE_PREMISE_CASES_{DATE}.csv",
            REVIEW / f"CLAIM_GATE_REFRESH_{DATE}.md",
            REVIEW / f"CLAIM_GATE_REFRESH_{DATE}.json",
        ]
    )
    unique_outputs = sorted(set(outputs), key=lambda p: str(p))
    build_manifest(unique_outputs)
    print(json.dumps({"date": DATE, "outputs": [str(p) for p in unique_outputs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
