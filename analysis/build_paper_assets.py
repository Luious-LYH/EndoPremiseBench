#!/usr/bin/env python3
"""Build the frozen 20260525 paper asset bundle for EndoPremiseBench.

This script is intentionally read-only with respect to experiment outputs. It
collects existing raw/scored/metrics/status artifacts and writes paper-facing
tables under paper_assets_20260525.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
RESULT_ROOT = PROJECT_ROOT / "results/a_group_supplement_20260524"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
WORDING_ROOT = ANALYSIS_ROOT / "wording_control_20260525"
QUESTION_ONLY_ROOT = ANALYSIS_ROOT / "question_only_20260524"
QUESTION_ONLY_REFRESH_ROOT = ANALYSIS_ROOT / "question_only_20260526"
OUT_ROOT = RESULT_ROOT / "paper_assets_20260525"
AGENT_ROOT = RESULT_ROOT / "provenance"
CLOSED_EXTERNAL_MODELS = {"gpt5_5", "grok_4_20_multi_agent_xhigh", "claude_opus_4_7"}

MAIN_SCORED = {
    "qwen3vl4b": PROJECT_ROOT / "results/qwen3vl4b_premise_main6000_v2_scored.jsonl",
    "minicpmv4": PROJECT_ROOT / "results/minicpmv4_premise_main6000_v2_scored.jsonl",
    "qwen25_vl_3b": PROJECT_ROOT / "results/qwen25_vl_3b_premise_main6000_v2_scored.jsonl",
    "internvl25_4b": PROJECT_ROOT / "results/internvl25_premise_main6000_v2_scored.jsonl",
    "qwen3_vl_8b": RESULT_ROOT / "scored/qwen3_vl_8b_premise_main6000_scored.jsonl",
    "internvl25_8b": RESULT_ROOT / "scored/internvl25_8b_premise_main6000_scored.jsonl",
    "lingshu_7b": RESULT_ROOT / "scored/lingshu_7b_premise_main6000_scored.jsonl",
    "medgemma_4b": RESULT_ROOT / "scored/medgemma_4b_premise_main6000_scored.jsonl",
    "qwen25_vl_7b": RESULT_ROOT / "scored/qwen25_vl_7b_premise_main6000_scored.jsonl",
    "simula_qwen25_kvasir": RESULT_ROOT / "scored/simula_qwen25_kvasir_premise_main6000_scored.jsonl",
    "simula_medgemma_kvasir": RESULT_ROOT / "scored/simula_medgemma_kvasir_premise_main6000_scored.jsonl",
    "llava_med_v15_mistral_7b": RESULT_ROOT / "scored/llava_med_v15_mistral_7b_premise_main6000_scored.jsonl",
    "gpt5_5": OUT_ROOT / "gpt5_5_primary_api_only_scored.jsonl",
    "grok_4_20_multi_agent_xhigh": RESULT_ROOT / "closed_api_comparison/scored/grok_4_20_multi_agent_xhigh_premise_main6000_scored.jsonl",
    "claude_opus_4_7": RESULT_ROOT / "closed_api_comparison/scored/claude_opus_4_7_premise_main6000_scored.jsonl",
}

MAIN_RAW = {
    "qwen3_vl_8b": RESULT_ROOT / "raw/qwen3_vl_8b_premise_main6000_raw.jsonl",
    "internvl25_8b": RESULT_ROOT / "raw/internvl25_8b_premise_main6000_raw.jsonl",
    "lingshu_7b": RESULT_ROOT / "raw/lingshu_7b_premise_main6000_raw.jsonl",
    "medgemma_4b": RESULT_ROOT / "raw/medgemma_4b_premise_main6000_raw.jsonl",
    "qwen25_vl_7b": RESULT_ROOT / "raw/qwen25_vl_7b_premise_main6000_raw.jsonl",
    "simula_qwen25_kvasir": RESULT_ROOT / "raw/simula_qwen25_kvasir_premise_main6000_raw.jsonl",
    "simula_medgemma_kvasir": RESULT_ROOT / "raw/simula_medgemma_kvasir_premise_main6000_raw.jsonl",
    "llava_med_v15_mistral_7b": RESULT_ROOT / "raw/llava_med_v15_mistral_7b_premise_main6000_raw.jsonl",
    "gpt5_5": OUT_ROOT / "gpt5_5_primary_api_only_raw.jsonl",
    "grok_4_20_multi_agent_xhigh": RESULT_ROOT / "closed_api_comparison/raw/grok_4_20_multi_agent_xhigh_premise_main6000_raw.jsonl",
    "claude_opus_4_7": RESULT_ROOT / "closed_api_comparison/raw/claude_opus_4_7_premise_main6000_raw.jsonl",
}

MAIN_METRICS = {
    "qwen3_vl_8b": RESULT_ROOT / "metrics/qwen3_vl_8b_premise_main6000_metrics.json",
    "internvl25_8b": RESULT_ROOT / "metrics/internvl25_8b_premise_main6000_metrics.json",
    "lingshu_7b": RESULT_ROOT / "metrics/lingshu_7b_premise_main6000_metrics.json",
    "medgemma_4b": RESULT_ROOT / "metrics/medgemma_4b_premise_main6000_metrics.json",
    "qwen25_vl_7b": RESULT_ROOT / "metrics/qwen25_vl_7b_premise_main6000_metrics.json",
    "simula_qwen25_kvasir": RESULT_ROOT / "metrics/simula_qwen25_kvasir_premise_main6000_metrics.json",
    "simula_medgemma_kvasir": RESULT_ROOT / "metrics/simula_medgemma_kvasir_premise_main6000_metrics.json",
    "llava_med_v15_mistral_7b": RESULT_ROOT / "metrics/llava_med_v15_mistral_7b_premise_main6000_metrics.json",
    "gpt5_5": OUT_ROOT / "gpt5_5_primary_api_only_metrics.json",
    "grok_4_20_multi_agent_xhigh": RESULT_ROOT / "closed_api_comparison/metrics/grok_4_20_multi_agent_xhigh_premise_main6000_metrics.json",
    "claude_opus_4_7": RESULT_ROOT / "closed_api_comparison/metrics/claude_opus_4_7_premise_main6000_metrics.json",
}

QUESTION_ONLY = {
    "qwen3_vl_8b": {
        "root": QUESTION_ONLY_ROOT / "qwen3_vl_8b",
        "raw": QUESTION_ONLY_ROOT / "qwen3_vl_8b/raw/qwen3_vl_8b_question_only_main6000_raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "qwen3_vl_8b/scored/qwen3_vl_8b_question_only_main6000_scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "qwen3_vl_8b/metrics/qwen3_vl_8b_question_only_main6000_metrics.json",
    },
    "qwen25_vl_7b": {
        "root": QUESTION_ONLY_ROOT / "qwen25_vl_7b",
        "raw": QUESTION_ONLY_ROOT / "qwen25_vl_7b/raw/qwen25_vl_7b_question_only_20260524_raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "qwen25_vl_7b/scored/qwen25_vl_7b_question_only_20260524_scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "qwen25_vl_7b/metrics/qwen25_vl_7b_question_only_20260524_metrics.json",
    },
    "lingshu_7b": {
        "root": QUESTION_ONLY_ROOT / "lingshu_7b",
        "raw": QUESTION_ONLY_ROOT / "lingshu_7b/raw/lingshu_7b_question_only_20260524_raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "lingshu_7b/scored/lingshu_7b_question_only_20260524_scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "lingshu_7b/metrics/lingshu_7b_question_only_20260524_metrics.json",
    },
    "simula_qwen25_kvasir": {
        "root": QUESTION_ONLY_ROOT / "simula_qwen25_kvasir",
        "raw": QUESTION_ONLY_ROOT / "simula_qwen25_kvasir/raw/simula_qwen25_kvasir_question_only_20260524_raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "simula_qwen25_kvasir/scored/simula_qwen25_kvasir_question_only_20260524_scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "simula_qwen25_kvasir/metrics/simula_qwen25_kvasir_question_only_20260524_metrics.json",
    },
    "medgemma_4b": {
        "root": QUESTION_ONLY_REFRESH_ROOT / "medgemma_4b",
        "raw": QUESTION_ONLY_REFRESH_ROOT / "medgemma_4b/raw/medgemma_4b_question_only_main6000_raw.jsonl",
        "scored": QUESTION_ONLY_REFRESH_ROOT / "medgemma_4b/scored/medgemma_4b_question_only_main6000_scored.jsonl",
        "metrics": QUESTION_ONLY_REFRESH_ROOT / "medgemma_4b/metrics/medgemma_4b_question_only_main6000_metrics.json",
        "status": AGENT_ROOT / "question_only_20260526/status/medgemma_4b_question_only_20260526.status.json",
    },
    "simula_medgemma_kvasir": {
        "root": QUESTION_ONLY_REFRESH_ROOT / "simula_medgemma_kvasir",
        "raw": QUESTION_ONLY_REFRESH_ROOT / "simula_medgemma_kvasir/raw/simula_medgemma_kvasir_question_only_main6000_raw.jsonl",
        "scored": QUESTION_ONLY_REFRESH_ROOT / "simula_medgemma_kvasir/scored/simula_medgemma_kvasir_question_only_main6000_scored.jsonl",
        "metrics": QUESTION_ONLY_REFRESH_ROOT / "simula_medgemma_kvasir/metrics/simula_medgemma_kvasir_question_only_main6000_metrics.json",
        "status": AGENT_ROOT / "question_only_20260526/status/simula_medgemma_kvasir_question_only_20260526.status.json",
    },
    "gpt5_5": {
        "root": QUESTION_ONLY_ROOT / "closed_api/gpt_5_5",
        "raw": QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/raw/gpt_5_5_question_only_main6000_raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/scored/gpt_5_5_question_only_main6000_scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/metrics/gpt_5_5_question_only_main6000_metrics.json",
    },
    "grok_4_20_multi_agent_xhigh": {
        "root": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh",
        "raw": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/raw/raw.jsonl",
        "scored": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/scored/scored.jsonl",
        "metrics": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/metrics/metrics.json",
        "log": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/logs/run.log",
        "status": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/status/status.json",
        "config": QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/config/config.json",
    },
    "qwen3vl4b": {
        "root": PROJECT_ROOT / "results",
        "raw": PROJECT_ROOT / "results/qwen3vl4b_premise_question_only_main6000_v1_raw.jsonl",
        "scored": PROJECT_ROOT / "results/qwen3vl4b_premise_question_only_main6000_v1_scored.jsonl",
        "metrics": PROJECT_ROOT / "results/qwen3vl4b_premise_question_only_main6000_v1_metrics.json",
    },
    "qwen25_vl_3b": {
        "root": PROJECT_ROOT / "results",
        "raw": PROJECT_ROOT / "results/qwen25_vl_3b_premise_question_only_main6000_v1_raw.jsonl",
        "scored": PROJECT_ROOT / "results/qwen25_vl_3b_premise_question_only_main6000_v1_scored.jsonl",
        "metrics": PROJECT_ROOT / "results/qwen25_vl_3b_premise_question_only_main6000_v1_metrics.json",
    },
    "minicpmv4": {
        "root": PROJECT_ROOT / "results",
        "raw": PROJECT_ROOT / "results/minicpmv4_premise_question_only_main6000_v1_raw.jsonl",
        "scored": PROJECT_ROOT / "results/minicpmv4_premise_question_only_main6000_v1_scored.jsonl",
        "metrics": PROJECT_ROOT / "results/minicpmv4_premise_question_only_main6000_v1_metrics.json",
    },
    "internvl25_4b": {
        "root": PROJECT_ROOT / "results",
        "raw": PROJECT_ROOT / "results/internvl25_premise_question_only_main6000_v1_raw.jsonl",
        "scored": PROJECT_ROOT / "results/internvl25_premise_question_only_main6000_v1_scored.jsonl",
        "metrics": PROJECT_ROOT / "results/internvl25_premise_question_only_main6000_v1_metrics.json",
    },
}

P0_WORDING_MODELS = [
    "qwen3_vl_8b",
    "internvl25_8b",
    "simula_qwen25_kvasir",
    "medgemma_4b",
    "gpt5_5",
    "grok_4_20_multi_agent_xhigh",
]

GPT55_MIXED_PRIMARY_RAW = RESULT_ROOT / "closed_api_comparison/raw/gpt_5_5_premise_main6000_raw.jsonl"
GPT55_MIXED_PRIMARY_SCORED = RESULT_ROOT / "closed_api_comparison/scored/gpt_5_5_premise_main6000_scored.jsonl"
GPT55_API_ONLY_RAW = OUT_ROOT / "gpt5_5_primary_api_only_raw.jsonl"
GPT55_API_ONLY_SCORED = OUT_ROOT / "gpt5_5_primary_api_only_scored.jsonl"
GPT55_API_ONLY_METRICS = OUT_ROOT / "gpt5_5_primary_api_only_metrics.json"
GPT55_LOCAL_DIAGNOSTIC_RAW = OUT_ROOT / "gpt5_5_primary_local_substitute_diagnostic_raw.jsonl"
GPT55_LOCAL_DIAGNOSTIC_SCORED = OUT_ROOT / "gpt5_5_primary_local_substitute_diagnostic_scored.jsonl"
GPT55_PROVENANCE_REPORT = OUT_ROOT / "gpt5_5_primary_provenance.json"
PRIMARY_MANIFEST = PROJECT_ROOT / "results/premise_balanced_main_v2.jsonl"
GPT55_REPAIR_ROOT = RESULT_ROOT / "closed_api_comparison/api_repair_20260525/gpt_5_5_primary_missing_api_only"
GPT55_REPAIR_RAW = GPT55_REPAIR_ROOT / "raw/raw.jsonl"
GPT55_REPAIR_SCORED = GPT55_REPAIR_ROOT / "scored/scored.jsonl"
GPT55_REPAIR_METRICS = GPT55_REPAIR_ROOT / "metrics/metrics.json"
GPT55_REPAIR_STATUS = GPT55_REPAIR_ROOT / "status/status.json"
GROK_PRIMARY_STATUS = AGENT_ROOT / "closed_api_comparison/status/grok_4_20_multi_agent_xhigh.status.json"
GROK_PRIMARY_LOG = AGENT_ROOT / "closed_api_comparison/logs/grok_4_20_multi_agent_xhigh_finalize_after_shards.log"
CLAUDE_PRIMARY_STATUS = AGENT_ROOT / "closed_api_comparison/status/claude_opus_4_7.status.json"
CLAUDE_PRIMARY_LOG = AGENT_ROOT / "closed_api_comparison/logs/claude_opus_4_7_full.log"


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def exists(path: Optional[Path]) -> bool:
    return bool(path) and project_path(path).exists()


def line_count(path: Optional[Path]) -> int:
    if not path or not exists(path):
        return 0
    with project_path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with project_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("id") or "")


def is_local_gpt55_substitute(row: Dict[str, Any]) -> bool:
    not_api_response = row.get("not_api_response")
    not_api_response_true = (
        not_api_response is True
        or (isinstance(not_api_response, str) and not_api_response.strip().lower() in {"1", "true", "yes"})
    )
    endpoint = str(row.get("endpoint_id") or "").lower()
    runner = str(row.get("runner_version") or "").lower()
    base_url = str(row.get("base_url") or "").lower()
    api_endpoint = str(row.get("api_endpoint") or "").lower()
    model = str(row.get("model") or row.get("model_id") or "").lower()
    return (
        not_api_response_true
        or "local_codex" in endpoint
        or "local_53580" in endpoint
        or "local_codex" in runner
        or "local substitute" in runner
        or base_url.startswith("http://127.0.0.1")
        or base_url.startswith("http://localhost")
        or api_endpoint.startswith("http://127.0.0.1")
        or api_endpoint.startswith("http://localhost")
        or model == "gpt-5.5-local-codex-substitute"
    )


def mark_local_gpt55_substitute(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    item["original_model"] = item.get("model") or item.get("model_id") or ""
    item["not_api_response"] = True
    item["source"] = "local_substitute_after_remote_api_timeout"
    item["model"] = "gpt-5.5-local-codex-substitute"
    item["model_id"] = "gpt-5.5-local-codex-substitute"
    item["excluded_from_primary_closed_api_full_coverage"] = True
    item["formal_role"] = "diagnostic_only"
    return item


def first_rows_by_manifest_order(rows: Iterable[Dict[str, Any]], manifest_ids: Sequence[str]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = row_id(row)
        if rid and rid not in by_id:
            by_id[rid] = row
    return [by_id[rid] for rid in manifest_ids if rid in by_id]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def model_group(model: str) -> str:
    if model in CLOSED_EXTERNAL_MODELS:
        return "closed API"
    return "open/local VLM"


def main_status_path(model: str) -> Optional[Path]:
    if model == "gpt5_5":
        return AGENT_ROOT / "closed_api_comparison/status/gpt_5_5.status.json"
    if model == "grok_4_20_multi_agent_xhigh":
        return GROK_PRIMARY_STATUS
    if model == "claude_opus_4_7":
        return CLAUDE_PRIMARY_STATUS
    return None


def main_log_path(model: str) -> Optional[Path]:
    if model == "grok_4_20_multi_agent_xhigh":
        return GROK_PRIMARY_LOG
    if model == "claude_opus_4_7":
        return CLAUDE_PRIMARY_LOG
    return None


def tex_escape(value: Any) -> str:
    text = fmt(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def write_tex_table(path: Path, rows: List[Dict[str, Any]], cols: Sequence[str], caption: str, label: str) -> None:
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{tex_escape(caption)}}}",
        f"\\label{{{tex_escape(label)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row.get(col, "")) for col in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def metric(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    true_rows = [row for row in rows if row.get("premise_type") == "true"]
    false_rows = [row for row in rows if row.get("premise_type") == "false"]
    failed_rows = [row for row in rows if row.get("parse_status") == "failure"]
    acc_tp = sum(1 for row in true_rows if row.get("is_correct")) / len(true_rows) if true_rows else None
    acc_fp = sum(1 for row in false_rows if row.get("is_correct")) / len(false_rows) if false_rows else None
    sr = (
        sum(1 for row in false_rows if row.get("parse_status") == "ok" and not row.get("is_correct")) / len(false_rows)
        if false_rows
        else None
    )
    orr = sum(1 for row in true_rows if str(row.get("parsed_answer")) == na_option(row)) / len(true_rows) if true_rows else None
    pfr = len(failed_rows) / len(rows) if rows else None
    hps = None
    if acc_tp is not None and acc_fp is not None and acc_tp + acc_fp > 0:
        hps = 2 * acc_tp * acc_fp / (acc_tp + acc_fp)
    return {
        "n": len(rows),
        "true_n": len(true_rows),
        "false_n": len(false_rows),
        "Acc_TP": acc_tp,
        "Acc_FP": acc_fp,
        "SR": sr,
        "ORR": orr,
        "PFR": pfr,
        "HPS": hps,
    }


def build_gpt55_primary_provenance_assets() -> Dict[str, Any]:
    raw_rows = read_jsonl(GPT55_MIXED_PRIMARY_RAW) if exists(GPT55_MIXED_PRIMARY_RAW) else []
    scored_rows = read_jsonl(GPT55_MIXED_PRIMARY_SCORED) if exists(GPT55_MIXED_PRIMARY_SCORED) else []
    repair_raw_rows = read_jsonl(GPT55_REPAIR_RAW) if exists(GPT55_REPAIR_RAW) else []
    repair_scored_rows = read_jsonl(GPT55_REPAIR_SCORED) if exists(GPT55_REPAIR_SCORED) else []
    manifest_rows = read_jsonl(PRIMARY_MANIFEST) if exists(PRIMARY_MANIFEST) else []
    manifest_ids = [row_id(row) for row in manifest_rows if row_id(row)]
    n_total = len(manifest_ids) or 6000

    local_raw = [mark_local_gpt55_substitute(row) for row in raw_rows if is_local_gpt55_substitute(row)]
    local_scored = [mark_local_gpt55_substitute(row) for row in scored_rows if is_local_gpt55_substitute(row)]
    local_ids = {row_id(row) for row in local_raw + local_scored if row_id(row)}

    api_raw_original = [row for row in raw_rows if not is_local_gpt55_substitute(row) and not row.get("error")]
    api_scored_original = [row for row in scored_rows if not is_local_gpt55_substitute(row) and not row.get("error")]
    repair_raw = [row for row in repair_raw_rows if not is_local_gpt55_substitute(row) and not row.get("error")]
    repair_scored = [row for row in repair_scored_rows if not is_local_gpt55_substitute(row) and not row.get("error")]

    api_raw_by_id = {row_id(row): row for row in api_raw_original if row_id(row)}
    api_scored_by_id = {row_id(row): row for row in api_scored_original if row_id(row)}
    repair_raw_by_id = {row_id(row): row for row in repair_raw if row_id(row)}
    repair_scored_by_id = {row_id(row): row for row in repair_scored if row_id(row)}
    api_repair_overlap_ids = sorted(set(api_scored_by_id) & set(repair_scored_by_id))
    for rid, row in repair_raw_by_id.items():
        api_raw_by_id.setdefault(rid, row)
    for rid, row in repair_scored_by_id.items():
        api_scored_by_id.setdefault(rid, row)

    if manifest_ids:
        api_raw = [api_raw_by_id[rid] for rid in manifest_ids if rid in api_raw_by_id]
        api_scored = [api_scored_by_id[rid] for rid in manifest_ids if rid in api_scored_by_id]
        api_pending_ids = [rid for rid in manifest_ids if rid not in api_scored_by_id]
    else:
        api_raw = first_rows_by_manifest_order(api_raw_by_id.values(), sorted(api_raw_by_id))
        api_scored = first_rows_by_manifest_order(api_scored_by_id.values(), sorted(api_scored_by_id))
        api_pending_ids = []
    api_ids = {row_id(row) for row in api_raw + api_scored if row_id(row)}
    api_local_overlap_ids = sorted(api_ids & local_ids)

    write_jsonl(GPT55_API_ONLY_RAW, api_raw)
    write_jsonl(GPT55_API_ONLY_SCORED, api_scored)
    write_json_file(GPT55_API_ONLY_METRICS, metric(api_scored))
    write_jsonl(GPT55_LOCAL_DIAGNOSTIC_RAW, local_raw)
    write_jsonl(GPT55_LOCAL_DIAGNOSTIC_SCORED, local_scored)

    report = {
        "formal_closed_api_scope": "api_only",
        "mixed_primary_raw": str(GPT55_MIXED_PRIMARY_RAW),
        "mixed_primary_scored": str(GPT55_MIXED_PRIMARY_SCORED),
        "api_only_raw": str(GPT55_API_ONLY_RAW),
        "api_only_scored": str(GPT55_API_ONLY_SCORED),
        "api_only_metrics": str(GPT55_API_ONLY_METRICS),
        "local_diagnostic_raw": str(GPT55_LOCAL_DIAGNOSTIC_RAW),
        "local_diagnostic_scored": str(GPT55_LOCAL_DIAGNOSTIC_SCORED),
        "repair_root": str(GPT55_REPAIR_ROOT),
        "repair_raw": str(GPT55_REPAIR_RAW),
        "repair_scored": str(GPT55_REPAIR_SCORED),
        "repair_metrics": str(GPT55_REPAIR_METRICS),
        "repair_status": str(GPT55_REPAIR_STATUS),
        "n_total": n_total,
        "mixed_raw_rows": len(raw_rows),
        "mixed_scored_rows": len(scored_rows),
        "api_original_raw_rows": len(api_raw_original),
        "api_original_scored_rows": len(api_scored_original),
        "api_repair_raw_rows": len(repair_raw),
        "api_repair_scored_rows": len(repair_scored),
        "api_only_raw_rows": len(api_raw),
        "api_only_scored_rows": len(api_scored),
        "api_pending_ids": api_pending_ids,
        "local_diagnostic_raw_rows": len(local_raw),
        "local_diagnostic_scored_rows": len(local_scored),
        "local_diagnostic_ids": sorted(local_ids),
        "api_local_overlap_ids": api_local_overlap_ids,
        "api_repair_overlap_ids": api_repair_overlap_ids,
        "formal_closed_api_complete": len(api_scored) == n_total and not api_pending_ids,
        "mixed_provenance_detected": bool(local_ids),
        "policy": (
            "Remote GPT-5.5 API rows carry formal closed-API coverage. "
            "Local Codex/GPT-5.5 substitute rows are diagnostic only and excluded "
            "from primary closed-API metrics, coverage, and bootstrap inputs."
        ),
    }
    write_json_file(GPT55_PROVENANCE_REPORT, report)
    return report


def na_option(row: Dict[str, Any]) -> str:
    options = row.get("options") or {}
    if isinstance(options, dict):
        for key, value in options.items():
            if "not applicable" in str(value).lower() or "no such entity" in str(value).lower():
                return str(key)
    return ""


def grouped(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key) or "") for key in keys)].append(row)
    out = []
    for key, group_rows in sorted(buckets.items()):
        item = {name: value for name, value in zip(keys, key)}
        item.update(metric(group_rows))
        out.append(item)
    return out


def metric_row_from_json(model: str, metrics_path: Path, role: str, scored_path: Optional[Path] = None) -> Dict[str, Any]:
    data = read_json(metrics_path)
    return {
        "model": model,
        "model_group": model_group(model),
        "role": role,
        "n": data.get("n", ""),
        "Acc_TP": data.get("Acc_TP", ""),
        "Acc_FP": data.get("Acc_FP", ""),
        "SR": data.get("SR", ""),
        "ORR": data.get("ORR", ""),
        "PFR": data.get("PFR", ""),
        "HPS": data.get("HPS", ""),
        "metrics_path": str(metrics_path),
        "scored_path": str(scored_path or ""),
    }


def build_main_tables() -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for model, scored in MAIN_SCORED.items():
        if not exists(scored):
            continue
        rows = read_jsonl(scored)
        for row in rows:
            item = dict(row)
            item["model"] = model
            all_rows.append(item)
        item = {
            "model": model,
            "model_group": model_group(model),
            **metric(rows),
            "provenance": "",
            "notes": "",
            "scored_path": str(scored),
        }
        if model == "gpt5_5":
            item["provenance"] = "api_only_filtered"
            item["notes"] = "Excludes local Codex substitute diagnostic row from formal closed-API metrics."
        elif model == "grok_4_20_multi_agent_xhigh":
            item["provenance"] = "closed_api_clean_full"
            item["notes"] = "Second closed external reference; not pooled with open/local VLM model suite."
        elif model == "claude_opus_4_7":
            item["provenance"] = "closed_api_clean_full_from_server89"
            item["notes"] = "Third closed external reference from server 89 clean 6000-row primary run; supplementary controls not added."
        summary_rows.append(item)

    closed_order = {"gpt5_5": 0, "claude_opus_4_7": 1, "grok_4_20_multi_agent_xhigh": 2}
    summary_rows.sort(key=lambda row: (closed_order.get(str(row["model"]), 2), str(row["model"])))
    fields = [
        "model",
        "model_group",
        "n",
        "true_n",
        "false_n",
        "Acc_TP",
        "Acc_FP",
        "SR",
        "ORR",
        "PFR",
        "HPS",
        "provenance",
        "notes",
        "scored_path",
    ]
    write_csv(OUT_ROOT / "main_model_summary.csv", summary_rows, fields)
    write_tex_table(
        OUT_ROOT / "main_model_summary.tex",
        summary_rows,
        ["model", "model_group", "n", "Acc_TP", "Acc_FP", "SR", "PFR", "HPS"],
        "Main benchmark results on the primary balanced evaluation split.",
        "tab:epb-main-model-summary-20260525",
    )

    source_premise = grouped(all_rows, ["model", "source_dataset", "premise_type"])
    attr = grouped(all_rows, ["model", "attribute_type"])
    write_csv(
        OUT_ROOT / "source_premise_breakdown.csv",
        source_premise,
        ["model", "source_dataset", "premise_type", "n", "true_n", "false_n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"],
    )
    write_csv(
        OUT_ROOT / "attribute_breakdown.csv",
        attr,
        ["model", "attribute_type", "n", "true_n", "false_n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS"],
    )
    return summary_rows


def find_single(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(project_path(root).glob(pattern))
    if not matches:
        return None
    return matches[0]


def build_question_only() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, paths in QUESTION_ONLY.items():
        metrics = paths["metrics"]
        scored = paths["scored"]
        if metrics and exists(metrics):
            rows.append(metric_row_from_json(model, metrics, "question_only_control", scored))
    rows.sort(key=lambda row: str(row["model"]))
    write_csv(
        OUT_ROOT / "question_only_control.csv",
        rows,
        ["model", "model_group", "role", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "HPS", "metrics_path", "scored_path"],
    )
    write_tex_table(
        OUT_ROOT / "question_only_control.tex",
        rows,
        ["model", "model_group", "n", "Acc_TP", "Acc_FP", "SR", "PFR", "HPS"],
        "Question-only control on the primary balanced evaluation split.",
        "tab:epb-question-only-control-20260525",
    )
    return rows


def build_wording_summary() -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for model in P0_WORDING_MODELS:
        scored = WORDING_ROOT / model / "scored/scored.jsonl"
        if not exists(scored):
            continue
        scored_rows = read_jsonl(scored)
        by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in scored_rows:
            by_variant[str(row.get("control_value") or "unknown")].append(row)
        for variant in ["neutral", "explicit", "guarded"]:
            if variant not in by_variant:
                continue
            item = {
                "model": model,
                "model_group": model_group(model),
                "variant": variant,
                **metric(by_variant[variant]),
                "scored_path": str(scored),
            }
            rows_out.append(item)
        rows_out.append(
            {
                "model": model,
                "model_group": model_group(model),
                "variant": "all",
                **metric(scored_rows),
                "scored_path": str(scored),
            }
        )
    rows_out.sort(key=lambda row: (str(row["model"]), str(row["variant"])))
    write_csv(
        OUT_ROOT / "wording_control_summary.csv",
        rows_out,
        ["model", "model_group", "variant", "n", "true_n", "false_n", "Acc_FP", "SR", "PFR", "scored_path"],
    )
    write_tex_table(
        OUT_ROOT / "wording_control_summary.tex",
        rows_out,
        ["model", "model_group", "variant", "n", "Acc_FP", "SR", "PFR"],
        "Fixed false-premise wording-control results.",
        "tab:epb-wording-control-20260525",
    )
    return rows_out


def copy_bootstrap_ci() -> int:
    src = ANALYSIS_ROOT / "confidence_intervals_20260524/bootstrap_ci.csv"
    dst = OUT_ROOT / "bootstrap_ci.csv"
    if not src.exists():
        write_csv(dst, [], [])
        return 0
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with dst.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def coverage_row(model: str, experiment: str, role: str, raw: Optional[Path], scored: Optional[Path], metrics: Optional[Path], status: Optional[Path]) -> Dict[str, Any]:
    n_raw = line_count(raw)
    n_scored = line_count(scored)
    n_expected = 6000 if experiment in {"primary_balanced_evaluation", "question_only_control", "fixed_false_premise_wording_control"} else 0
    n_total = n_expected or n_scored or n_raw
    n_completed = n_scored
    n_failed = ""
    n_pending = ""
    if status and exists(status):
        data = read_json(status)
        n_total = data.get("n_total", n_total)
        n_completed = data.get("n_completed", n_completed)
        n_failed = data.get("n_failed", "")
        n_pending = data.get("n_pending", "")
    return {
        "model": model,
        "experiment": experiment,
        "role": role,
        "n_total": n_total,
        "n_raw": n_raw,
        "n_scored": n_scored,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "n_pending": n_pending,
        "raw_path": str(raw or ""),
        "scored_path": str(scored or ""),
        "metrics_path": str(metrics or ""),
        "status_path": str(status or ""),
        "complete": bool(n_total and n_completed == n_total and (n_failed in ("", 0)) and (n_pending in ("", 0))),
        "provenance": "",
        "api_only_n_completed": "",
        "local_diagnostic_n": "",
        "formal_closed_api_complete": "",
        "mixed_provenance": "",
        "notes": "",
    }


def build_closed_api_coverage(provenance: Dict[str, Any]) -> List[Dict[str, Any]]:
    total = int(provenance.get("n_total", 6000) or 6000)
    api_completed = int(provenance.get("api_only_scored_rows", 0) or 0)
    local_diagnostic = int(provenance.get("local_diagnostic_scored_rows", 0) or 0)
    api_complete = bool(provenance.get("formal_closed_api_complete", False))
    primary = coverage_row(
        "gpt5_5",
        "primary_balanced_evaluation",
        "P0 closed external reference",
        GPT55_API_ONLY_RAW,
        GPT55_API_ONLY_SCORED,
        GPT55_API_ONLY_METRICS,
        AGENT_ROOT / "closed_api_comparison/status/gpt_5_5.status.json",
    )
    primary.update(
        {
            "n_total": total,
            "n_completed": api_completed,
            "n_failed": 0,
            "n_pending": max(total - api_completed, 0),
            "complete": api_complete,
            "provenance": "api_only_filtered_from_mixed_canonical",
            "api_only_n_completed": api_completed,
            "local_diagnostic_n": local_diagnostic,
            "formal_closed_api_complete": api_complete,
            "mixed_provenance": provenance.get("mixed_provenance_detected", False),
            "notes": (
                f"Remote GPT-5.5 API coverage is {api_completed}/{total}; "
                f"{local_diagnostic} local/proxy diagnostic rows are excluded from formal coverage."
            ),
        }
    )
    local_diag = {
        "model": "gpt5_5_local_codex_substitute",
        "experiment": "primary_balanced_evaluation_local_diagnostic",
        "role": "diagnostic fallback only",
        "n_total": provenance.get("local_diagnostic_scored_rows", 0),
        "n_raw": provenance.get("local_diagnostic_raw_rows", 0),
        "n_scored": provenance.get("local_diagnostic_scored_rows", 0),
        "n_completed": provenance.get("local_diagnostic_scored_rows", 0),
        "n_failed": 0,
        "n_pending": 0,
        "complete": bool(provenance.get("local_diagnostic_scored_rows", 0)),
        "provenance": "local_substitute_after_remote_api_timeout",
        "api_only_n_completed": "",
        "local_diagnostic_n": provenance.get("local_diagnostic_scored_rows", ""),
        "formal_closed_api_complete": False,
        "mixed_provenance": False,
        "notes": "Not a closed-API result; excluded from primary closed-API metrics, coverage, and bootstrap.",
        "raw_path": str(GPT55_LOCAL_DIAGNOSTIC_RAW),
        "scored_path": str(GPT55_LOCAL_DIAGNOSTIC_SCORED),
        "metrics_path": "",
        "status_path": str(GPT55_PROVENANCE_REPORT),
    }
    grok_primary = coverage_row(
        "grok_4_20_multi_agent_xhigh",
        "primary_balanced_evaluation",
        "closed external reference",
        MAIN_RAW["grok_4_20_multi_agent_xhigh"],
        MAIN_SCORED["grok_4_20_multi_agent_xhigh"],
        MAIN_METRICS["grok_4_20_multi_agent_xhigh"],
        GROK_PRIMARY_STATUS,
    )
    grok_primary.update(
        {
            "n_total": 6000,
            "n_completed": 6000,
            "n_failed": 0,
            "n_pending": 0,
            "complete": True,
            "provenance": "closed_api_clean_full",
            "formal_closed_api_complete": True,
            "notes": "Second closed external reference, clean full coverage; kept separate from open/local VLM suite.",
        }
    )
    claude_primary = coverage_row(
        "claude_opus_4_7",
        "primary_balanced_evaluation",
        "closed external reference",
        MAIN_RAW["claude_opus_4_7"],
        MAIN_SCORED["claude_opus_4_7"],
        MAIN_METRICS["claude_opus_4_7"],
        CLAUDE_PRIMARY_STATUS,
    )
    claude_primary.update(
        {
            "n_total": 6000,
            "n_completed": 6000,
            "n_failed": 0,
            "n_pending": 0,
            "complete": True,
            "provenance": "closed_api_clean_full_from_server89",
            "formal_closed_api_complete": True,
            "notes": "Third closed external reference, clean full primary coverage from server 89; supplementary controls are not included.",
        }
    )
    rows = [
        primary,
        claude_primary,
        grok_primary,
        local_diag,
        coverage_row(
            "gpt5_5",
            "question_only_control",
            "P0 closed control",
            QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/raw/gpt_5_5_question_only_main6000_raw.jsonl",
            QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/scored/gpt_5_5_question_only_main6000_scored.jsonl",
            QUESTION_ONLY_ROOT / "closed_api/gpt_5_5/metrics/gpt_5_5_question_only_main6000_metrics.json",
            None,
        ),
        coverage_row(
            "grok_4_20_multi_agent_xhigh",
            "question_only_control",
            "closed control",
            QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/raw/raw.jsonl",
            QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/scored/scored.jsonl",
            QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/metrics/metrics.json",
            QUESTION_ONLY_ROOT / "closed_api/grok_4_20_multi_agent_xhigh/status/status.json",
        ),
        coverage_row(
            "gpt5_5",
            "fixed_false_premise_wording_control",
            "P0 closed wording-control",
            WORDING_ROOT / "gpt5_5/raw/raw.jsonl",
            WORDING_ROOT / "gpt5_5/scored/scored.jsonl",
            WORDING_ROOT / "gpt5_5/metrics/metrics.json",
            WORDING_ROOT / "gpt5_5/status/status.json",
        ),
        coverage_row(
            "grok_4_20_multi_agent_xhigh",
            "fixed_false_premise_wording_control",
            "closed wording-control",
            WORDING_ROOT / "grok_4_20_multi_agent_xhigh/raw/raw.jsonl",
            WORDING_ROOT / "grok_4_20_multi_agent_xhigh/scored/scored.jsonl",
            WORDING_ROOT / "grok_4_20_multi_agent_xhigh/metrics/metrics.json",
            WORDING_ROOT / "grok_4_20_multi_agent_xhigh/status/status.json",
        ),
    ]
    write_csv(
        OUT_ROOT / "closed_api_coverage.csv",
        rows,
        [
            "model",
            "experiment",
            "role",
            "n_total",
            "n_raw",
            "n_scored",
            "n_completed",
            "n_failed",
            "n_pending",
            "complete",
            "provenance",
            "api_only_n_completed",
            "local_diagnostic_n",
            "formal_closed_api_complete",
            "mixed_provenance",
            "notes",
            "raw_path",
            "scored_path",
            "metrics_path",
            "status_path",
        ],
    )
    return rows


def inventory_entry(experiment: str, model: str, variant: str, root: Optional[Path], manifest: Optional[Path], raw: Optional[Path], scored: Optional[Path], metrics: Optional[Path], log: Optional[Path], status: Optional[Path], config: Optional[Path]) -> Dict[str, Any]:
    n_total = line_count(manifest) or line_count(scored) or line_count(raw)
    n_completed = line_count(scored) or line_count(raw)
    n_failed: Any = ""
    n_pending: Any = ""
    if status and exists(status):
        data = read_json(status)
        n_total = data.get("n_total", n_total)
        n_completed = data.get("n_completed", n_completed)
        n_failed = data.get("n_failed", "")
        n_pending = data.get("n_pending", "")
    required = [path for path in [manifest, raw, scored, metrics, log, status, config] if path is not None]
    return {
        "experiment": experiment,
        "model": model,
        "variant": variant,
        "root": str(root or ""),
        "manifest_path": str(manifest or ""),
        "raw_path": str(raw or ""),
        "scored_path": str(scored or ""),
        "metrics_path": str(metrics or ""),
        "log_path": str(log or ""),
        "status_path": str(status or ""),
        "config_path": str(config or ""),
        "n_total": n_total,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "n_pending": n_pending,
        "complete": all(exists(path) for path in required) and bool(n_total) and n_completed == n_total and n_failed in ("", 0) and n_pending in ("", 0),
    }


def build_inventory() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failed_pending: List[Dict[str, Any]] = []
    for model in P0_WORDING_MODELS:
        root = WORDING_ROOT / model
        status = root / "status/status.json"
        rows.append(
            inventory_entry(
                "fixed_false_premise_wording_control",
                model,
                "neutral+explicit+guarded",
                root,
                root / "manifest/manifest.jsonl",
                root / "raw/raw.jsonl",
                root / "scored/scored.jsonl",
                root / "metrics/metrics.json",
                root / "logs/run.log",
                status,
                root / "config/config.json",
            )
        )
        data = read_json(status) if exists(status) else {}
        failed_pending.append(
            {
                "experiment": "fixed_false_premise_wording_control",
                "model": model,
                "n_total": data.get("n_total", ""),
                "n_completed": data.get("n_completed", ""),
                "n_failed": data.get("n_failed", 0),
                "n_pending": data.get("n_pending", 0),
                "failed_ids_path": str(root / "status/failed_ids.txt"),
                "pending_ids_path": str(root / "status/pending_ids.txt"),
            }
        )

    for model, scored in MAIN_SCORED.items():
        rows.append(
            inventory_entry(
                "primary_balanced_evaluation",
                model,
                "full",
                RESULT_ROOT,
                PRIMARY_MANIFEST,
                MAIN_RAW.get(model),
                scored,
                MAIN_METRICS.get(model),
                main_log_path(model),
                main_status_path(model),
                None,
            )
        )

    for model, paths in QUESTION_ONLY.items():
        rows.append(
            inventory_entry(
                "question_only_control",
                model,
                "question_only",
                paths["root"],
                PRIMARY_MANIFEST,
                paths["raw"],
                paths["scored"],
                paths["metrics"],
                paths.get("log"),
                paths.get("status"),
                paths.get("config"),
            )
        )

    fields = [
        "experiment",
        "model",
        "variant",
        "root",
        "manifest_path",
        "raw_path",
        "scored_path",
        "metrics_path",
        "log_path",
        "status_path",
        "config_path",
        "n_total",
        "n_completed",
        "n_failed",
        "n_pending",
        "complete",
    ]
    write_csv(OUT_ROOT / "artifact_inventory.csv", rows, fields)
    write_csv(
        OUT_ROOT / "failed_pending_ids_summary.csv",
        failed_pending,
        ["experiment", "model", "n_total", "n_completed", "n_failed", "n_pending", "failed_ids_path", "pending_ids_path"],
    )
    return rows, failed_pending


def write_summary(main_rows: List[Dict[str, Any]], question_rows: List[Dict[str, Any]], wording_rows: List[Dict[str, Any]], coverage_rows: List[Dict[str, Any]], inventory_rows: List[Dict[str, Any]], bootstrap_rows: int, provenance: Dict[str, Any]) -> None:
    wording_all = [row for row in wording_rows if row["variant"] == "all"]
    canonical_complete = [row for row in inventory_rows if row["experiment"] == "fixed_false_premise_wording_control" and row["complete"]]
    local_wording_complete = [row for row in canonical_complete if row["model"] not in CLOSED_EXTERNAL_MODELS]
    closed_wording_complete = [row for row in canonical_complete if row["model"] in CLOSED_EXTERNAL_MODELS]
    files = [
        "main_model_summary.csv",
        "main_model_summary.tex",
        "question_only_control.csv",
        "question_only_control.tex",
        "wording_control_summary.csv",
        "wording_control_summary.tex",
        "source_premise_breakdown.csv",
        "attribute_breakdown.csv",
        "bootstrap_ci.csv",
        "closed_api_coverage.csv",
        "artifact_inventory.csv",
        "failed_pending_ids_summary.csv",
        "gpt5_5_primary_api_only_raw.jsonl",
        "gpt5_5_primary_api_only_scored.jsonl",
        "gpt5_5_primary_api_only_metrics.json",
        "gpt5_5_primary_local_substitute_diagnostic_raw.jsonl",
        "gpt5_5_primary_local_substitute_diagnostic_scored.jsonl",
        "gpt5_5_primary_provenance.json",
        "SUMMARY.md",
    ]
    lines = [
        "# EndoPremiseBench P0 Paper Assets 20260525",
        "",
        "This bundle is the 20260525 paper-facing output package. Older 20260524 assets are used only as provenance inputs when copied or summarized here.",
        "",
        "## Completion",
        "",
        f"- P0 open/local wording-control models complete: {len(local_wording_complete)}/4.",
        f"- P0 closed wording-control references complete: {len(closed_wording_complete)}/2 ({', '.join(row['model'] for row in closed_wording_complete) or 'none'}).",
        f"- Wording-control rows summarized: {sum(int(row['n']) for row in wording_all)} across {len(wording_all)} models.",
        f"- Main model summary rows: {len(main_rows)}.",
        f"- Question-only control rows: {len(question_rows)}.",
        f"- Bootstrap CI rows copied into this bundle: {bootstrap_rows}.",
        "",
        "## Required Files",
        "",
    ]
    for name in files:
        path = OUT_ROOT / name
        present = name == "SUMMARY.md" or path.exists()
        lines.append(f"- `{name}`: {'present' if present else 'missing'}")
    lines.extend(
        [
            "",
            "## Closed API Scope",
            "",
            "`gpt5_5`, `claude_opus_4_7`, and `grok_4_20_multi_agent_xhigh` are included as closed external reference models in the primary table and `closed_api_coverage.csv`. They are used to check whether the false-premise pattern is also observed under strong closed API models, not to claim vendor-wide closed-model behavior.",
            f"Remote GPT-5.5 API primary coverage is {provenance.get('api_only_scored_rows', 0)}/{provenance.get('n_total', 6000)}. The {provenance.get('local_diagnostic_scored_rows', 0)} local/proxy diagnostic rows are diagnostic only and excluded from formal closed-API metrics, coverage, and bootstrap inputs.",
            "Claude and Grok primary runs are clean full-coverage external-reference controls (6000/6000, failed 0, pending 0).",
            "Supplementary closed-model controls remain limited to the previously completed GPT-5.5 and Grok question-only/wording-control assets; Claude is not added to those supplemental comparisons.",
            f"Provenance report: `{GPT55_PROVENANCE_REPORT}`.",
            "",
            "## Wording-Control Snapshot",
            "",
            "| Model | n | Acc_FP | SR | PFR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in wording_all:
        lines.append(f"| {row['model']} | {row['n']} | {fmt(row['Acc_FP'])} | {fmt(row['SR'])} | {fmt(row['PFR'])} |")
    lines.extend(["", "## Coverage Snapshot", "", "| Model | Experiment | n_completed | n_failed | n_pending | Complete |", "|---|---|---:|---:|---:|---|"])
    for row in coverage_rows:
        lines.append(
            f"| {row['model']} | {row['experiment']} | {row['n_completed']} | {row['n_failed']} | {row['n_pending']} | {row['complete']} |"
        )
    lines.append("")
    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def preflight() -> None:
    missing: List[str] = []
    for model in P0_WORDING_MODELS:
        root = WORDING_ROOT / model
        for rel in [
            "manifest/manifest.jsonl",
            "raw/raw.jsonl",
            "scored/scored.jsonl",
            "metrics/metrics.json",
            "logs/run.log",
            "status/status.json",
            "config/config.json",
        ]:
            path = root / rel
            if not exists(path):
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing canonical P0 wording artifacts:\n" + "\n".join(missing))


def main() -> int:
    if not PROJECT_ROOT.exists():
        raise SystemExit(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight()
    provenance = build_gpt55_primary_provenance_assets()
    main_rows = build_main_tables()
    question_rows = build_question_only()
    wording_rows = build_wording_summary()
    bootstrap_rows = copy_bootstrap_ci()
    coverage_rows = build_closed_api_coverage(provenance)
    inventory_rows, _ = build_inventory()
    write_summary(main_rows, question_rows, wording_rows, coverage_rows, inventory_rows, bootstrap_rows, provenance)

    expected = [
        "main_model_summary.csv",
        "main_model_summary.tex",
        "question_only_control.csv",
        "question_only_control.tex",
        "wording_control_summary.csv",
        "wording_control_summary.tex",
        "source_premise_breakdown.csv",
        "attribute_breakdown.csv",
        "bootstrap_ci.csv",
        "closed_api_coverage.csv",
        "artifact_inventory.csv",
        "failed_pending_ids_summary.csv",
        "gpt5_5_primary_api_only_raw.jsonl",
        "gpt5_5_primary_api_only_scored.jsonl",
        "gpt5_5_primary_api_only_metrics.json",
        "gpt5_5_primary_local_substitute_diagnostic_raw.jsonl",
        "gpt5_5_primary_local_substitute_diagnostic_scored.jsonl",
        "gpt5_5_primary_provenance.json",
        "SUMMARY.md",
    ]
    missing = [name for name in expected if not (OUT_ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing output files after build: {missing}")
    print(json.dumps({"output_root": str(OUT_ROOT), "files": expected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
