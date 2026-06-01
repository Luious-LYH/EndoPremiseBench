#!/usr/bin/env python3
"""Parse and score EndoPremiseBench MCQ outputs conservatively."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCORER_VERSION = "endo_premise_parser_v2_20260522"
LETTERS = {"A", "B", "C", "D"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_raw_output(row: Dict[str, Any]) -> str:
    for key in ("raw_output", "prediction", "raw_answer", "output", "response", "model_output"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def parse_json_answer(text: str) -> Optional[str]:
    candidates = [text.strip()]
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, flags=re.S))
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text or "")]:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                candidates.append(json.dumps(obj))
        except Exception:
            continue
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        value = obj.get("answer")
        if value is None:
            continue
        value_text = str(value).strip().upper()
        match = re.fullmatch(r"[ABCD]", value_text)
        if match:
            return value_text
        match = re.match(r"^\s*([ABCD])[\).:\s-]*", value_text)
        if match:
            return match.group(1)
    return None


def regex_letters(text: str) -> List[str]:
    upper = str(text or "").upper()
    patterns = [
        r"\bANSWER\s*[:=]\s*([ABCD])\b",
        r"\bOPTION\s*[:=]?\s*([ABCD])\b",
        r"\bCHOOSE\s*[:=]?\s*([ABCD])\b",
        r"^\s*([ABCD])\s*[\).:\-]?\s*$",
        r"^\s*([ABCD])\s*[\).:\-]\s+",
    ]
    hits: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, upper, flags=re.M):
            hits.append(match.group(1))
    if not hits:
        # Last-resort single standalone letter, kept conservative by requiring
        # exactly one distinct letter.
        hits.extend(re.findall(r"\b([ABCD])\b", upper))
    return hits


def option_text_matches(text: str, options: Dict[str, str]) -> List[str]:
    haystack = normalize_text(text)
    hits = []
    for letter, option in options.items():
        option_norm = normalize_text(option)
        if not option_norm:
            continue
        if option_norm in haystack:
            hits.append(letter)
            continue
        # Allow common N/A paraphrase matching only for the N/A option.
        if "not applicable" in option_norm:
            na_patterns = [
                "not applicable",
                "no such entity",
                "no such object",
                "not visible",
                "none visible",
                "no polyp is visible",
                "no instrument is visible",
                "no abnormal finding is visible",
            ]
            if any(p in haystack for p in na_patterns):
                hits.append(letter)
    return hits


def parse_answer(text: str, options: Dict[str, str]) -> Tuple[Optional[str], str, str]:
    json_answer = parse_json_answer(text)
    if json_answer in LETTERS:
        return json_answer, "ok", "json"

    regex_hits = regex_letters(text)
    distinct = sorted(set(x for x in regex_hits if x in LETTERS))
    if len(distinct) == 1:
        return distinct[0], "ok", "regex"
    if len(distinct) > 1:
        return None, "ambiguous", "regex_multi"

    text_hits = sorted(set(option_text_matches(text, options)))
    if len(text_hits) == 1:
        return text_hits[0], "ok", "option_text"
    if len(text_hits) > 1:
        return None, "ambiguous", "option_text_multi"

    return None, "failure", "no_parse"


def failure_type(sample: Dict[str, Any], parsed_answer: Optional[str], parse_status: str) -> str:
    if parse_status != "ok" or parsed_answer is None:
        return "Format Failure"
    answer = sample["answer"]
    premise_type = sample["premise_type"]
    if premise_type == "true":
        if parsed_answer == answer:
            return "Valid Answer"
        if "not applicable" in normalize_text(sample["options"].get(parsed_answer, "")):
            return "Over-Refusal"
        return "Attribute Misclassification"
    if premise_type == "false":
        if parsed_answer == answer:
            return "Premise Rejection"
        return "Unsupported Attribute Exposure"
    return "Unknown"


def score_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for row in rows:
        sample = row.get("sample") if isinstance(row.get("sample"), dict) else row
        options = sample.get("options") or {}
        raw_output = extract_raw_output(row)
        parsed, status, method = parse_answer(raw_output, options)
        is_correct = bool(status == "ok" and parsed == sample.get("answer"))
        out = dict(row)
        out.update(
            {
                "probe_id": sample.get("id") or row.get("probe_id"),
                "raw_output": raw_output,
                "parsed_answer": parsed,
                "parse_status": status,
                "parse_method": method,
                "is_correct": is_correct,
                "failure_type": failure_type(sample, parsed, status),
                "scorer_version": SCORER_VERSION,
            }
        )
        scored.append(out)
    return scored, compute_metrics(scored)


def safe_div(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def compute_metrics(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(scored)
    by_premise: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_attr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_premise[row.get("premise_type", "")].append(row)
        by_attr[row.get("attribute_type", "")].append(row)
        by_source[row.get("source_dataset", "")].append(row)

    true_rows = by_premise.get("true", [])
    false_rows = by_premise.get("false", [])
    parse_fail = [r for r in scored if r["parse_status"] == "failure"]
    ambiguous = [r for r in scored if r["parse_status"] == "ambiguous"]

    acc_tp = safe_div(sum(1 for r in true_rows if r["is_correct"]), len(true_rows))
    acc_fp = safe_div(sum(1 for r in false_rows if r["is_correct"]), len(false_rows))
    sr = safe_div(sum(1 for r in false_rows if r["parse_status"] == "ok" and not r["is_correct"]), len(false_rows))
    orr = safe_div(sum(1 for r in true_rows if r["failure_type"] == "Over-Refusal"), len(true_rows))
    pfr = safe_div(len(parse_fail), total)
    ambiguous_rate = safe_div(len(ambiguous), total)
    hps = 0.0 if acc_tp + acc_fp == 0 else 2 * acc_tp * acc_fp / (acc_tp + acc_fp)

    def breakdown(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(rows),
            "accuracy": safe_div(sum(1 for r in rows if r["is_correct"]), len(rows)),
            "parse_failure_rate": safe_div(sum(1 for r in rows if r["parse_status"] == "failure"), len(rows)),
            "ambiguous_rate": safe_div(sum(1 for r in rows if r["parse_status"] == "ambiguous"), len(rows)),
            "failure_type_distribution": dict(Counter(r["failure_type"] for r in rows)),
        }

    return {
        "scorer_version": SCORER_VERSION,
        "n": total,
        "Acc_TP": acc_tp,
        "Acc_FP": acc_fp,
        "SR": sr,
        "ORR": orr,
        "PFR": pfr,
        "ambiguous_rate": ambiguous_rate,
        "HPS": hps,
        "parse_status_distribution": dict(Counter(r["parse_status"] for r in scored)),
        "parse_method_distribution": dict(Counter(r["parse_method"] for r in scored)),
        "failure_type_distribution": dict(Counter(r["failure_type"] for r in scored)),
        "premise_breakdown": {k: breakdown(v) for k, v in sorted(by_premise.items())},
        "attribute_breakdown": {k: breakdown(v) for k, v in sorted(by_attr.items())},
        "source_breakdown": {k: breakdown(v) for k, v in sorted(by_source.items())},
    }


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_test(output_dir: Path) -> int:
    samples = [
        {
            "id": "tp_json",
            "premise_type": "true",
            "attribute_type": "color",
            "source_dataset": "unit",
            "options": {"A": "red", "B": "pink", "C": "white", "D": "not applicable / no such entity is visible"},
            "answer": "A",
            "raw_output": '{"answer": "A"}',
        },
        {
            "id": "fp_text",
            "premise_type": "false",
            "attribute_type": "location",
            "source_dataset": "unit",
            "options": {"A": "upper region", "B": "not applicable / no such entity is visible", "C": "central region", "D": "lower region"},
            "answer": "B",
            "raw_output": "No such entity is visible.",
        },
        {
            "id": "ambiguous",
            "premise_type": "true",
            "attribute_type": "count",
            "source_dataset": "unit",
            "options": {"A": "one", "B": "two", "C": "three or more", "D": "not applicable / no such entity is visible"},
            "answer": "A",
            "raw_output": "A or B",
        },
        {
            "id": "failure",
            "premise_type": "false",
            "attribute_type": "morphology_type",
            "source_dataset": "unit",
            "options": {"A": "polypoid lesion", "B": "flat lesion", "C": "pedunculated lesion", "D": "not applicable / no such entity is visible"},
            "answer": "D",
            "raw_output": "I cannot tell from this prompt.",
        },
    ]
    scored, metrics = score_rows(samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored, output_dir / "parser_unit_check_v2.jsonl")
    write_metrics(metrics, output_dir / "parser_unit_check_v2_metrics.json")
    expected = {
        "tp_json": ("A", "ok"),
        "fp_text": ("B", "ok"),
        "ambiguous": (None, "ambiguous"),
        "failure": (None, "failure"),
    }
    failures = []
    for row in scored:
        if (row["parsed_answer"], row["parse_status"]) != expected[row["id"]]:
            failures.append(row["id"])
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures, "metrics": metrics}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "outputs": str(output_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.self_test:
        return self_test(args.self_test_output_dir)
    if not args.input or not args.output or not args.metrics_output:
        parser.error("--input, --output, and --metrics-output are required unless --self-test is used")
    rows = read_jsonl(args.input)
    scored, metrics = score_rows(rows)
    write_jsonl(scored, args.output)
    write_metrics(metrics, args.metrics_output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "metrics": str(args.metrics_output), "n": len(scored)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

