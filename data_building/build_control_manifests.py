#!/usr/bin/env python3
"""Build EndoPremiseBench control manifests for N/A position and wording ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


LETTERS = ["A", "B", "C", "D"]
NA_TEXT = "not applicable / no such entity is visible"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def is_na_option(text: str) -> bool:
    lowered = str(text or "").lower()
    return "not applicable" in lowered or "no such entity" in lowered


def reorder_na(row: Dict[str, Any], na_letter: str) -> Dict[str, Any]:
    options = row.get("options") or {}
    na_value = None
    distractors: List[str] = []
    for letter in LETTERS:
        value = options.get(letter)
        if is_na_option(value):
            na_value = value
        elif value is not None:
            distractors.append(value)
    if na_value is None:
        na_value = NA_TEXT
    out_options: Dict[str, str] = {}
    distractor_iter = iter(distractors)
    for letter in LETTERS:
        if letter == na_letter:
            out_options[letter] = na_value
        else:
            out_options[letter] = next(distractor_iter)
    item = dict(row)
    item["id"] = f"{row.get('id')}::na_pos_{na_letter}"
    item["probe_id"] = item["id"]
    item["options"] = out_options
    item["answer"] = na_letter
    item["control_kind"] = "na_position_permutation"
    item["control_value"] = na_letter
    item["base_id"] = row.get("id")
    return item


def wording_question(row: Dict[str, Any], variant: str) -> str:
    question = str(row.get("question") or "").strip()
    entity = str(row.get("target_entity") or "the queried entity").strip()
    if variant == "neutral":
        return f"Based on the image, answer this attribute question: {question}"
    if variant == "explicit":
        return f"Assume the {entity} is present in the image. {question}"
    if variant == "guarded":
        return (
            f"First verify whether the {entity} is visible in the image. "
            f"If it is not visible, choose the not-applicable option. If it is visible, answer: {question}"
        )
    raise ValueError(f"unknown wording variant: {variant}")


def wording_variant(row: Dict[str, Any], variant: str) -> Dict[str, Any]:
    item = dict(row)
    item["id"] = f"{row.get('id')}::wording_{variant}"
    item["probe_id"] = item["id"]
    item["question"] = wording_question(row, variant)
    item["control_kind"] = "wording_control"
    item["control_value"] = variant
    item["base_id"] = row.get("id")
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/premise_balanced_main_v2.jsonl"))
    parser.add_argument("--out-na", type=Path, default=Path("results/premise_false2000_na_position_all_v1.jsonl"))
    parser.add_argument("--out-wording", type=Path, default=Path("results/premise_false2000_wording_controls_v1.jsonl"))
    parser.add_argument("--out-report", type=Path, default=Path("tables/control_manifest_report_v1.md"))
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    false_rows = [row for row in rows if row.get("premise_type") == "false"]
    na_rows = [reorder_na(row, letter) for row in false_rows for letter in LETTERS]
    wording_rows = [wording_variant(row, variant) for row in false_rows for variant in ["neutral", "explicit", "guarded"]]
    na_count = write_jsonl(na_rows, args.out_na)
    wording_count = write_jsonl(wording_rows, args.out_wording)

    report = [
        "# EndoPremiseBench Control Manifest Report v1",
        "",
        f"Base false-premise rows: `{len(false_rows)}`",
        "",
        "| Manifest | Rows | Purpose |",
        "|---|---:|---|",
        f"| `{args.out_na}` | {na_count} | Paired N/A-position control with all A/B/C/D positions for each false-premise row. |",
        f"| `{args.out_wording}` | {wording_count} | False-premise wording controls: neutral, explicit, guarded. |",
        "",
        "Run note: smoke each manifest before full inference. The N/A-position manifest is paired and larger than false2000 because it contains four variants per base row.",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"false_rows": len(false_rows), "na_rows": na_count, "wording_rows": wording_count, "report": str(args.out_report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

