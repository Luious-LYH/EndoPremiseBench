#!/usr/bin/env python3
"""Recompute the paper's attribute and filtered-split sensitivity tables.

Inputs are scored primary-split JSONL files, one per model. No model inference
is performed. Outputs contain aggregate percentages only and omit deterministic
reference-policy rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ATTRIBUTES = ("color", "count", "location", "morphology_type", "removal_status")
EXPECTED_SUPPORTED_COUNTS = {
    "color": 27,
    "count": 696,
    "location": 2365,
    "morphology_type": 839,
    "removal_status": 73,
}
EXCLUDED_ATTRIBUTES = {"location", "morphology_type"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity(row: Mapping[str, Any]) -> str:
    value = row.get("id") or row.get("probe_id")
    if not value:
        raise ValueError("each scored row must contain id or probe_id")
    return str(value)


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}"


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    true_rows = [row for row in rows if row.get("premise_type") == "true"]
    false_rows = [row for row in rows if row.get("premise_type") == "false"]
    if not true_rows or not false_rows:
        raise ValueError("both supported and false-premise rows are required")

    acc_tp = safe_div(sum(bool(row.get("is_correct")) for row in true_rows), len(true_rows))
    acc_fp = safe_div(sum(bool(row.get("is_correct")) for row in false_rows), len(false_rows))
    exposure = safe_div(
        sum(row.get("parse_status") == "ok" and not bool(row.get("is_correct")) for row in false_rows),
        len(false_rows),
    )
    over_refusal = safe_div(
        sum(row.get("failure_type") == "Over-Refusal" for row in true_rows), len(true_rows)
    )
    parse_failure = safe_div(
        sum(row.get("parse_status") == "failure" for row in rows), len(rows)
    )
    hps = 2 * acc_tp * acc_fp / (acc_tp + acc_fp) if acc_tp + acc_fp else 0.0
    return {
        "n": len(rows),
        "true_n": len(true_rows),
        "false_n": len(false_rows),
        "Acc_TP": acc_tp,
        "Acc_FP": acc_fp,
        "SR": exposure,
        "ORR": over_refusal,
        "PFR": parse_failure,
        "HPS": hps,
    }


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(model_rows: Sequence[Tuple[str, Path]], output_dir: Path) -> None:
    reference_ids: set[str] | None = None
    attribute_output: List[Dict[str, Any]] = []
    filtered_output: List[Dict[str, Any]] = []

    for model, path in model_rows:
        rows = read_jsonl(path)
        ids = [identity(row) for row in rows]
        if len(rows) != 6000 or len(set(ids)) != 6000:
            raise ValueError(f"{model}: expected 6,000 rows with unique IDs, got {len(rows)}")
        if reference_ids is None:
            reference_ids = set(ids)
        elif set(ids) != reference_ids:
            raise ValueError(f"{model}: scored IDs do not match the other primary runs")

        true_rows = [row for row in rows if row.get("premise_type") == "true"]
        false_rows = [row for row in rows if row.get("premise_type") == "false"]
        if len(true_rows) != 4000 or len(false_rows) != 2000:
            raise ValueError(f"{model}: expected 4,000 supported and 2,000 false-premise rows")

        attr_counts = Counter(str(row.get("attribute_type") or "") for row in true_rows)
        attr_row: Dict[str, Any] = {"model": model, "n_true": len(true_rows)}
        for attribute in ATTRIBUTES:
            selected = [row for row in true_rows if row.get("attribute_type") == attribute]
            if len(selected) != EXPECTED_SUPPORTED_COUNTS[attribute]:
                raise ValueError(
                    f"{model}: {attribute} denominator {len(selected)} does not match the frozen split"
                )
            attr_row[f"{attribute}_n"] = len(selected)
            attr_row[f"{attribute}_accuracy_pct"] = percent(
                safe_div(sum(bool(row.get("is_correct")) for row in selected), len(selected))
            )
        if any(attr_counts.get(attribute, 0) != count for attribute, count in EXPECTED_SUPPORTED_COUNTS.items()):
            raise ValueError(f"{model}: supported attribute counts differ from the frozen split")
        attr_row["overall_supported_accuracy_pct"] = percent(summarize(rows)["Acc_TP"])
        attribute_output.append(attr_row)

        filtered = [row for row in rows if row.get("attribute_type") not in EXCLUDED_ATTRIBUTES]
        metrics = summarize(filtered)
        filtered_output.append(
            {
                "model": model,
                "n": metrics["n"],
                "true_n": metrics["true_n"],
                "false_n": metrics["false_n"],
                "supported_accuracy_pct": percent(metrics["Acc_TP"]),
                "false_premise_rejection_pct": percent(metrics["Acc_FP"]),
                "unsupported_exposure_pct": percent(metrics["SR"]),
                "true_premise_over_refusal_pct": percent(metrics["ORR"]),
                "parse_failure_pct": percent(metrics["PFR"]),
                "balance_score_pct": percent(metrics["HPS"]),
            }
        )

    combined_output: List[Dict[str, Any]] = []
    for attr_row, filtered_row in zip(attribute_output, filtered_output):
        combined = dict(attr_row)
        combined.update({f"filtered_{key}": value for key, value in filtered_row.items() if key != "model"})
        combined_output.append(combined)

    fields = ["model", "n_true"]
    for attribute in ATTRIBUTES:
        fields.extend([f"{attribute}_n", f"{attribute}_accuracy_pct"])
    fields.extend(
        [
            "overall_supported_accuracy_pct", "filtered_n", "filtered_true_n", "filtered_false_n",
            "filtered_supported_accuracy_pct", "filtered_false_premise_rejection_pct",
            "filtered_unsupported_exposure_pct", "filtered_true_premise_over_refusal_pct",
            "filtered_parse_failure_pct", "filtered_balance_score_pct",
        ]
    )
    write_csv(
        output_dir / "rebuttal_sensitivity.csv",
        fields,
        combined_output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME=PATH",
        help="scored 6,000-row primary JSONL; repeat once per model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_rows: List[Tuple[str, Path]] = []
    for item in args.model:
        if "=" not in item:
            parser.error(f"invalid --model {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        model_rows.append((name, Path(raw_path)))
    if len({name for name, _ in model_rows}) != len(model_rows):
        parser.error("model names must be unique")

    analyze(model_rows, args.output_dir)
    print(json.dumps({"models": len(model_rows), "output_dir": args.output_dir.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
