#!/usr/bin/env python3
"""Analyze 2x2 crossed premise-control outputs with block-clustered CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LETTERS = {"A", "B", "C", "D"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_compatible(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_compatible(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def parse_letter(row: Mapping[str, Any]) -> str | None:
    for key in (
        "parsed_answer",
        "parsed_letter",
        "prediction",
        "predicted_answer",
        "selected_option",
        "answer_letter",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip().upper() in LETTERS:
            return value.strip().upper()
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        for key in ("answer", "letter", "prediction"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip().upper() in LETTERS:
                return value.strip().upper()
    return None


def group_by(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return groups


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows or len(rows) % 4 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("manifest must contain a nonempty set of unique four-cell blocks")
    if len(group_by(rows, "block_id")) != len(rows) // 4 or len(group_by(rows, "query_id")) != len(rows) // 2:
        raise ValueError("manifest block/query cardinality is inconsistent")
    for block_id, block_rows in group_by(rows, "block_id").items():
        if len(block_rows) != 4:
            raise ValueError(f"{block_id}: expected four rows")
        if Counter(row["premise_type"] for row in block_rows) != {"true": 2, "false": 2}:
            raise ValueError(f"{block_id}: invalid premise crossover")
    for query_id, query_rows in group_by(rows, "query_id").items():
        if len(query_rows) != 2 or Counter(row["premise_type"] for row in query_rows) != {"true": 1, "false": 1}:
            raise ValueError(f"{query_id}: invalid query crossover")
        for key in ("question", "options", "target_entity", "attribute_type", "na_letter"):
            if query_rows[0][key] != query_rows[1][key]:
                raise ValueError(f"{query_id}: immutable mismatch for {key}")


def bind_predictions(
    manifest: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    raw_by_id = {str(row.get("id") or row.get("probe_id") or ""): row for row in raw_rows}
    if len(raw_rows) != len(raw_by_id) or set(raw_by_id) != {str(row["id"]) for row in manifest}:
        raise ValueError("raw output IDs do not exactly match the manifest")
    scored = []
    for gold in manifest:
        raw = raw_by_id[str(gold["id"])]
        prediction = parse_letter(raw)
        row = dict(gold)
        row["prediction"] = prediction
        row["parse_status"] = raw.get("parse_status") or ("ok" if prediction else "failure")
        row["is_na"] = prediction == row["na_letter"]
        row["is_exact"] = prediction == row["answer"]
        row["is_concrete"] = prediction in LETTERS and prediction != row["na_letter"]
        scored.append(row)
    return scored


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    supported = [row for row in rows if row["premise_type"] == "true"]
    unsupported = [row for row in rows if row["premise_type"] == "false"]
    query_pairs = list(group_by(rows, "query_id").values())
    blocks = list(group_by(rows, "block_id").values())
    supported_exact = safe_mean([float(row["is_exact"]) for row in supported])
    supported_na = safe_mean([float(row["is_na"]) for row in supported])
    unsupported_rejection = safe_mean([float(row["is_na"]) for row in unsupported])
    contrast = unsupported_rejection - supported_na
    joint = []
    conditional = []
    for pair in query_pairs:
        true_row = next(row for row in pair if row["premise_type"] == "true")
        false_row = next(row for row in pair if row["premise_type"] == "false")
        joint.append(float(true_row["is_exact"] and false_row["is_na"]))
        if true_row["is_exact"]:
            conditional.append(float(false_row["is_na"]))
    strict_blocks = []
    for block in blocks:
        true_ok = all(row["is_exact"] for row in block if row["premise_type"] == "true")
        false_ok = all(row["is_na"] for row in block if row["premise_type"] == "false")
        strict_blocks.append(float(true_ok and false_ok))
    return {
        "n_rows": float(len(rows)),
        "n_blocks": float(len(blocks)),
        "n_supported": float(len(supported)),
        "n_unsupported": float(len(unsupported)),
        "supported_exact_accuracy": supported_exact,
        "supported_na_rate": supported_na,
        "unsupported_rejection": unsupported_rejection,
        "unsupported_concrete_exposure": safe_mean([float(row["is_concrete"]) for row in unsupported]),
        "premise_sensitivity_contrast": contrast,
        "premise_status_balanced_accuracy": (1.0 + contrast) / 2.0,
        "query_pair_joint_exact": safe_mean(joint),
        "conditional_rejection_given_supported_exact": safe_mean(conditional),
        "conditional_denominator": float(len(conditional)),
        "strict_block_success": safe_mean(strict_blocks),
        "parse_failure_rate": safe_mean([float(row["prediction"] is None) for row in rows]),
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], samples: int, seed: int
) -> Dict[str, Dict[str, float]]:
    by_block = group_by(rows, "block_id")
    block_ids = sorted(by_block)
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for _ in range(samples):
        sampled_rows = []
        for index in range(len(block_ids)):
            chosen = rng.choice(block_ids)
            for row in by_block[chosen]:
                copied = dict(row)
                copied["block_id"] = f"boot_{index}"
                copied["query_id"] = f"boot_{index}::{row['query_role']}"
                sampled_rows.append(copied)
        result = compute_metrics(sampled_rows)
        for metric in metrics:
            draws[metric].append(result[metric])
    return {
        metric: {"low": percentile(values, 0.025), "high": percentile(values, 0.975)}
        for metric, values in draws.items()
    }


def exact_mcnemar(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    supported_na_only = 0
    unsupported_na_only = 0
    for pair in group_by(rows, "query_id").values():
        true_row = next(row for row in pair if row["premise_type"] == "true")
        false_row = next(row for row in pair if row["premise_type"] == "false")
        if true_row["is_na"] and not false_row["is_na"]:
            supported_na_only += 1
        elif false_row["is_na"] and not true_row["is_na"]:
            unsupported_na_only += 1
    n = supported_na_only + unsupported_na_only
    if n == 0:
        p_value = 1.0
    else:
        tail = min(supported_na_only, unsupported_na_only)
        one_sided = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
        p_value = min(1.0, 2.0 * one_sided)
    return {
        "supported_na_only": float(supported_na_only),
        "unsupported_na_only": float(unsupported_na_only),
        "exact_two_sided_p": p_value,
    }


def score_fixed_policy(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> List[Dict[str, Any]]:
    scored = []
    for gold in rows:
        if policy == "always_one":
            matches = [
                letter
                for letter, option in gold["options"].items()
                if str(option).strip().lower() == "one"
            ]
            if len(matches) != 1:
                raise ValueError(f"{gold['id']}: expected exactly one 'one' option")
            prediction = matches[0]
        elif policy == "always_na":
            prediction = str(gold["na_letter"])
        elif policy == "always_A":
            prediction = "A"
        else:
            raise ValueError(f"unknown reference policy: {policy}")
        row = dict(gold)
        row["prediction"] = prediction
        row["parse_status"] = "ok"
        row["is_na"] = prediction == row["na_letter"]
        row["is_exact"] = prediction == row["answer"]
        row["is_concrete"] = prediction != row["na_letter"]
        scored.append(row)
    return scored


def compute_reference_policies(
    manifest: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    count_rows = [row for row in manifest if row["attribute_type"] == "count"]
    return {
        policy: json_compatible(compute_metrics(score_fixed_policy(count_rows, policy)))
        for policy in ("always_one", "always_na", "always_A")
    }


def format_pct(value: float) -> str:
    return "--" if math.isnan(value) else f"{100 * value:.1f}"


def format_ci(point: float, interval: Mapping[str, float]) -> str:
    return f"{format_pct(point)} [{format_pct(interval['low'])}, {format_pct(interval['high'])}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="MODEL_NAME=RAW_JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    validate_manifest(manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "analysis_config.json",
        {
            "schema": "atomic_crossover_analysis_config_v1",
            "manifest": args.manifest.name,
            "models": [specification.split("=", 1)[0] for specification in args.model],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "adjudicated block",
            "confidence_interval": "percentile 95%",
            "primary_attribute_stratum": "count",
        },
    )
    ci_metrics = (
        "supported_exact_accuracy",
        "supported_na_rate",
        "unsupported_rejection",
        "premise_sensitivity_contrast",
        "premise_status_balanced_accuracy",
        "query_pair_joint_exact",
        "conditional_rejection_given_supported_exact",
        "strict_block_success",
    )
    results: Dict[str, Any] = {}
    count_mcnemar: Dict[str, Any] = {}
    table_rows = []
    for specification in args.model:
        name, raw_path = specification.split("=", 1)
        scored = bind_predictions(manifest, read_jsonl(Path(raw_path)))
        count_scored = [row for row in scored if row["attribute_type"] == "count"]
        metrics = compute_metrics(scored)
        cis = bootstrap(scored, ci_metrics, args.bootstrap_samples, args.seed)
        attribute_strata = {
            key: compute_metrics(value)
            for key, value in sorted(group_by(scored, "attribute_type").items())
        }
        attribute_strata_ci = {
            key: bootstrap(value, ci_metrics, args.bootstrap_samples, args.seed + index + 1)
            for index, (key, value) in enumerate(sorted(group_by(scored, "attribute_type").items()))
        }
        entity_attribute_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in scored:
            entity_attribute_rows[f"{row['target_entity']}|{row['attribute_type']}"].append(row)
        entity_attribute_strata = {
            key: compute_metrics(value) for key, value in sorted(entity_attribute_rows.items())
        }
        results[name] = {
            "metrics": metrics,
            "clustered_bootstrap_95ci": cis,
            "mcnemar_na_decision": exact_mcnemar(scored),
            "attribute_strata": attribute_strata,
            "attribute_strata_clustered_bootstrap_95ci": attribute_strata_ci,
            "entity_attribute_strata": entity_attribute_strata,
        }
        count_mcnemar[name] = exact_mcnemar(count_scored)
        count_metrics = attribute_strata["count"]
        count_intervals = attribute_strata_ci["count"]
        table_rows.append(
            {
                "Model": name,
                "Supported exact": format_ci(count_metrics["supported_exact_accuracy"], count_intervals["supported_exact_accuracy"]),
                "Supported N/A": format_ci(count_metrics["supported_na_rate"], count_intervals["supported_na_rate"]),
                "Unsupported rejection": format_ci(count_metrics["unsupported_rejection"], count_intervals["unsupported_rejection"]),
                "Premise contrast": format_ci(count_metrics["premise_sensitivity_contrast"], count_intervals["premise_sensitivity_contrast"]),
                "Status BA": format_ci(count_metrics["premise_status_balanced_accuracy"], count_intervals["premise_status_balanced_accuracy"]),
                "Query-pair joint": format_ci(count_metrics["query_pair_joint_exact"], count_intervals["query_pair_joint_exact"]),
                "Strict block": format_ci(count_metrics["strict_block_success"], count_intervals["strict_block_success"]),
                "Conditional rejection": format_ci(count_metrics["conditional_rejection_given_supported_exact"], count_intervals["conditional_rejection_given_supported_exact"]),
                "Conditional n": str(int(count_metrics["conditional_denominator"])),
            }
        )

    write_json(output_dir / "crossed_metrics.json", results)
    write_json(output_dir / "count_mcnemar.json", count_mcnemar)
    columns = list(table_rows[0])
    with (output_dir / "crossed_main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table_rows)
    markdown = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"]
    markdown += ["| " + " | ".join(row[column] for column in columns) + " |" for row in table_rows]
    (output_dir / "crossed_main_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
