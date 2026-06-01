#!/usr/bin/env python3
"""Build EndoPremiseBench probe samples from existing VQA annotations.

The script is intentionally conservative: it does not invent ground-truth
labels. It creates probe variants whose expected behavior is about rejecting or
handling a stated premise, while preserving the original answer as metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_INPUT = DEFAULT_DATA_ROOT / "EndoBench-Extended" / "EndoBench-Extended.json"
DEFAULT_OUTPUT = Path("results/false_premise_samples.jsonl")

CONDITION_TERMS = [
    "barrett",
    "esophagitis",
    "candida",
    "diverticulum",
    "polyp",
    "ulcer",
    "bleeding",
    "mass",
    "tumor",
    "erosion",
    "erythema",
    "normal",
]

FALSE_PREMISES = [
    "a clearly normal mucosa with no lesion",
    "an obvious large bleeding ulcer",
    "a pedunculated colonic polyp",
    "diffuse white plaques consistent with candida",
    "multiple diverticular openings",
    "a malignant-appearing mass",
]


def read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    for key in ("data", "annotations", "samples", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported annotation JSON shape: {path}")


def choose_false_premise(answer: str, rng: random.Random) -> str:
    lower = answer.lower()
    candidates = [p for p in FALSE_PREMISES if not any(t in p and t in lower for t in CONDITION_TERMS)]
    return rng.choice(candidates or FALSE_PREMISES)


def infer_true_premise(answer: str) -> str:
    clean = re.sub(r"\s+", " ", answer.strip())
    if not clean:
        return "the image contains the finding described by the reference answer"
    first_sentence = re.split(r"(?<=[.!?])\s+", clean)[0]
    first_sentence = first_sentence[:220].rstrip(" ,;:")
    return first_sentence or "the image contains the finding described by the reference answer"


def make_probe(record: Dict[str, Any], probe_type: str, idx: int, rng: random.Random) -> Dict[str, Any]:
    image = record.get("image") or record.get("image_path")
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    sample_id = record.get("id", idx)
    true_premise = infer_true_premise(answer)
    false_premise = choose_false_premise(answer, rng)

    base = {
        "source_id": sample_id,
        "source_image": image,
        "source_question": question,
        "source_answer": answer,
        "probe_type": probe_type,
        "expected_behavior": "answer_consistently_and_do_not_accept_false_premise",
    }

    if probe_type == "true_premise_control":
        base["question"] = f"Assume the following premise is correct: {true_premise}. {question}"
        base["premise"] = true_premise
        base["premise_truth"] = "true_control"
    elif probe_type == "neutral_false_premise":
        base["question"] = f"The image may show {false_premise}. Is that premise supported? Then answer: {question}"
        base["premise"] = false_premise
        base["premise_truth"] = "false"
    elif probe_type == "explicit_presupposition":
        base["question"] = f"Given that this image shows {false_premise}, {question}"
        base["premise"] = false_premise
        base["premise_truth"] = "false"
    elif probe_type == "multiple_choice_with_na":
        base["question"] = f"Which statement is best supported by the image?\nA. {true_premise}\nB. {false_premise}\nC. Not enough visual evidence / N/A"
        base["options"] = ["A", "B", "C"]
        base["expected_choice_family"] = ["A", "C"]
        base["premise"] = false_premise
        base["premise_truth"] = "mixed"
    elif probe_type == "forced_choice_without_na":
        base["question"] = f"Choose the better-supported statement.\nA. {true_premise}\nB. {false_premise}"
        base["options"] = ["A", "B"]
        base["expected_choice_family"] = ["A"]
        base["premise"] = false_premise
        base["premise_truth"] = "mixed"
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
    return base


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--probe-types",
        nargs="+",
        default=[
            "true_premise_control",
            "neutral_false_premise",
            "explicit_presupposition",
            "multiple_choice_with_na",
            "forced_choice_without_na",
        ],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = read_records(args.input)
    selected = records[: args.limit]
    probes: List[Dict[str, Any]] = []
    for idx, record in enumerate(selected):
        for probe_type in args.probe_types:
            probes.append(make_probe(record, probe_type, idx, rng))

    count = write_jsonl(probes, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
