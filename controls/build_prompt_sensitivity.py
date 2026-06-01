#!/usr/bin/env python3
"""Freeze P1 prompt-sensitivity subset and prompt templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
INPUT = PROJECT_ROOT / "results/premise_balanced_main_v2.jsonl"
OUT_ROOT = PROJECT_ROOT / "results/a_group_supplement_20260524/analysis/prompt_sensitivity_20260525"
SEED = 20260525
SUBSET_SIZE = 2000

PROMPTS = {
    "strict_json": """Answer the question based on the image.
Choose exactly one option: A, B, C, or D.
Return JSON only: {"answer": "<A/B/C/D>"}""",
    "answer_only": """Answer the question based on the image.
Choose exactly one option.
Respond with only the option letter: A, B, C, or D.
Do not include explanation or JSON.""",
    "rationale_then_json": """Answer the question based on the image.
First write one short sentence explaining the visual evidence.
Then return a JSON object on the final line: {"answer": "<A/B/C/D>"}.
Choose exactly one option: A, B, C, or D.""",
    "premise_guarded": """Answer the question based on the image.
Before choosing, check whether the question's visual premise is supported by the image.
If the relevant entity or attribute is not visible or not supported, choose the not-applicable option.
Otherwise choose the best supported option.
Return JSON only: {"answer": "<A/B/C/D>"}""",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_key(seed: int, row: Dict[str, Any]) -> str:
    item_id = str(row.get("id") or row.get("probe_id") or "")
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def allocate(group_sizes: Dict[Tuple[str, str], int], total: int) -> Dict[Tuple[str, str], int]:
    available = sum(group_sizes.values())
    if total <= 0 or total >= available:
        return dict(group_sizes)
    raw = {key: value * total / available for key, value in group_sizes.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - counts[key], group_sizes[key]), reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def subset(rows: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("source_dataset") or ""), str(row.get("premise_type") or ""))].append(row)
    counts = allocate({key: len(value) for key, value in groups.items()}, size)
    selected_ids = set()
    for key, group_rows in groups.items():
        chosen = sorted(group_rows, key=lambda row: stable_key(seed, row))[: counts[key]]
        selected_ids.update(str(row.get("id") or row.get("probe_id") or "") for row in chosen)
    return [row for row in rows if str(row.get("id") or row.get("probe_id") or "") in selected_ids]


def profile(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for key in ("source_dataset", "premise_type", "attribute_type"):
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row.get(key) or "")] += 1
        out[key] = dict(sorted(counts.items()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    chosen = subset(rows, args.subset_size, args.seed)
    manifest = args.out_root / "manifests" / f"prompt_sensitivity_subset{len(chosen)}_seed{args.seed}.jsonl"
    count = write_jsonl(manifest, chosen)

    prompt_dir = args.out_root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths: Dict[str, str] = {}
    for name, text in PROMPTS.items():
        path = prompt_dir / f"{name}.txt"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        prompt_paths[name] = str(path)

    report = {
        "input": str(args.input),
        "manifest": str(manifest),
        "subset_size": count,
        "selection_seed": args.seed,
        "selection_rule": "deterministic SHA256(seed,id) within source_dataset x premise_type proportional allocation",
        "profile": profile(chosen),
        "prompt_paths": prompt_paths,
    }
    report_path = args.out_root / "manifests" / f"prompt_sensitivity_subset{len(chosen)}_seed{args.seed}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
