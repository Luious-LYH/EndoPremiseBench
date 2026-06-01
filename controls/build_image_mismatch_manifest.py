#!/usr/bin/env python3
"""Build the frozen 20260525 formal image-mismatch manifest.

The historical 20260524 shuffled-image control remains useful provenance, but
the formal 20260525 appendix matrix gets its own manifest, seed, report, and
root so it cannot be accidentally mixed with prior or secondary runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
DEFAULT_INPUT = PROJECT_ROOT / "results/premise_balanced_main_v2.jsonl"
DEFAULT_ROOT = PROJECT_ROOT / (
    "results/a_group_supplement_20260524/analysis/"
    "image_mismatch_control_20260525/manifests"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_id(row: Dict[str, Any]) -> str:
    for key in ("probe_id", "id", "sample_id"):
        if row.get(key):
            return str(row[key])
    sample = row.get("sample")
    if isinstance(sample, dict):
        for key in ("probe_id", "id", "sample_id"):
            if sample.get(key):
                return str(sample[key])
    return ""


def stable_hash(seed: int, *parts: Any) -> str:
    text = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def allocate_counts(group_sizes: Dict[Tuple[str, str], int], target: int) -> Dict[Tuple[str, str], int]:
    total = sum(group_sizes.values())
    if target <= 0 or target >= total:
        return dict(group_sizes)
    raw = {key: value * target / total for key, value in group_sizes.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = target - sum(counts.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - counts[key], group_sizes[key], str(key)), reverse=True)
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def subset_rows(rows: Sequence[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    if target <= 0 or target >= len(rows):
        return list(rows)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("source_dataset") or ""), str(row.get("premise_type") or ""))].append(row)
    counts = allocate_counts({key: len(value) for key, value in groups.items()}, target)
    selected_ids = set()
    for key, group_rows in groups.items():
        ranked = sorted(group_rows, key=lambda row: stable_hash(seed, "subset", row_id(row)))
        for row in ranked[: counts[key]]:
            selected_ids.add(row_id(row))
    return [row for row in rows if row_id(row) in selected_ids]


def initial_donor_order(n: int, seed: int, group_key: str) -> List[int]:
    if n <= 1:
        return list(range(n))
    ranked = sorted(range(n), key=lambda idx: stable_hash(seed, group_key, idx))
    return ranked[1:] + ranked[:1]


def donor_order_without_same_image(rows: Sequence[Dict[str, Any]], seed: int, group_key: str) -> List[int]:
    """Return a deterministic donor order, repairing same-id/image matches if possible."""

    n = len(rows)
    order = initial_donor_order(n, seed, group_key)
    if n <= 1:
        return order

    def bad_at(idx: int, candidate_order: Sequence[int]) -> bool:
        donor = rows[candidate_order[idx]]
        row = rows[idx]
        return row_id(donor) == row_id(row) or donor.get("image_path") == row.get("image_path")

    # Greedy pair swaps. This keeps the construction deterministic while
    # avoiding expensive search over large source groups.
    for _ in range(n * 2):
        bad_indices = [idx for idx in range(n) if bad_at(idx, order)]
        if not bad_indices:
            break
        idx = bad_indices[0]
        swapped = False
        ranked_swaps = sorted(range(n), key=lambda j: stable_hash(seed, "swap", group_key, idx, j))
        for j in ranked_swaps:
            if j == idx:
                continue
            trial = list(order)
            trial[idx], trial[j] = trial[j], trial[idx]
            if not bad_at(idx, trial) and not bad_at(j, trial):
                order = trial
                swapped = True
                break
        if not swapped:
            break
    return order


def make_image_mismatch(rows: Sequence[Dict[str, Any]], seed: int, scope: str) -> List[Dict[str, Any]]:
    if scope != "same_source":
        raise ValueError("Only same_source is allowed for the formal 20260525 manifest")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source_dataset") or "")].append(row)

    by_base_id: Dict[str, Dict[str, Any]] = {}
    for group_key, group_rows in groups.items():
        order = donor_order_without_same_image(group_rows, seed, group_key)
        for idx, donor_idx in enumerate(order):
            row = group_rows[idx]
            donor = group_rows[donor_idx]
            base = row_id(row)
            out = dict(row)
            out["id"] = f"{base}::image_mismatch_same_source_seed{seed}"
            out["probe_id"] = out["id"]
            out["base_id"] = base
            out["source_probe_id"] = base
            out["original_image_path"] = row.get("image_path")
            out["image_path"] = donor.get("image_path")
            out["mismatched_from_id"] = row_id(donor)
            out["mismatched_from_source_dataset"] = donor.get("source_dataset")
            out["image_mismatch_kind"] = scope
            out["image_mismatch_seed"] = seed
            out["control_kind"] = "image_mismatch"
            out["selection_seed"] = seed
            by_base_id[base] = out
    return [by_base_id[row_id(row)] for row in rows]


def profile(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    fields = ["source_dataset", "premise_type", "attribute_type", "control_kind"]
    result: Dict[str, Dict[str, int]] = {}
    for field in fields:
        counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            counts[str(row.get(field) or "")] += 1
        result[field] = dict(sorted(counts.items()))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--scope", choices=["same_source"], default="same_source")
    parser.add_argument("--subset-size", type=int, default=2000)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    selected = subset_rows(rows, args.subset_size, args.seed)
    mismatched = make_image_mismatch(selected, args.seed, args.scope)

    manifest_path = args.output_root / f"premise_main{args.subset_size}_image_mismatch_{args.scope}_seed{args.seed}.jsonl"
    report_path = args.output_root / f"image_mismatch_report_{args.scope}_seed{args.seed}.json"
    n = write_jsonl(manifest_path, mismatched)
    same_image = sum(1 for row in mismatched if row.get("image_path") == row.get("original_image_path"))
    same_id = sum(1 for row in mismatched if str(row.get("mismatched_from_id") or "") == str(row.get("base_id") or ""))
    cross_source = sum(
        1
        for row in mismatched
        if str(row.get("source_dataset") or "") != str(row.get("mismatched_from_source_dataset") or "")
    )
    report = {
        "input": args.input.as_posix(),
        "input_sha256": sha256_file(args.input),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "scope": args.scope,
        "seed": args.seed,
        "subset_size": args.subset_size,
        "n": n,
        "same_image_after_mismatch": same_image,
        "same_id_after_mismatch": same_id,
        "cross_source_donors": cross_source,
        "construction_rule": (
            "Select the 2000-row subset before inference by source_dataset x premise_type "
            "proportional allocation using SHA256(seed,id), then replace each image_path "
            "with a deterministic same-source donor image while preserving question, options, "
            "gold answer, and labels. Rows with identical original and donor images are audited "
            "and disallowed for the formal PASS gate."
        ),
        "profile": profile(mismatched),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if same_image == 0 and same_id == 0 and cross_source == 0 and n == args.subset_size else 1


if __name__ == "__main__":
    raise SystemExit(main())
