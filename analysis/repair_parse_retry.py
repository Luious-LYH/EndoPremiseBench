#!/usr/bin/env python3
"""Build and finalize parse-retry repairs for EndoPremiseBench full10000.

The repair protocol is row-index based because full10000 intentionally contains
two duplicate ids. Original artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
DEFAULT_EXP = PROJECT_ROOT / (
    "results/a_group_supplement_20260524/"
    "analysis/expanded_kvasir_consistency_20260526"
)
MANIFEST_NAME = "kvasir_expanded_source_size_full10000_seed2026052602.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("id") or "")


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_root(exp: Path, slug: str) -> Path:
    return exp / "repair_parse_retry_20260529" / slug


def attempt_dir(exp: Path, slug: str, attempt: int) -> Path:
    return repair_root(exp, slug) / f"attempt{attempt:02d}"


def original_paths(exp: Path, slug: str) -> Dict[str, Path]:
    return {
        "raw": exp / "raw" / f"{slug}_full10000_raw.jsonl",
        "scored": exp / "scored" / f"{slug}_full10000_scored.jsonl",
        "metrics": exp / "metrics" / f"{slug}_full10000_metrics.json",
    }


def attempt_paths(exp: Path, slug: str, attempt: int) -> Dict[str, Path]:
    root = attempt_dir(exp, slug, attempt)
    return {
        "manifest": root / "manifest" / f"{slug}_repair_attempt{attempt:02d}_manifest.jsonl",
        "raw": root / "raw" / f"{slug}_repair_attempt{attempt:02d}_raw.jsonl",
        "scored": root / "scored" / f"{slug}_repair_attempt{attempt:02d}_scored.jsonl",
        "metrics": root / "metrics" / f"{slug}_repair_attempt{attempt:02d}_metrics.json",
        "status": root / "status" / f"{slug}_repair_attempt{attempt:02d}_status.json",
    }


def load_scored_for_attempt_source(exp: Path, slug: str, attempt: int) -> List[Dict[str, Any]]:
    if attempt == 1:
        return read_jsonl(original_paths(exp, slug)["scored"])
    return read_jsonl(attempt_paths(exp, slug, attempt - 1)["scored"])


def prepare_cmd(args: argparse.Namespace) -> None:
    exp = args.exp_root
    slug = args.slug
    manifest_rows = read_jsonl(exp / "manifest" / MANIFEST_NAME)
    source_rows = load_scored_for_attempt_source(exp, slug, args.attempt)
    selected: List[Dict[str, Any]] = []
    original_status_counts: Dict[str, int] = {}
    for source_pos, scored in enumerate(source_rows):
        status = str(scored.get("parse_status") or "missing")
        if status == "ok":
            continue
        if status == "ambiguous" and not args.include_ambiguous:
            continue
        if args.attempt == 1:
            row_index = source_pos
        else:
            row_index = int(scored.get("repair_row_index"))
        if row_index < 0 or row_index >= len(manifest_rows):
            raise SystemExit(f"repair_row_index out of range: {row_index}")
        base = dict(manifest_rows[row_index])
        original_id = str(base.get("id") or "")
        base["original_id"] = original_id
        base["original_probe_id"] = original_id
        base["repair_row_index"] = row_index
        base["repair_attempt"] = args.attempt
        base["repair_source_parse_status"] = status
        base["repair_source_parsed_answer"] = scored.get("parsed_answer")
        base["repair_protocol"] = "parse_retry_max3_row_index_20260529"
        base["id"] = f"{original_id}__repair_row_{row_index:05d}__attempt{args.attempt:02d}"
        selected.append(base)
        original_status_counts[status] = original_status_counts.get(status, 0) + 1
    paths = attempt_paths(exp, slug, args.attempt)
    write_jsonl(selected, paths["manifest"])
    status = {
        "status": "prepared",
        "slug": slug,
        "attempt": args.attempt,
        "include_ambiguous": args.include_ambiguous,
        "manifest": str(paths["manifest"]),
        "selected_rows": len(selected),
        "source_status_counts": original_status_counts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["status"].parent.mkdir(parents=True, exist_ok=True)
    paths["status"].write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


def split_cmd(args: argparse.Namespace) -> None:
    paths = attempt_paths(args.exp_root, args.slug, args.attempt)
    rows = read_jsonl(paths["manifest"])
    out_dir = attempt_dir(args.exp_root, args.slug, args.attempt) / "manifest_shards"
    counts = []
    for shard in range(args.num_shards):
        part = [row for idx, row in enumerate(rows) if idx % args.num_shards == shard]
        out = out_dir / f"{args.slug}_repair_attempt{args.attempt:02d}_shard{shard:03d}_of{args.num_shards}_manifest.jsonl"
        write_jsonl(part, out)
        counts.append(len(part))
    print(json.dumps({"slug": args.slug, "attempt": args.attempt, "num_shards": args.num_shards, "counts": counts}, ensure_ascii=False))


def merge_shards_cmd(args: argparse.Namespace) -> None:
    paths = attempt_paths(args.exp_root, args.slug, args.attempt)
    root = attempt_dir(args.exp_root, args.slug, args.attempt)
    rows = []
    shard_info = []
    for shard in range(args.num_shards):
        raw = root / "raw_shards" / f"{args.slug}_repair_attempt{args.attempt:02d}_shard{shard:03d}_of{args.num_shards}_raw.jsonl"
        part = read_jsonl(raw)
        rows.extend(part)
        shard_info.append({"shard": shard, "rows": len(part), "raw": str(raw), "sha256": sha256(raw)})
    # Restore manifest order by repair_row_index; ThreadPool writes API rows as they finish.
    rows.sort(key=lambda row: int(row.get("repair_row_index", -1)))
    write_jsonl(rows, paths["raw"])
    payload = {
        "status": "merged",
        "slug": args.slug,
        "attempt": args.attempt,
        "rows": len(rows),
        "raw": str(paths["raw"]),
        "raw_sha256": sha256(paths["raw"]),
        "shards": shard_info,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["status"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def finalize_cmd(args: argparse.Namespace) -> None:
    exp = args.exp_root
    slug = args.slug
    original_raw = read_jsonl(original_paths(exp, slug)["raw"])
    original_scored = read_jsonl(original_paths(exp, slug)["scored"])
    final_rows = [dict(row) for row in original_raw]
    replacements: Dict[int, Dict[str, Any]] = {}
    replacement_meta: Dict[int, Dict[str, Any]] = {}
    attempt_reports = []
    for attempt in range(1, args.max_attempts + 1):
        paths = attempt_paths(exp, slug, attempt)
        raw_rows = read_jsonl(paths["raw"])
        scored_rows = read_jsonl(paths["scored"])
        by_idx = {int(row.get("repair_row_index")): row for row in raw_rows if row.get("repair_row_index") is not None}
        counts: Dict[str, int] = {}
        newly_ok = 0
        for scored in scored_rows:
            status = str(scored.get("parse_status") or "missing")
            counts[status] = counts.get(status, 0) + 1
            if status != "ok":
                continue
            idx = int(scored["repair_row_index"])
            if idx in replacements:
                continue
            raw = dict(by_idx[idx])
            orig = original_raw[idx]
            raw["id"] = orig.get("id")
            raw["probe_id"] = orig.get("probe_id") or orig.get("id")
            raw["original_parse_status"] = original_scored[idx].get("parse_status")
            raw["original_parse_method"] = original_scored[idx].get("parse_method")
            raw["repair_success_attempt"] = attempt
            raw["repair_protocol"] = "parse_retry_max3_row_index_20260529"
            replacements[idx] = raw
            replacement_meta[idx] = {
                "row_index": idx,
                "id": orig.get("id"),
                "original_parse_status": original_scored[idx].get("parse_status"),
                "success_attempt": attempt,
            }
            newly_ok += 1
        attempt_reports.append({"attempt": attempt, "raw_rows": len(raw_rows), "scored_rows": len(scored_rows), "parse_status_counts": counts, "newly_ok": newly_ok})
    for idx, row in replacements.items():
        final_rows[idx] = row
    out_raw = exp / "raw" / f"{slug}_full10000_parse_repaired3_raw.jsonl"
    write_jsonl(final_rows, out_raw)
    remaining = []
    for idx, scored in enumerate(original_scored):
        if scored.get("parse_status") == "ok":
            continue
        if idx not in replacements:
            remaining.append({"row_index": idx, "id": original_raw[idx].get("id"), "original_parse_status": scored.get("parse_status")})
    audit = {
        "status": "finalized_raw",
        "slug": slug,
        "protocol": "parse_retry_max3_row_index_20260529",
        "original_raw": str(original_paths(exp, slug)["raw"]),
        "final_raw": str(out_raw),
        "original_rows": len(original_raw),
        "final_rows": len(final_rows),
        "replacement_count": len(replacements),
        "remaining_non_ok_count_before_rescore": len(remaining),
        "attempt_reports": attempt_reports,
        "replacement_examples": list(replacement_meta.values())[:20],
        "remaining_examples": remaining[:20],
        "original_raw_sha256": sha256(original_paths(exp, slug)["raw"]),
        "final_raw_sha256": sha256(out_raw),
        "merge_key": "repair_row_index",
        "id_merge_used": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_audit = exp / "status" / f"{slug}_full10000_parse_repaired3_audit.json"
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--attempt", type=int, required=True)
    sp.add_argument("--include-ambiguous", action="store_true")
    sp.set_defaults(func=prepare_cmd)

    sp = sub.add_parser("split")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--attempt", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.set_defaults(func=split_cmd)

    sp = sub.add_parser("merge-shards")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--attempt", type=int, required=True)
    sp.add_argument("--num-shards", type=int, required=True)
    sp.set_defaults(func=merge_shards_cmd)

    sp = sub.add_parser("finalize")
    sp.add_argument("--slug", required=True)
    sp.add_argument("--max-attempts", type=int, default=3)
    sp.set_defaults(func=finalize_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
