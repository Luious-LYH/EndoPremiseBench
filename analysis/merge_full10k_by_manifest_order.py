#!/usr/bin/env python3
"""Merge full10000 API shards without using id/probe_id as unique keys.

The expanded full10000 manifest intentionally contains two duplicate ids whose
options/answers differ. This finalizer preserves manifest physical row order and
therefore must never collapse rows by id.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SIGNATURE_KEYS = (
    "id",
    "image_path",
    "question",
    "options",
    "answer",
    "premise_type",
    "attribute_type",
    "source_dataset",
    "source_id",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_paths(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for raw in matches or [pattern]:
            path = Path(raw)
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(values: Iterable[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(v) for v in values)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def signature(row: Dict[str, Any]) -> str:
    payload = {key: row.get(key) for key in SIGNATURE_KEYS}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pending-row-indices", type=Path, default=None)
    parser.add_argument("--error-row-indices", type=Path, default=None)
    parser.add_argument("--input", action="append", default=[], help="Raw shard jsonl path or glob. Repeatable.")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    manifest_sigs = [signature(row) for row in manifest_rows]
    merged: List[Dict[str, Any] | None] = [None] * len(manifest_rows)
    error_indices = set()
    malformed = 0
    outside_manifest = 0
    missing_source_index = 0
    duplicate_assigned = 0
    source_index_mismatch = 0
    per_file: List[Dict[str, Any]] = []

    for path in iter_paths(args.input):
        info = {"path": str(path), "exists": path.exists(), "rows": 0, "errors": 0}
        if not path.exists():
            per_file.append(info)
            continue
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            info["rows"] += 1
            try:
                row = json.loads(line)
            except Exception:
                malformed += 1
                continue
            raw_idx = row.get("source_manifest_index")
            if raw_idx is None:
                missing_source_index += 1
                continue
            try:
                idx = int(raw_idx)
            except Exception:
                source_index_mismatch += 1
                continue
            if idx < 0 or idx >= len(manifest_rows):
                outside_manifest += 1
                continue
            if manifest_sigs[idx] != signature(row):
                source_index_mismatch += 1
                continue
            if merged[idx] is not None:
                current = merged[idx] or {}
                # Prefer a later successful retry over an earlier error row.
                if current.get("error") and not row.get("error"):
                    out = dict(row)
                    out["source_manifest_index"] = idx
                    out["merge_protocol"] = "manifest_row_index_only_v2_20260601"
                    merged[idx] = out
                    error_indices.discard(idx)
                    if row.get("error"):
                        error_indices.add(idx)
                        info["errors"] += 1
                else:
                    duplicate_assigned += 1
                continue
            out = dict(row)
            out["source_manifest_index"] = idx
            out["merge_protocol"] = "manifest_row_index_only_v2_20260601"
            merged[idx] = out
            if row.get("error"):
                error_indices.add(idx)
                info["errors"] += 1
        per_file.append(info)

    pending = [idx for idx, row in enumerate(merged) if row is None]
    ids = [str(row.get("id") or row.get("probe_id") or "") for row in manifest_rows]
    duplicate_ids = {key: count for key, count in Counter(ids).items() if count > 1}

    status = "ok"
    if pending and not args.allow_partial:
        status = "incomplete"
    if error_indices and not args.allow_errors:
        status = "has_errors"
    complete_rows = [row for row in merged if row is not None]
    if status == "ok" or args.allow_partial:
        write_jsonl(complete_rows, args.output)

    if args.pending_row_indices:
        write_lines(pending, args.pending_row_indices)
    if args.error_row_indices:
        write_lines(sorted(error_indices), args.error_row_indices)

    report = {
        "status": status,
        "manifest": str(args.manifest),
        "expected_rows": len(manifest_rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "output": str(args.output),
        "output_written": args.output.exists(),
        "output_rows": len(complete_rows) if args.output.exists() else 0,
        "missing_rows": len(pending),
        "error_rows": len(error_indices),
        "malformed_rows": malformed,
        "outside_manifest_rows": outside_manifest,
        "missing_source_index_rows": missing_source_index,
        "duplicate_assigned_rows": duplicate_assigned,
        "source_index_mismatch_rows": source_index_mismatch,
        "inputs": per_file,
        "output_sha256": sha256(args.output),
        "merge_key": "source_manifest_index",
        "id_merge_used": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if status != "ok":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
