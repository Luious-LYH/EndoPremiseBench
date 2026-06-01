#!/usr/bin/env python3
"""Merge closed/API shard raw files into one canonical raw file.

The merge is intentionally conservative:
- manifest order defines output order;
- successful rows are preferred over error rows;
- duplicate successful rows keep the first occurrence;
- error-only IDs are reported, but not written to the canonical raw by default.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def row_key(row: Dict[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("id") or "")


def iter_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ids(ids: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pending-ids", type=Path, default=None)
    parser.add_argument("--error-only-ids", type=Path, default=None)
    parser.add_argument("--input", action="append", default=[], help="Raw jsonl path or glob. Repeatable.")
    parser.add_argument("--include-error-only", action="store_true")
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    manifest_ids = [row_key(row) for row in manifest_rows]
    manifest_set = set(manifest_ids)
    success: Dict[str, Dict[str, Any]] = {}
    error_only: Dict[str, Dict[str, Any]] = {}
    malformed = 0
    duplicate_success = 0
    duplicate_error = 0
    outside_manifest = 0
    per_file: List[Dict[str, Any]] = []

    for path in iter_paths(args.input):
        file_rows = 0
        file_success = 0
        file_error = 0
        if not path.exists():
            per_file.append({"path": str(path), "exists": False})
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                file_rows += 1
                try:
                    row = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                key = row_key(row)
                if not key:
                    malformed += 1
                    continue
                if key not in manifest_set:
                    outside_manifest += 1
                    continue
                if row.get("error"):
                    file_error += 1
                    if key not in success and key not in error_only:
                        error_only[key] = row
                    else:
                        duplicate_error += 1
                    continue
                file_success += 1
                if key in success:
                    duplicate_success += 1
                    continue
                success[key] = row
                error_only.pop(key, None)
        per_file.append(
            {
                "path": str(path),
                "exists": True,
                "rows": file_rows,
                "success_rows": file_success,
                "error_rows": file_error,
            }
        )

    output_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    error_only_ids: List[str] = []
    for key in manifest_ids:
        if key in success:
            output_rows.append(success[key])
        elif args.include_error_only and key in error_only:
            output_rows.append(error_only[key])
            error_only_ids.append(key)
        else:
            missing.append(key)
            if key in error_only:
                error_only_ids.append(key)

    write_jsonl(output_rows, args.output)
    if args.pending_ids:
        write_ids(missing, args.pending_ids)
    if args.error_only_ids:
        error_only_set = set(error_only_ids)
        write_ids([key for key in manifest_ids if key in error_only_set], args.error_only_ids)

    report = {
        "manifest": str(args.manifest),
        "expected_n": len(manifest_ids),
        "output": str(args.output),
        "output_rows": len(output_rows),
        "success_ids": len(success),
        "missing_ids": len(missing),
        "error_only_ids": len(set(error_only_ids)),
        "error_only_ids_path": str(args.error_only_ids) if args.error_only_ids else "",
        "duplicate_success_rows": duplicate_success,
        "duplicate_error_rows": duplicate_error,
        "outside_manifest_rows": outside_manifest,
        "malformed_rows": malformed,
        "include_error_only": args.include_error_only,
        "inputs": per_file,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
