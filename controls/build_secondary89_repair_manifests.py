#!/usr/bin/env python3
"""Build isolated secondary-89 repair manifests from prior failed/pending IDs.

These manifests are explicitly secondary/provenance-only. They must not be
merged into canonical P1/P2 gates without a later row-level audit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
ROOT = PROJECT_ROOT / "results/a_group_supplement_20260524/analysis"
PROMPT_MANIFEST = ROOT / "prompt_sensitivity_20260525/manifests/prompt_sensitivity_subset2000_seed20260525.jsonl"
NA_MANIFEST = PROJECT_ROOT / "results/premise_false2000_na_position_all_v1.jsonl"
OUT_ROOT = ROOT / "secondary89_repair_manifests_20260525"


def row_id(row: dict) -> str:
    for key in ("probe_id", "id"):
        if row.get(key):
            return str(row[key])
    sample = row.get("sample")
    if isinstance(sample, dict):
        for key in ("probe_id", "id"):
            if sample.get(key):
                return str(sample[key])
    return ""


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_status_ids(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    data = json.loads(status_path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for key in ("failed_ids", "pending_ids"):
        for item_id in data.get(key) or []:
            if item_id:
                ids.add(str(item_id))
    return ids


def write_filtered(source_manifest: Path, ids: Iterable[str], out_path: Path) -> dict:
    selected = set(ids)
    rows = [row for row in read_jsonl(source_manifest) if row_id(row) in selected]
    found = {row_id(row) for row in rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "path": str(out_path),
        "source_manifest": str(source_manifest),
        "requested_ids": len(selected),
        "written_rows": len(rows),
        "missing_ids": sorted(selected - found),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    bad: list[str] = []

    for variant in ("strict_json", "answer_only", "rationale_then_json", "premise_guarded"):
        status_path = (
            ROOT
            / "prompt_sensitivity_20260525"
            / "simula_qwen25_kvasir"
            / variant
            / "status/status.json"
        )
        ids = read_status_ids(status_path)
        out_path = OUT_ROOT / "prompt_sensitivity_simula_qwen25_kvasir" / f"{variant}_failed_pending.jsonl"
        key = f"prompt_sensitivity/simula_qwen25_kvasir/{variant}"
        report[key] = write_filtered(PROMPT_MANIFEST, ids, out_path)
        if report[key]["requested_ids"] != report[key]["written_rows"] or report[key]["missing_ids"]:
            bad.append(key)

    na_status = ROOT / "na_position_control_20260525/simula_qwen25_kvasir/strict_json/status/status.json"
    na_ids = read_status_ids(na_status)
    na_out = OUT_ROOT / "na_position_simula_qwen25_kvasir" / "strict_json_failed_pending.jsonl"
    key = "na_position/simula_qwen25_kvasir/strict_json"
    report[key] = write_filtered(NA_MANIFEST, na_ids, na_out)
    if report[key]["requested_ids"] != report[key]["written_rows"] or report[key]["missing_ids"]:
        bad.append(key)

    report_path = OUT_ROOT / "manifest_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "items": report}, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit("repair_manifest_id_mismatch:" + ",".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
