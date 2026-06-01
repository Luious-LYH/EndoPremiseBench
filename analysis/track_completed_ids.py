#!/usr/bin/env python3
"""Track completed, failed, duplicate, and pending EndoPremiseBench IDs.

This script treats the main manifest as the source of truth and recomputes
per-model ID state from current raw JSONL files. It is safe to run while jobs
are active; a partial trailing line is ignored and reported as malformed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
DEFAULT_MANIFEST = PROJECT_ROOT / "results/premise_balanced_main_v2.jsonl"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "results/a_group_supplement_20260524"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/a_group_supplement_20260524/id_tracking"

OPEN_MODELS = [
    "qwen3_vl_8b",
    "medgemma_4b",
    "qwen25_vl_7b",
    "internvl25_8b",
    "simula_medgemma_kvasir",
    "llava_med_v15_mistral_7b",
    "lingshu_7b",
    "simula_qwen25_kvasir",
]

CLOSED_MODELS = [
    "gpt_5_5",
    "claude_opus_4_7",
    "gemini_3_1_pro_high",
    "grok_4_20_multi_agent_xhigh",
]

QUESTION_ONLY_CONTROLS = {
    "qwen3_vl_8b_question_only": Path(
        "analysis/question_only_20260524/qwen3_vl_8b/raw/qwen3_vl_8b_question_only_main6000_raw.jsonl"
    ),
    "medgemma_4b_question_only": Path(
        "analysis/question_only_20260524/medgemma_4b/raw/medgemma_4b_question_only_main6000_raw.jsonl"
    ),
}


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return rows, malformed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def row_id(row: Dict[str, Any]) -> str:
    for key in ("probe_id", "id"):
        value = row.get(key)
        if value:
            return str(value)
    sample = row.get("sample")
    if isinstance(sample, dict):
        for key in ("probe_id", "id"):
            value = sample.get(key)
            if value:
                return str(value)
    return ""


def is_local_gpt55_substitute(row: Dict[str, Any]) -> bool:
    not_api_response = row.get("not_api_response")
    not_api_response_true = (
        not_api_response is True
        or (isinstance(not_api_response, str) and not_api_response.strip().lower() in {"1", "true", "yes"})
    )
    endpoint = str(row.get("endpoint_id") or "").lower()
    runner = str(row.get("runner_version") or "").lower()
    base_url = str(row.get("base_url") or "").lower()
    api_endpoint = str(row.get("api_endpoint") or "").lower()
    model = str(row.get("model") or row.get("model_id") or "").lower()
    return (
        not_api_response_true
        or "local_codex" in endpoint
        or "local_53580" in endpoint
        or "local_codex" in runner
        or "local substitute" in runner
        or base_url.startswith("http://127.0.0.1")
        or base_url.startswith("http://localhost")
        or api_endpoint.startswith("http://127.0.0.1")
        or api_endpoint.startswith("http://localhost")
        or model == "gpt-5.5-local-codex-substitute"
    )


def write_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in ids), encoding="utf-8")


def write_manifest_subset(path: Path, ids: Iterable[str], manifest_by_id: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item_id in ids:
            row = manifest_by_id.get(item_id)
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_path_for(group: str, slug: str, result_root: Path, phase: str, smoke_limit: int) -> Path:
    if phase == "smoke":
        if group == "closed_api":
            return result_root / "closed_api_comparison" / "smoke" / f"{slug}_premise_main6000_smoke{smoke_limit}_raw.jsonl"
        return result_root / "smoke" / f"{slug}_main6000_smoke{smoke_limit}_raw.jsonl"
    if group == "closed_api":
        return result_root / "closed_api_comparison" / "raw" / f"{slug}_premise_main6000_raw.jsonl"
    return result_root / "raw" / f"{slug}_premise_main6000_raw.jsonl"


def raw_paths_for(group: str, slug: str, result_root: Path, phase: str, smoke_limit: int) -> List[Path]:
    paths = [raw_path_for(group, slug, result_root, phase, smoke_limit)]
    if phase == "full" and group == "closed_api":
        shard_root = result_root / "closed_api_comparison" / "raw_shards" / slug
        paths.extend(sorted(shard_root.glob(f"{slug}_premise_main6000_shard*_raw.jsonl")))
    return paths


def model_specs(result_root: Path, phase: str, smoke_limit: int) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    for slug in OPEN_MODELS:
        raw_paths = raw_paths_for("open_local", slug, result_root, phase, smoke_limit)
        specs.append(
            {
                "phase": phase,
                "group": "open_local",
                "slug": slug,
                "raw": str(raw_paths[0]),
                "raw_paths": [str(path) for path in raw_paths],
            }
        )
    for slug in CLOSED_MODELS:
        raw_paths = raw_paths_for("closed_api", slug, result_root, phase, smoke_limit)
        specs.append(
            {
                "phase": phase,
                "group": "closed_api",
                "slug": slug,
                "raw": str(raw_paths[0]),
                "raw_paths": [str(path) for path in raw_paths],
            }
        )
    return specs


def question_only_specs(result_root: Path) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for slug, rel_path in QUESTION_ONLY_CONTROLS.items():
        raw = result_root / rel_path
        specs.append(
            {
                "phase": "question_only",
                "group": "control",
                "slug": slug,
                "raw": str(raw),
                "raw_paths": [str(raw)],
            }
        )
    return specs


def analyze_model(
    spec: Dict[str, str],
    manifest_ids: List[str],
    manifest_by_id: Dict[str, Dict[str, Any]],
    output_root: Path,
) -> Dict[str, Any]:
    phase = spec.get("phase", "full")
    raw_paths = [Path(path) for path in spec.get("raw_paths", [spec["raw"]])]
    counts: Counter[str] = Counter()
    success_ids: set[str] = set()
    error_ids: set[str] = set()
    unknown_ids: set[str] = set()
    local_substitute_ids: set[str] = set()
    error_examples: Dict[str, str] = {}
    malformed = 0
    rows_total = 0
    raw_path_states: List[Dict[str, Any]] = []

    for raw_path in raw_paths:
        rows, file_malformed = read_jsonl(raw_path)
        malformed += file_malformed
        rows_total += len(rows)
        raw_path_states.append(
            {
                "path": str(raw_path),
                "exists": raw_path.exists(),
                "raw_rows": len(rows),
                "malformed_lines": file_malformed,
            }
        )
        for row in rows:
            item_id = row_id(row)
            if not item_id:
                unknown_ids.add("")
                continue
            counts[item_id] += 1
            if item_id not in manifest_by_id:
                unknown_ids.add(item_id)
            if phase == "full" and spec["group"] == "closed_api" and spec["slug"] == "gpt_5_5" and is_local_gpt55_substitute(row):
                if item_id in manifest_by_id:
                    local_substitute_ids.add(item_id)
                continue
            if row.get("error"):
                error_ids.add(item_id)
                error_examples.setdefault(item_id, str(row.get("error"))[:600])
            else:
                success_ids.add(item_id)

    manifest_id_set = set(manifest_ids)
    attempted_ids = {item_id for item_id in counts if item_id in manifest_id_set}
    completed_ids = {item_id for item_id in success_ids if item_id in manifest_id_set}
    failed_only_ids = sorted((error_ids & manifest_id_set) - completed_ids, key=manifest_ids.index)
    pending_ids = [item_id for item_id in manifest_ids if item_id not in completed_ids]
    never_attempted_ids = [item_id for item_id in manifest_ids if item_id not in attempted_ids]
    duplicate_ids = sorted([item_id for item_id, count in counts.items() if item_id in manifest_id_set and count > 1], key=manifest_ids.index)
    completed_ordered = [item_id for item_id in manifest_ids if item_id in completed_ids]
    attempted_ordered = [item_id for item_id in manifest_ids if item_id in attempted_ids]

    if phase == "full":
        model_dir = output_root / spec["group"] / spec["slug"]
    else:
        model_dir = output_root / phase / spec["group"] / spec["slug"]
    write_ids(model_dir / "completed_success_ids.txt", completed_ordered)
    write_ids(model_dir / "attempted_ids.txt", attempted_ordered)
    write_ids(model_dir / "failed_only_ids.txt", failed_only_ids)
    write_ids(model_dir / "pending_ids.txt", pending_ids)
    write_ids(model_dir / "never_attempted_ids.txt", never_attempted_ids)
    write_ids(model_dir / "duplicate_ids.txt", duplicate_ids)
    write_ids(model_dir / "local_substitute_diagnostic_ids.txt", sorted(local_substitute_ids, key=manifest_ids.index))
    write_ids(model_dir / "unknown_ids.txt", sorted(item for item in unknown_ids if item))
    write_manifest_subset(model_dir / "pending_manifest.jsonl", pending_ids, manifest_by_id)
    write_manifest_subset(model_dir / "failed_only_manifest.jsonl", failed_only_ids, manifest_by_id)
    write_manifest_subset(model_dir / "never_attempted_manifest.jsonl", never_attempted_ids, manifest_by_id)

    state = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "group": spec["group"],
        "slug": spec["slug"],
        "raw_path": spec["raw"],
        "raw_paths": raw_path_states,
        "raw_exists": any(item["exists"] for item in raw_path_states),
        "manifest_total": len(manifest_ids),
        "raw_rows": rows_total,
        "unique_attempted": len(attempted_ids),
        "completed_success": len(completed_ids),
        "failed_only": len(failed_only_ids),
        "pending": len(pending_ids),
        "never_attempted": len(never_attempted_ids),
        "duplicate_ids": len(duplicate_ids),
        "excluded_local_substitute_diagnostic": len(local_substitute_ids),
        "formal_api_only": phase == "full" and spec["group"] == "closed_api" and spec["slug"] == "gpt_5_5",
        "unknown_ids": len([item for item in unknown_ids if item]),
        "malformed_lines": malformed,
        "paths": {
            "completed_success_ids": str(model_dir / "completed_success_ids.txt"),
            "attempted_ids": str(model_dir / "attempted_ids.txt"),
            "failed_only_ids": str(model_dir / "failed_only_ids.txt"),
            "pending_ids": str(model_dir / "pending_ids.txt"),
            "never_attempted_ids": str(model_dir / "never_attempted_ids.txt"),
            "duplicate_ids": str(model_dir / "duplicate_ids.txt"),
            "local_substitute_diagnostic_ids": str(model_dir / "local_substitute_diagnostic_ids.txt"),
            "pending_manifest": str(model_dir / "pending_manifest.jsonl"),
            "failed_only_manifest": str(model_dir / "failed_only_manifest.jsonl"),
            "never_attempted_manifest": str(model_dir / "never_attempted_manifest.jsonl"),
        },
        "error_examples": {item_id: error_examples[item_id] for item_id in failed_only_ids[:10] if item_id in error_examples},
    }
    (model_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def write_summary(states: List[Dict[str, Any]], output_root: Path) -> None:
    fields = [
        "phase",
        "group",
        "slug",
        "raw_exists",
        "manifest_total",
        "raw_rows",
        "unique_attempted",
        "completed_success",
        "failed_only",
        "pending",
        "never_attempted",
        "duplicate_ids",
        "unknown_ids",
        "malformed_lines",
        "raw_path",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for state in states:
            writer.writerow({field: state.get(field) for field in fields})

    lines = [
        "# EndoPremiseBench ID Tracking",
        "",
        f"- Updated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Model/phase entries tracked: {len(states)}",
        "",
        "| Phase | Group | Model | Raw rows | Success | Failed-only | Pending | Never attempted | Duplicates | Malformed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in states:
        lines.append(
            "| {phase} | {group} | {slug} | {raw_rows} | {completed_success} | {failed_only} | {pending} | "
            "{never_attempted} | {duplicate_ids} | {malformed_lines} |".format(**state)
        )
    lines.extend(
        [
            "",
            "Full-run per-model directories live under `id_tracking/{open_local,closed_api}/MODEL/`.",
            "Smoke-run per-model directories live under `id_tracking/smoke/{open_local,closed_api}/MODEL/`.",
            "Question-only control directories live under `id_tracking/question_only/control/MODEL/`.",
            "Each per-model directory contains `completed_success_ids.txt`, `attempted_ids.txt`, "
            "`failed_only_ids.txt`, `pending_ids.txt`, `never_attempted_ids.txt`, `duplicate_ids.txt`, "
            "`pending_manifest.jsonl`, and `state.json`.",
        ]
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "state.json").write_text(json.dumps({"models": states}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-limit", type=int, default=24)
    args = parser.parse_args()

    manifest_rows, malformed = read_jsonl(args.manifest)
    if malformed:
        raise SystemExit(f"Manifest has malformed lines: {malformed}")

    manifest_ids: List[str] = []
    manifest_by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: defaultdict[str, int] = defaultdict(int)
    for row in manifest_rows:
        item_id = row_id(row)
        if not item_id:
            continue
        duplicates[item_id] += 1
        if item_id not in manifest_by_id:
            manifest_ids.append(item_id)
            manifest_by_id[item_id] = row
    duplicate_manifest_ids = sorted(item_id for item_id, count in duplicates.items() if count > 1)
    if duplicate_manifest_ids:
        raise SystemExit(f"Manifest ID duplicates found: {duplicate_manifest_ids[:10]}")

    states = [
        analyze_model(spec, manifest_ids, manifest_by_id, args.output_root)
        for spec in model_specs(args.result_root, "full", args.smoke_limit)
    ]
    states.extend(
        analyze_model(spec, manifest_ids, manifest_by_id, args.output_root)
        for spec in model_specs(args.result_root, "smoke", args.smoke_limit)
    )
    states.extend(
        analyze_model(spec, manifest_ids, manifest_by_id, args.output_root)
        for spec in question_only_specs(args.result_root)
    )
    write_summary(states, args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "models": len(states), "manifest_total": len(manifest_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
