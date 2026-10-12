#!/usr/bin/env python3
"""Apply the model-output-blind dual-annotator adjudication decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXCLUDED = {"xctrl_5f4f4266fd181ad1"}
REASON = (
    "Conservative adjudication of the only annotator disagreement: positive "
    "polyp_count/polyp_type co-occurred with polyp_size=none. Size is outside "
    "the experiment's entity-status rule, but the block is excluded to avoid "
    "depending on that interpretation. Model outputs were not consulted."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    blocks = read_jsonl(args.blocks)
    probes = read_jsonl(args.manifest)
    kept_blocks = [row for row in blocks if row["block_id"] not in EXCLUDED]
    kept_probes = [row for row in probes if row["block_id"] not in EXCLUDED]
    if len(blocks) != 165 or len(probes) != 660:
        raise ValueError("unexpected pre-adjudication manifest size")
    if len(kept_blocks) != 164 or len(kept_probes) != 656:
        raise ValueError("unexpected post-adjudication manifest size")
    if Counter(row["attribute_type"] for row in kept_blocks) != {"count": 158, "location": 6}:
        raise ValueError("unexpected post-adjudication stratum counts")
    if len({row["source_image_id"] for row in kept_probes}) != 328:
        raise ValueError("unexpected post-adjudication image count")
    if len({row["query_id"] for row in kept_probes}) != 328:
        raise ValueError("unexpected post-adjudication query count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    block_out = args.output_dir / "crossed_blocks_164.jsonl"
    manifest_out = args.output_dir / "crossed_manifest_656.jsonl"
    write_jsonl(block_out, kept_blocks)
    write_jsonl(manifest_out, kept_probes)
    keep_ids = {row["id"] for row in kept_probes}
    raw_records = {}
    raw_out_dir = args.output_dir / "raw"
    for path in sorted(args.raw_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        filtered = [row for row in rows if str(row.get("id") or row.get("probe_id") or "") in keep_ids]
        if len(rows) != 660 or len(filtered) != 656:
            raise ValueError(f"{path}: unexpected raw size {len(rows)} -> {len(filtered)}")
        output = raw_out_dir / path.name
        write_jsonl(output, filtered)
        raw_records[path.stem] = {"file": output.name, "sha256": sha256(output), "rows": len(filtered)}

    validation = {
        "schema": "crossed_atomic_adjudication_v1",
        "model_outputs_consulted": False,
        "input_blocks": len(blocks),
        "input_probes": len(probes),
        "excluded_block_ids": sorted(EXCLUDED),
        "adjudication_reason": REASON,
        "blocks": len(kept_blocks),
        "probes": len(kept_probes),
        "unique_images": len({row["source_image_id"] for row in kept_probes}),
        "unique_queries": len({row["query_id"] for row in kept_probes}),
        "stratum_counts": dict(Counter(row["attribute_type"] for row in kept_blocks)),
        "block_manifest": block_out.name,
        "block_manifest_sha256": sha256(block_out),
        "probe_manifest": manifest_out.name,
        "probe_manifest_sha256": sha256(manifest_out),
        "raw_outputs": raw_records,
    }
    validation_path = args.output_dir / "adjudication_validation.json"
    with validation_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
