#!/usr/bin/env python3
"""Run EndoPremiseBench v2 MCQ inference with raw and parsed outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_PROMPT_PATH = Path("prompts/mcq_json_v1.txt")


def bootstrap_imports(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scoring"))


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def row_key(row: Dict[str, Any]) -> str:
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


def read_resume_rows(path: Path) -> Tuple[set[str], List[Dict[str, Any]], int]:
    """Keep unique successful rows and drop error/duplicate rows for resume."""
    done: set[str] = set()
    kept: List[Dict[str, Any]] = []
    dropped = 0
    if not path.exists():
        return done, kept, dropped
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                dropped += 1
                continue
            key = row_key(item)
            if key and not item.get("error") and key not in done:
                done.add(key)
                kept.append(item)
            else:
                dropped += 1
    return done, kept, dropped


def resolve_image(image_path: str, project_root: Path, data_root: Path) -> Path:
    normalized = str(image_path).replace("\\", "/")
    raw = Path(normalized)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([project_root / raw, data_root / raw])
    parts = list(raw.parts)
    if parts and parts[0].lower() == "data":
        stripped = Path(*parts[1:])
        candidates.append(data_root / stripped)
    if len(parts) >= 2 and parts[0] in {"Kvasir-VQA", "Kvasir-VQA-x1", "EndoBench", "EndoBench-Extended"}:
        candidates.append(data_root / raw)
    if raw.name:
        candidates.extend(
            [
                project_root / "shared_cache" / "reusable_predictions" / "kvasir_vqa_images" / raw.name,
                data_root / "Kvasir-VQA-x1" / "images" / raw.name,
                data_root / "Kvasir-VQA" / "images" / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image_path={image_path}; tried={candidates[:4]}")


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(sample: Dict[str, Any], template: str) -> str:
    options = sample.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict for sample {sample.get('id')}")
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options)
    return f"{template}\n\nQuestion: {sample['question']}\nOptions:\n{option_text}"


def parse_locally(sample: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    try:
        from parse_and_score import failure_type, parse_answer

        parsed, status, method = parse_answer(raw_output, sample.get("options") or {})
        return {
            "parsed_answer": parsed,
            "parse_status": status,
            "parse_method": method,
            "is_correct": bool(status == "ok" and parsed == sample.get("answer")),
            "failure_type": failure_type(sample, parsed, status),
        }
    except Exception as exc:
        return {
            "parsed_answer": None,
            "parse_status": "failure",
            "parse_method": "runner_parse_error",
            "is_correct": False,
            "failure_type": "Format Failure",
            "parser_error": f"{type(exc).__name__}: {exc}",
        }


def dry_run(rows: List[Dict[str, Any]], prompt_template: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            item = dict(row)
            item["prompt"] = build_prompt(row, prompt_template)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("shared_cache/model_downloads"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=0,
        help="Resize images above this total pixel count before inference. 0 keeps original resolution.",
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=0,
        help="Qwen vision max_pixels value. 0 omits the field and uses model defaults.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default="auto",
        help="Transformers/Accelerate device_map for generic, Qwen, Gemma, and PEFT routes.",
    )
    parser.add_argument(
        "--internvl-device-map",
        choices=["single", "split"],
        default="single",
        help="InternVL placement. 'single' preserves the old .cuda() route; 'split' uses a multi-GPU layer map.",
    )
    parser.add_argument(
        "--adapter",
        choices=[
            "auto",
            "generic",
            "internvl",
            "minicpm",
            "gemma",
            "qwen25",
            "simula_medgemma",
            "simula_qwen25",
            "llava_med",
        ],
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    bootstrap_imports(project_root)
    rows = read_jsonl(args.input, args.limit)
    done, resume_rows, dropped_resume_rows = read_resume_rows(args.output) if args.resume else (set(), [], 0)
    rows = [row for row in rows if row_key(row) not in done]
    prompt_template = load_prompt_template(args.prompt_path)

    if args.dry_run:
        dry_run(rows, prompt_template, args.output)
        print(json.dumps({"status": "dry_run", "records": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from run_transformers_probe_inference import (
        apply_image_pixel_limit,
        image_policy_metadata,
        load_any_model,
        predict,
    )

    import torch

    started = time.time()
    bundle = load_any_model(
        args.model_id,
        args.cache_dir,
        args.trust_remote_code,
        args.adapter,
        args.device_map,
        args.internvl_device_map,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        with args.output.open("w", encoding="utf-8") as f:
            for row in resume_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    append = bool(args.resume and resume_rows)
    with args.output.open("a" if append else "w", encoding="utf-8") as f:
        for idx, sample in enumerate(rows):
            result = {
                **sample,
                "probe_id": sample.get("id"),
                "model": args.model_id,
                "model_id": args.model_id,
                "adapter": bundle.get("adapter"),
                "model_class": bundle.get("model_class"),
                "requested_device_map": bundle.get("requested_device_map", args.device_map),
                "actual_device_map": bundle.get("actual_device_map", bundle.get("requested_device_map", args.device_map)),
                "prompt_version": "mcq_json_v1",
                "runner_version": "run_premise_inference_v2_20260522",
            }
            result.update(
                image_policy_metadata(
                    None,
                    max_image_pixels=args.max_image_pixels,
                    qwen_max_pixels=args.qwen_max_pixels,
                )
            )
            try:
                image_path = resolve_image(sample["image_path"], project_root, args.data_root)
                image = Image.open(image_path).convert("RGB")
                image.filename = str(image_path)
                image = apply_image_pixel_limit(image, args.max_image_pixels)
                image.filename = str(image_path)
                result.update(
                    image_policy_metadata(
                        image,
                        max_image_pixels=args.max_image_pixels,
                        qwen_max_pixels=args.qwen_max_pixels,
                    )
                )
                prompt = build_prompt(sample, prompt_template)
                t0 = time.time()
                with torch.inference_mode():
                    raw_output = predict(bundle, image, prompt, args.max_new_tokens, args.qwen_max_pixels)
                result["raw_output"] = str(raw_output)
                result["resolved_image_path"] = str(image_path)
                result["latency_seconds"] = round(time.time() - t0, 3)
                result["error"] = ""
                result.update(parse_locally(sample, str(raw_output)))
            except Exception as exc:
                result["raw_output"] = ""
                result["latency_seconds"] = None
                result["error"] = f"{type(exc).__name__}: {exc}"
                result.update(parse_locally(sample, ""))
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    summary = {
        "model_id": args.model_id,
        "requested_device_map": bundle.get("requested_device_map", args.device_map),
        "actual_device_map": bundle.get("actual_device_map", bundle.get("requested_device_map", args.device_map)),
        "input": str(args.input),
        "output": str(args.output),
        "records": len(rows),
        "skipped_existing": len(done),
        "dropped_resume_rows": dropped_resume_rows,
        "max_image_pixels": args.max_image_pixels,
        "qwen_max_pixels": args.qwen_max_pixels,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
