#!/usr/bin/env python3
"""Run EndoPremiseBench MCQ probes without image input for language-prior controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(sample: Dict[str, Any], template: str) -> str:
    options = sample.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict for sample {sample.get('id')}")
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options)
    return (
        "No image is provided for this control condition. "
        "Answer using only the question text and answer options.\n\n"
        f"{template}\n\nQuestion: {sample['question']}\nOptions:\n{option_text}"
    )


def first_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cuda:0")


def move_inputs(inputs: Any, device: Any) -> Any:
    try:
        return inputs.to(device)
    except Exception:
        if isinstance(inputs, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs


def generic_text_predict(bundle: Dict[str, Any], prompt: str, max_new_tokens: int) -> str:
    from run_transformers_probe_inference import decode

    processor = bundle["processor"]
    model = bundle["model"]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    texts: List[str] = []
    try:
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        pass
    texts.append(prompt)
    last_error: Optional[Exception] = None
    for text in texts:
        for kwargs in (
            {"text": [text], "return_tensors": "pt", "padding": True},
            {"text": text, "return_tensors": "pt"},
        ):
            try:
                inputs = processor(**kwargs)
            except Exception as exc:
                last_error = exc
                continue
            inputs = move_inputs(inputs, first_device(model))
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return decode(processor, generated, inputs)
    raise RuntimeError(f"Processor text-only failed: {type(last_error).__name__}: {last_error}")


def predict_text(bundle: Dict[str, Any], prompt: str, max_new_tokens: int) -> str:
    adapter = bundle["adapter"]
    if adapter in {"gemma", "simula_medgemma"}:
        import torch

        processor = bundle["processor"]
        model = bundle["model"]
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(first_device(model), dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
    if adapter == "internvl":
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        return bundle["model"].chat(bundle["tokenizer"], None, prompt, generation_config)
    if adapter == "minicpm":
        msgs = [{"role": "user", "content": prompt}]
        return bundle["model"].chat(
            msgs=msgs,
            image=None,
            tokenizer=bundle["tokenizer"],
            processor=bundle["processor"],
            max_new_tokens=max_new_tokens,
            sampling=False,
        )
    return generic_text_predict(bundle, prompt, max_new_tokens)


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
            item["ablation_kind"] = "question_only"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("shared_cache/model_downloads"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--adapter",
        choices=["auto", "generic", "internvl", "minicpm", "gemma", "simula_medgemma"],
        default="auto",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    bootstrap_imports(project_root)
    rows = read_jsonl(args.input, args.limit)
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    done, resume_rows, dropped_resume_rows = read_resume_rows(args.output) if args.resume else (set(), [], 0)
    rows = [row for row in rows if row_key(row) not in done]
    prompt_template = load_prompt_template(args.prompt_path)

    if args.dry_run:
        dry_run(rows, prompt_template, args.output)
        print(json.dumps({"status": "dry_run", "records": len(rows), "output": str(args.output)}, ensure_ascii=False))
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from run_transformers_probe_inference import load_any_model

    import torch

    started = time.time()
    bundle = load_any_model(args.model_id, args.cache_dir, args.trust_remote_code, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        with args.output.open("w", encoding="utf-8") as f:
            for row in resume_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    append = bool(args.resume and resume_rows)
    with args.output.open("a" if append else "w", encoding="utf-8") as f:
        for sample in rows:
            result = {
                **sample,
                "probe_id": sample.get("id"),
                "model": args.model_id,
                "model_id": args.model_id,
                "adapter": bundle.get("adapter"),
                "model_class": bundle.get("model_class"),
                "prompt_version": "mcq_json_v1_question_only",
                "runner_version": "run_premise_text_only_v1_20260522",
                "ablation_kind": "question_only",
            }
            try:
                prompt = build_prompt(sample, prompt_template)
                t0 = time.time()
                with torch.inference_mode():
                    raw_output = predict_text(bundle, prompt, args.max_new_tokens)
                result["raw_output"] = str(raw_output)
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
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "input": str(args.input),
                "output": str(args.output),
                "records": len(rows),
                "skipped_existing": len(done),
                "dropped_resume_rows": dropped_resume_rows,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
