#!/usr/bin/env python3
"""Run EndoPremiseBench MCQ inference through closed/API VLM endpoints.

The runner writes raw API outputs in the same row-level shape as the local VLM
runner, so `parse_and_score.py` can score the result without special cases.
Secrets are read only from environment variables.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_PROMPT_PATH = Path("prompts/mcq_json_v1.txt")


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
    return str(row.get("probe_id") or row.get("id") or "")


def read_resume_rows(path: Path) -> Tuple[set[str], List[Dict[str, Any]], int]:
    """Return successful resume rows and the number of dropped error/duplicate rows."""
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
    candidates: List[Path] = []
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
                data_root / "Kvasir-VQA-x1" / "images" / raw.name,
                data_root / "Kvasir-VQA" / "images" / raw.name,
                data_root / "EndoBench" / "EndoBench-Images" / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image_path={image_path}")


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(sample: Dict[str, Any], template: str) -> str:
    options = sample.get("options") or {}
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options)
    return f"{template}\n\nQuestion: {sample['question']}\nOptions:\n{option_text}"


def endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


def image_data_url(path: Path) -> Tuple[str, str, int, int]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with Image.open(path) as image:
        width, height = image.size
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}", mime, width, height


def anthropic_image_source(path: Path) -> Tuple[Dict[str, str], str, int, int]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with Image.open(path) as image:
        width, height = image.size
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "base64", "media_type": mime, "data": encoded}, mime, width, height


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if value:
                    pieces.append(str(value))
        return "\n".join(pieces).strip()
    return str(content)


def extract_openai_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return content_to_text(message.get("content"))


def extract_anthropic_text(data: Dict[str, Any]) -> str:
    parts = data.get("content") or []
    text_parts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            text_parts.append(part)
    return "\n".join(text_parts).strip()


class ApiClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.effort_disabled = False
        self.temperature_disabled = False
        self.last_request_at = 0.0

    def _wait_for_rate_limit(self) -> None:
        if self.args.min_request_interval <= 0:
            return
        with self.lock:
            now = time.time()
            wait = self.args.min_request_interval - (now - self.last_request_at)
            if wait > 0:
                time.sleep(wait)
            self.last_request_at = time.time()

    def call(self, prompt: str, image_path: Path) -> Tuple[str, Dict[str, Any]]:
        last_error = ""
        attempts = max(1, self.args.retries)
        for attempt in range(1, attempts + 1):
            send_effort = bool(self.args.reasoning_effort) and not self.effort_disabled
            try:
                if self.args.api_type == "anthropic_compatible":
                    text, meta = self._call_anthropic(prompt, image_path, send_effort)
                else:
                    text, meta = self._call_openai(prompt, image_path, send_effort)
                meta["attempts"] = attempt
                return text, meta
            except requests.HTTPError as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                body = response.text[:1000] if response is not None else str(exc)
                last_error = f"HTTPError status={status}: {body}"
                if status == 400 and send_effort and self._looks_like_effort_error(body):
                    with self.lock:
                        self.effort_disabled = True
                    continue
                if status == 400 and (not self.temperature_disabled) and self._looks_like_temperature_error(body):
                    with self.lock:
                        self.temperature_disabled = True
                    continue
                if status in {408, 409, 429, 500, 502, 503, 504} and attempt < attempts:
                    time.sleep(min(self.args.retry_max_sleep, self.args.retry_sleep * (2 ** (attempt - 1))))
                    continue
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts:
                    time.sleep(min(self.args.retry_max_sleep, self.args.retry_sleep * (2 ** (attempt - 1))))
                    continue
                break
        raise RuntimeError(last_error)

    @staticmethod
    def _looks_like_effort_error(body: str) -> bool:
        lowered = body.lower()
        return any(key in lowered for key in ["reasoning_effort", "thinking"])

    @staticmethod
    def _looks_like_temperature_error(body: str) -> bool:
        lowered = body.lower()
        return "temperature" in lowered and any(key in lowered for key in ["unsupported", "unknown", "not support", "invalid"])

    def _call_openai(self, prompt: str, image_path: Path, send_effort: bool) -> Tuple[str, Dict[str, Any]]:
        data_url, mime, width, height = image_data_url(image_path)
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": self.args.max_tokens,
        }
        if not self.temperature_disabled:
            payload["temperature"] = 0
        if send_effort:
            payload["reasoning_effort"] = self.args.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.args.api_key}",
            "Content-Type": "application/json",
        }
        started = time.time()
        self._wait_for_rate_limit()
        response = self.session.post(
            endpoint(self.args.base_url, "/chat/completions"),
            headers=headers,
            json=payload,
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return extract_openai_text(data), {
            "http_status": response.status_code,
            "api_endpoint": endpoint(self.args.base_url, "/chat/completions"),
            "image_mime_type": mime,
            "image_original_width": width,
            "image_original_height": height,
            "usage": usage,
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }

    def _call_anthropic(self, prompt: str, image_path: Path, send_effort: bool) -> Tuple[str, Dict[str, Any]]:
        source, mime, width, height = anthropic_image_source(image_path)
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": source},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if not self.temperature_disabled:
            payload["temperature"] = 0
        if send_effort and self.args.reasoning_effort in {"high", "xhigh", "max"}:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.args.thinking_budget_tokens}
            payload.pop("temperature", None)
        headers = {
            "x-api-key": self.args.api_key,
            "Authorization": f"Bearer {self.args.api_key}",
            "anthropic-version": self.args.anthropic_version,
            "Content-Type": "application/json",
        }
        started = time.time()
        self._wait_for_rate_limit()
        response = self.session.post(
            endpoint(self.args.base_url, "/messages"),
            headers=headers,
            json=payload,
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return extract_anthropic_text(data), {
            "http_status": response.status_code,
            "api_endpoint": endpoint(self.args.base_url, "/messages"),
            "image_mime_type": mime,
            "image_original_width": width,
            "image_original_height": height,
            "usage": usage,
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }


def run_one(
    client: ApiClient,
    sample: Dict[str, Any],
    prompt_template: str,
    project_root: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        **sample,
        "probe_id": sample.get("id"),
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model,
        "api_type": args.api_type,
        "base_url": args.base_url,
        "reasoning_effort_requested": args.reasoning_effort,
        "prompt_version": "mcq_json_v1_closed_api",
        "runner_version": "run_closed_api_premise_inference_v2_20260524",
        "image_policy": "original_base64",
        "max_image_pixels": 0,
        "qwen_max_pixels": 0,
        "image_resized": False,
    }
    try:
        image_path = resolve_image(sample["image_path"], project_root, data_root)
        prompt = build_prompt(sample, prompt_template)
        result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        result["request_hash"] = hashlib.sha256(
            json.dumps(
                {
                    "provider": args.provider,
                    "api_type": args.api_type,
                    "base_url": args.base_url,
                    "model": args.model,
                    "sample_id": row_key(sample),
                    "image_path": str(sample.get("image_path") or ""),
                    "prompt": prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        raw_output, meta = client.call(prompt, image_path)
        result["raw_output"] = raw_output
        result["response_raw"] = raw_output
        result["retry_count"] = max(0, int(meta.get("attempts") or 1) - 1)
        result["resolved_image_path"] = str(image_path)
        result["error"] = ""
        result.update(meta)
    except Exception as exc:
        result.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        result.setdefault("request_hash", "")
        result["raw_output"] = ""
        result["response_raw"] = ""
        result["retry_count"] = max(0, int(args.retries or 1) - 1)
        result["latency_seconds"] = round(time.time() - started, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--api-type", choices=["openai_compatible", "anthropic_compatible"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--retry-max-sleep", type=float, default=30.0)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--thinking-budget-tokens", type=int, default=1024)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--skip-success-output",
        type=Path,
        action="append",
        default=[],
        help="Existing raw jsonl to read as successful IDs to skip without rewriting.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    args.api_key = api_key

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    prompt_template = load_prompt_template(args.prompt_path)
    rows = read_jsonl(args.input, args.limit)
    for source_index, row in enumerate(rows):
        row.setdefault("source_manifest_index", source_index)
    input_rows = len(rows)
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    shard_rows = len(rows)
    done, resume_rows, dropped_resume_rows = read_resume_rows(args.output) if args.resume else (set(), [], 0)
    skip_done: set[str] = set()
    skipped_from_outputs: Dict[str, int] = {}
    for skip_path in args.skip_success_output:
        external_done, _, dropped = read_resume_rows(skip_path)
        skip_done.update(external_done)
        skipped_from_outputs[str(skip_path)] = len(external_done)
        if dropped:
            skipped_from_outputs[f"{skip_path}::dropped"] = dropped
    skip_done.difference_update(done)
    rows = [row for row in rows if row_key(row) not in done]
    rows = [row for row in rows if row_key(row) not in skip_done]
    client = ApiClient(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    append = bool(args.resume and resume_rows)
    if args.resume and args.output.exists():
        with args.output.open("w", encoding="utf-8") as f:
            for row in resume_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    started = time.time()
    completed = 0
    errors = 0
    with args.output.open("a" if append else "w", encoding="utf-8") as f:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {
                executor.submit(run_one, client, row, prompt_template, project_root, data_root, args): row_key(row)
                for row in rows
            }
            for future in futures.as_completed(future_map):
                item = future.result()
                if item.get("error"):
                    errors += 1
                completed += 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                if completed % max(1, args.progress_every) == 0:
                    print(
                        json.dumps(
                            {
                                "provider": args.provider,
                                "model": args.model,
                                "shard_index": args.shard_index,
                                "num_shards": args.num_shards,
                                "completed_this_run": completed,
                                "remaining_this_run": len(rows) - completed,
                                "errors_this_run": errors,
                                "output": str(args.output),
                                "elapsed_seconds": round(time.time() - started, 3),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "input": str(args.input),
                "output": str(args.output),
                "input_rows": input_rows,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "shard_rows_before_resume_skip": shard_rows,
                "records_requested_this_run": len(rows),
                "records_completed_this_run": completed,
                "skipped_existing": len(done),
                "skipped_success_outputs": skipped_from_outputs,
                "skipped_external_success": len(skip_done),
                "dropped_resume_rows": dropped_resume_rows,
                "errors_this_run": errors,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

