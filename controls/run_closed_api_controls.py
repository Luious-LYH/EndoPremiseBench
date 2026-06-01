#!/usr/bin/env python3
"""Run closed-model wording and question-only controls for EndoPremiseBench."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import requests


PROJECT_ROOT = Path(os.environ.get("EPB_PROJECT_ROOT", ".")).expanduser().resolve()
PROMPT_PATH = PROJECT_ROOT / "prompts/mcq_json_v1.txt"
SCORER_PATH = PROJECT_ROOT / "scoring/parse_and_score.py"
PRIMARY_MANIFEST = PROJECT_ROOT / "results/premise_balanced_main_v2.jsonl"
WORDING_MANIFEST = PROJECT_ROOT / "results/premise_false2000_wording_controls_v1.jsonl"
RESULT_BASE = PROJECT_ROOT / "results/a_group_supplement_20260524/analysis"


MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "grok_4_20_multi_agent_xhigh": {
        "provider": "openai_compatible",
        "api_type": "openai_compatible",
        "base_url_env": "ENDOPREMISE_OPENAI_COMPAT_BASE_URL",
        "api_key_env": "ENDOPREMISE_API_KEY",
        "model_env": "ENDOPREMISE_OPENAI_COMPAT_MODEL",
        "default_base_url": "",
        "default_model": "grok-4.20-multi-agent-xhigh",
        "reasoning_effort_env": "ENDOPREMISE_REASONING_EFFORT",
        "default_reasoning_effort": "xhigh",
    },
    "claude_opus_4_7": {
        "provider": "anthropic_compatible",
        "api_type": "anthropic_compatible",
        "base_url_env": "ENDOPREMISE_ANTHROPIC_BASE_URL",
        "api_key_env": "ENDOPREMISE_API_KEY",
        "model_env": "ENDOPREMISE_ANTHROPIC_MODEL",
        "default_base_url": "",
        "default_model": "claude-opus-4-7",
        "reasoning_effort_env": "ENDOPREMISE_REASONING_EFFORT",
        "default_reasoning_effort": "max",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("id") or "")


def endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


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
    return content_to_text(data.get("content"))


def json_obj_from_text(text: str) -> Dict[str, Any] | None:
    text = str(text or "").strip()
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_question_only_prompt(sample: Dict[str, Any], expected_model: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8").strip()
    options = sample.get("options") or {}
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options)
    return (
        "No image is provided for this control condition. "
        "Answer using only the question text and answer options.\n"
        f'Return JSON only: {{"answer": "<A/B/C/D>", "model_confirm": "{expected_model}"}}\n\n'
        f"{template}\n\nQuestion: {sample['question']}\nOptions:\n{option_text}"
    )


def build_wording_prompt(sample: Dict[str, Any], expected_model: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8").strip()
    options = sample.get("options") or {}
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options)
    return (
        f'Return JSON only: {{"answer": "<A/B/C/D>", "model_confirm": "{expected_model}"}}\n\n'
        f"{template}\n\nQuestion: {sample['question']}\nOptions:\n{option_text}"
    )


class TextOnlyClient:
    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args
        self.session = requests.Session()
        self.effort_disabled = False
        self.temperature_disabled = False

    def call(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        last_error = ""
        attempts = max(1, int(self.args.retries))
        for attempt in range(1, attempts + 1):
            send_effort = bool(self.args.reasoning_effort) and not self.effort_disabled
            try:
                if self.args.api_type == "anthropic_compatible":
                    text, meta = self._call_anthropic(prompt, send_effort)
                else:
                    text, meta = self._call_openai(prompt, send_effort)
                meta["attempts"] = attempt
                return text, meta
            except requests.HTTPError as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                body = response.text[:1000] if response is not None else str(exc)
                last_error = f"HTTPError status={status}: {body}"
                lowered = body.lower()
                if status == 400 and send_effort and any(k in lowered for k in ["reasoning_effort", "thinking"]):
                    self.effort_disabled = True
                    continue
                if status == 400 and (not self.temperature_disabled) and "temperature" in lowered:
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

    def _call_openai(self, prompt: str, send_effort: bool) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.args.max_tokens,
        }
        if not self.temperature_disabled:
            payload["temperature"] = 0
        if send_effort:
            payload["reasoning_effort"] = self.args.reasoning_effort
        headers = {"Authorization": f"Bearer {self.args.api_key}", "Content-Type": "application/json"}
        started = time.time()
        response = self.session.post(endpoint(self.args.base_url, "/chat/completions"), headers=headers, json=payload, timeout=self.args.timeout)
        response.raise_for_status()
        data = response.json()
        return extract_openai_text(data), {
            "http_status": response.status_code,
            "api_endpoint": endpoint(self.args.base_url, "/chat/completions"),
            "api_response_model": data.get("model", ""),
            "usage": data.get("usage") or {},
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }

    def _call_anthropic(self, prompt: str, send_effort: bool) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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
        response = self.session.post(endpoint(self.args.base_url, "/messages"), headers=headers, json=payload, timeout=self.args.timeout)
        response.raise_for_status()
        data = response.json()
        return extract_anthropic_text(data), {
            "http_status": response.status_code,
            "api_endpoint": endpoint(self.args.base_url, "/messages"),
            "api_response_model": data.get("model", ""),
            "usage": data.get("usage") or {},
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    spec = MODEL_SPECS[args.slug]
    api_key_env = args.api_key_env or spec["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")
    base_url = args.base_url or os.environ.get(spec["base_url_env"]) or spec["default_base_url"]
    if not base_url:
        raise SystemExit(
            "Missing API base URL. Pass --base-url or set "
            f"{spec['base_url_env']} for {args.slug}."
        )
    return SimpleNamespace(
        slug=args.slug,
        provider=spec["provider"],
        api_type=spec["api_type"],
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        model=args.model or os.environ.get(spec["model_env"]) or spec["default_model"],
        reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else os.environ.get(spec["reasoning_effort_env"], spec["default_reasoning_effort"]),
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        retry_max_sleep=args.retry_max_sleep,
        anthropic_version=args.anthropic_version,
        thinking_budget_tokens=args.thinking_budget_tokens,
    )


def result_root(experiment: str, slug: str) -> Path:
    if experiment == "wording":
        return RESULT_BASE / "wording_control_20260525" / slug
    if experiment == "question_only":
        return RESULT_BASE / "question_only_20260524/closed_api" / slug
    raise ValueError(experiment)


def manifest_for(experiment: str) -> Path:
    return WORDING_MANIFEST if experiment == "wording" else PRIMARY_MANIFEST


def output_prefix(experiment: str, slug: str) -> str:
    return f"{slug}_{'wording_control_20260525' if experiment == 'wording' else 'question_only_main6000'}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_one(client: TextOnlyClient, sample: Dict[str, Any], args: SimpleNamespace, experiment: str) -> Dict[str, Any]:
    rid = row_id(sample)
    prompt = build_wording_prompt(sample, args.model) if experiment == "wording" else build_question_only_prompt(sample, args.model)
    result: Dict[str, Any] = {
        **sample,
        "probe_id": sample.get("id"),
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model,
        "api_type": args.api_type,
        "base_url": args.base_url,
        "prompt_version": f"mcq_json_v1_{experiment}_closed_api_model_confirm_20260525",
        "runner_version": "run_closed_api_controls_20260525",
        "ablation_kind": experiment,
        "image_policy": "no_image_text_only",
        "image_resized": False,
        "timestamp_utc": utc_now(),
        "request_hash": hashlib.sha256(json.dumps({"model": args.model, "sample_id": rid, "prompt": prompt, "experiment": experiment}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    started = time.time()
    try:
        raw_output, meta = client.call(prompt)
        result["raw_output"] = raw_output
        result["response_raw"] = raw_output
        result["retry_count"] = max(0, int(meta.get("attempts") or 1) - 1)
        result["error"] = ""
        result.update(meta)
    except Exception as exc:
        result["raw_output"] = ""
        result["response_raw"] = ""
        result["retry_count"] = max(0, int(args.retries or 1) - 1)
        result["latency_seconds"] = round(time.time() - started, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_existing_success(path: Path) -> Tuple[set[str], List[Dict[str, Any]]]:
    done: set[str] = set()
    kept: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        rid = row_id(row)
        if rid and not row.get("error") and rid not in done:
            done.add(rid)
            kept.append(row)
    return done, kept


def run_cmd(args: argparse.Namespace) -> int:
    margs = model_args(args)
    root = result_root(args.experiment, args.slug)
    raw_shard = root / "raw_shards" / f"{output_prefix(args.experiment, args.slug)}_shard{args.shard_index:03d}_of{args.num_shards}_raw.jsonl"
    rows = read_jsonl(manifest_for(args.experiment), args.limit)
    rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    done, kept = read_existing_success(raw_shard) if args.resume else (set(), [])
    rows = [row for row in rows if row_id(row) not in done]
    if args.resume:
        write_jsonl(raw_shard, kept)
    client = TextOnlyClient(margs)
    completed = 0
    errors = 0
    started = time.time()
    raw_shard.parent.mkdir(parents=True, exist_ok=True)
    with raw_shard.open("a", encoding="utf-8") as handle:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {executor.submit(run_one, client, row, margs, args.experiment): row_id(row) for row in rows}
            for future in futures.as_completed(future_map):
                item = future.result()
                completed += 1
                if item.get("error"):
                    errors += 1
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                if completed % max(1, args.progress_every) == 0:
                    print(json.dumps({"slug": args.slug, "experiment": args.experiment, "completed_this_run": completed, "remaining_this_run": len(rows) - completed, "errors_this_run": errors, "elapsed_seconds": round(time.time() - started, 3)}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "done", "slug": args.slug, "experiment": args.experiment, "output": str(raw_shard), "records_completed_this_run": completed, "errors_this_run": errors, "elapsed_seconds": round(time.time() - started, 3)}, ensure_ascii=False), flush=True)
    return 0


def finalize_cmd(args: argparse.Namespace) -> int:
    margs = model_args(args)
    scorer = load_module("premise_scorer", SCORER_PATH)
    root = result_root(args.experiment, args.slug)
    manifest_path = manifest_for(args.experiment)
    manifest_rows = read_jsonl(manifest_path)
    manifest_ids = [row_id(row) for row in manifest_rows]
    clean: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, Dict[str, Any]] = {}
    for path in sorted((root / "raw_shards").glob("*.jsonl")):
        for row in read_jsonl(path):
            rid = row_id(row)
            if not rid:
                continue
            if row.get("error"):
                failed.setdefault(rid, row)
            elif rid not in clean:
                clean[rid] = row
    merged = [clean[rid] for rid in manifest_ids if rid in clean]
    completed_ids = [rid for rid in manifest_ids if rid in clean]
    failed_ids = [rid for rid in manifest_ids if rid in failed and rid not in clean]
    pending_ids = [rid for rid in manifest_ids if rid not in clean and rid not in failed]
    raw = root / "raw/raw.jsonl"
    scored = root / "scored/scored.jsonl"
    metrics = root / "metrics/metrics.json"
    status = root / "status/status.json"
    config = root / "config/config.json"
    manifest = root / "manifest/manifest.jsonl"
    log = root / "logs/run.log"
    write_jsonl(raw, merged)
    write_jsonl(manifest, manifest_rows)
    scored_rows, metric_payload = scorer.score_rows(merged)
    write_jsonl(scored, scored_rows)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({
        "experiment": "wording_control_20260525" if args.experiment == "wording" else "question_only_20260524",
        "slug": args.slug,
        "model": margs.model,
        "provider": margs.provider,
        "api_type": margs.api_type,
        "base_url": margs.base_url,
        "source_manifest": display_path(manifest_path),
        "prompt_path": display_path(PROMPT_PATH),
        "prompt_version": f"mcq_json_v1_{args.experiment}_closed_api_model_confirm_20260525",
        "runner_version": "run_closed_api_controls_20260525",
        "selection_rule": "fixed false-premise wording-control split" if args.experiment == "wording" else "primary balanced evaluation split, question text and options only",
        "created_utc": utc_now(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({
        "experiment": "wording_control_20260525" if args.experiment == "wording" else "question_only_20260524",
        "slug": args.slug,
        "model": margs.model,
        "status": "done" if len(completed_ids) == len(manifest_ids) and not failed_ids else "partial",
        "dataset_manifest": display_path(manifest_path),
        "prompt_variant": "neutral|explicit|guarded encoded in id suffix" if args.experiment == "wording" else "question_only",
        "n_total": len(manifest_ids),
        "n_completed": len(completed_ids),
        "n_pending": len(pending_ids),
        "n_failed": len(failed_ids),
        "completed_ids": completed_ids,
        "pending_ids": pending_ids,
        "failed_ids": failed_ids,
        "skipped_ids": [],
        "last_update": utc_now(),
        "resume_command": "python controls/run_closed_api_controls.py run ...",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, ids in [("completed_ids.txt", completed_ids), ("pending_ids.txt", pending_ids), ("failed_ids.txt", failed_ids)]:
        (root / "status" / name).write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] finalized {args.slug} {args.experiment} rows={len(completed_ids)} failed={len(failed_ids)} pending={len(pending_ids)}\n")
    print(json.dumps({"status": "COMPLETE" if len(completed_ids) == len(manifest_ids) and not failed_ids else "PARTIAL", "slug": args.slug, "experiment": args.experiment, "n_total": len(manifest_ids), "n_completed": len(completed_ids), "n_failed": len(failed_ids), "n_pending": len(pending_ids), **{k: metric_payload.get(k) for k in ["Acc_TP", "Acc_FP", "SR", "ORR", "HPS", "PFR"]}}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--experiment", choices=["wording", "question_only"], required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--thinking-budget-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--retry-max-sleep", type=float, default=180.0)
    parser.add_argument("--anthropic-version", default="2023-06-01")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--progress-every", type=int, default=20)
    run.add_argument("--resume", action="store_true")
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.cmd == "run":
        return run_cmd(args)
    if args.cmd == "finalize":
        return finalize_cmd(args)
    raise SystemExit(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
