# EndoPremiseBench

**Diagnosing false-premise attribute answering in endoscopic VQA.**

[![Paper](https://img.shields.io/badge/Paper-ARR%20submission-B31B1B.svg)](#citation)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-2563EB.svg)](https://luious-lyh.github.io/EndoPremiseBench/)
[![Code](https://img.shields.io/badge/Code-reproducible%20pipeline-2F855A.svg)](#quick-start)
[![Data Policy](https://img.shields.io/badge/Data-bring%20licensed%20copies-6B7280.svg)](#data)

[Project page](https://luious-lyh.github.io/EndoPremiseBench/) | [Local page](docs/index.html) | [English](#english) | [中文](#中文)

<p align="center">
  <img src="assets/readme/benchmark_overview.jpg" width="95%" alt="EndoPremiseBench construction and evaluation overview">
</p>

## English

EndoPremiseBench asks whether a vision-language model can answer an endoscopic
question only when the visual premise is supported by the image. Each probe is a
4-way multiple-choice question with one `Not applicable` option. A reliable
model should preserve utility on true-premise questions while refusing
unsupported attributes on false-premise questions.

## News

- **2026-05-26:** Final paper software package staged with sanitized code,
  updated benchmark visuals, supplemental-control utilities, and reviewer-facing
  documentation.

## Highlights

<p align="center">
  <img src="assets/readme/result_panel.jpg" width="95%" alt="Utility-restraint profile and unsupported-attribute exposure">
</p>

- Endoscopic VQA systems need both **utility** and **restraint**: high supported
  accuracy alone can hide unsupported-attribute exposure.
- Refusal-only behavior is not enough: always choosing `Not applicable` solves
  false-premise rejection while destroying useful supported answering.
- The released pipeline is source-balanced and auditable: construction,
  inference, parsing, scoring, control generation, and paper tables are separate
  scripts.

## What This Repository Contains

- Benchmark construction scripts for source profiling, premise conversion,
  balanced split creation, and control manifests.
- Local open-weight VLM runners, text-only controls, and API-compatible closed
  model inference.
- Conservative answer parsing and metrics for supported-question accuracy,
  false-premise rejection, unsupported-attribute exposure, over-refusal, parse
  failures, and balance.
- Paper-facing analysis utilities, including the frozen 20260525 asset builder.
- Supplemental control utilities for prompt sensitivity, image mismatch,
  closed/API text-only controls, parse repair, shard merging, and ID tracking.

This repository does **not** redistribute endoscopy images, model weights, API
keys, raw provider logs, or private review-stage material.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Build candidate probes:

```bash
python data_building/profile_premise_data.py \
  --config configs/premise_probe_v2.json \
  --output-json results/data_reality_check_v2.json \
  --output-candidates results/premise_candidates_v2.jsonl \
  --output-md tables/data_reality_check_v2.md
```

Create the balanced primary split and controls:

```bash
python data_building/build_balanced_subset.py \
  --input results/premise_candidates_v2.jsonl \
  --output results/premise_balanced_main_v2.jsonl

python data_building/build_control_manifests.py \
  --input results/premise_balanced_main_v2.jsonl \
  --out-na results/premise_false2000_na_position_all_v1.jsonl \
  --out-wording results/premise_false2000_wording_controls_v1.jsonl \
  --out-report tables/control_manifest_report_v1.md
```

Run a local open-weight VLM:

```bash
python inference/run_premise_inference.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter qwen25 \
  --input results/premise_balanced_main_v2.jsonl \
  --output results/qwen25_vl_7b_raw.jsonl \
  --data-root data \
  --project-root . \
  --trust-remote-code
```

Run a closed/API-compatible VLM. Secrets are read only from environment
variables:

```bash
export ENDOPREMISE_API_KEY=...
python inference/run_closed_api_premise_inference.py \
  --provider openai-compatible \
  --api-type openai_compatible \
  --base-url https://api.example.com/v1 \
  --api-key-env ENDOPREMISE_API_KEY \
  --model MODEL_NAME \
  --input results/premise_balanced_main_v2.jsonl \
  --output results/api_model_raw.jsonl \
  --data-root data
```

Score and summarize outputs:

```bash
python scoring/parse_and_score.py \
  --input results/qwen25_vl_7b_raw.jsonl \
  --output results/qwen25_vl_7b_scored.jsonl \
  --metrics-output results/qwen25_vl_7b_metrics.json

python analysis/summarize_main_metrics.py \
  --results-dir results \
  --table-output tables/main_results_v2.md \
  --csv-output tables/main_results_v2.csv
```

Build supplemental control manifests and paper-facing assets:

```bash
python controls/build_prompt_sensitivity.py
python controls/build_image_mismatch_manifest.py
python analysis/build_paper_assets.py
```

## Data

Place licensed local copies under `data/` or override paths in
`configs/premise_probe_v2.json`:

```text
data/
  Kvasir-VQA/
  Kvasir-VQA-x1/
  EndoBench/
  EndoBench-Extended/
```

The code resolves paths relative to the repository root and `--data-root`. Keep
dataset licenses, patient privacy restrictions, and redistribution limits with
the original providers.

## Metrics

| Metric | Meaning |
| --- | --- |
| `Acc_TP` | Accuracy on supported true-premise probes. |
| `Acc_FP` | Correct rejection rate on unsupported false-premise probes. |
| `SR` | Unsupported-attribute exposure rate on false-premise probes. |
| `ORR` | Over-refusal rate on supported probes. |
| `PFR` | Parse failure rate after conservative answer parsing. |
| `HPS` | Harmonic balance of supported accuracy and false-premise rejection. |

## Repository Layout

```text
configs/        Probe ontology and source-path configuration.
prompts/        MCQ prompt templates.
data_building/  Source profiling and benchmark manifest construction.
inference/      Open-weight, text-only, and closed/API-compatible runners.
scoring/        Answer parsing, self-tests, and metric computation.
analysis/       Result aggregation and paper-facing asset generation.
controls/       Supplemental controls, repair manifests, and API control runs.
examples/       Copy-pasteable command recipes.
docs/           GitHub Pages project site, audits, and release notes.
assets/         README and project-page visuals.
results/        Generated manifests, raw outputs, scored outputs.
tables/         Generated CSV/Markdown/LaTeX tables.
figures/        Generated paper figures.
```

`results/`, `tables/`, and `figures/` are ignored except for placeholders so
fresh clones stay light.

## Reproducibility Notes

- Use `--limit` for smoke tests before launching full 6,000-item runs.
- Closed/API runs support sharding and resume/skip logic through
  `--num-shards`, `--shard-index`, `--resume`, and `--skip-success-output`.
- `docs/software_manifest.tsv` records the staged software attachment contents
  and hashes used for the paper package.
- `docs/code_audit.md` records what was included from the old local tree and
  the two experiment servers, and why raw/log/endpoint artifacts were excluded.
- Paper asset scripts assume that generated result bundles already exist; they
  do not rerun models.

## 中文

EndoPremiseBench 是一个面向内镜 VQA 的诊断型基准，核心问题是：模型是否只在图像真正支持问题前提时回答属性问题，并在问题前提不成立时拒绝继续“顺着题目编答案”。每个样本都是四选一 MCQ，其中包含一个 `Not applicable` 选项。一个可靠模型既要能回答真实前提问题，也要能拒绝不被图像支持的属性前提。

## 最新动态

- **2026-05-26：** 最终论文软件包已整理完成，包含脱敏后的公开代码、更新后的基准图示、补充控制实验工具和面向审稿人的文档。

## 亮点

- 内镜 VQA 需要同时评估**有用性**和**克制性**：只看真实前提准确率，会掩盖模型在虚假前提下暴露不支持属性的问题。
- 单纯拒答不是解决方案：总是选择 `Not applicable` 虽然能提高虚假前提拒绝率，但会破坏真实前提问题上的可用性。
- 公开代码将数据构建、推理、解析、评分、控制实验和论文表格生成拆开，便于复核和复现。

## 仓库内容

- 基准构建脚本：源数据画像、前提转换、平衡划分和控制 manifest 构建。
- 推理脚本：本地开源 VLM、text-only 控制、API-compatible 闭源模型推理。
- 保守答案解析与指标计算：真实前提准确率、虚假前提拒绝率、不支持属性暴露率、过度拒答率、解析失败率和平衡指标。
- 论文分析工具：包括 20260525 冻结版论文资产构建器。
- 补充实验工具：prompt sensitivity、image mismatch、闭源 API text-only control、parse repair、shard merge 和 ID tracking。

本仓库不重新分发内镜图像、模型权重、API key、原始 provider 日志或私有审稿阶段材料。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

构建候选样本、主划分和控制集：

```bash
python data_building/profile_premise_data.py \
  --config configs/premise_probe_v2.json \
  --output-json results/data_reality_check_v2.json \
  --output-candidates results/premise_candidates_v2.jsonl \
  --output-md tables/data_reality_check_v2.md

python data_building/build_balanced_subset.py \
  --input results/premise_candidates_v2.jsonl \
  --output results/premise_balanced_main_v2.jsonl

python data_building/build_control_manifests.py \
  --input results/premise_balanced_main_v2.jsonl \
  --out-na results/premise_false2000_na_position_all_v1.jsonl \
  --out-wording results/premise_false2000_wording_controls_v1.jsonl \
  --out-report tables/control_manifest_report_v1.md
```

运行推理、评分和论文资产构建：

```bash
python inference/run_premise_inference.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter qwen25 \
  --input results/premise_balanced_main_v2.jsonl \
  --output results/qwen25_vl_7b_raw.jsonl \
  --data-root data \
  --project-root . \
  --trust-remote-code

python scoring/parse_and_score.py \
  --input results/qwen25_vl_7b_raw.jsonl \
  --output results/qwen25_vl_7b_scored.jsonl \
  --metrics-output results/qwen25_vl_7b_metrics.json

python analysis/build_paper_assets.py
```

## 数据说明

请将已获授权的本地数据放在 `data/` 下，或在 `configs/premise_probe_v2.json` 中覆写路径。仓库只提供代码和构建逻辑，不包含原始内镜图像。

## 指标说明

| 指标 | 含义 |
| --- | --- |
| `Acc_TP` | 真实前提问题上的回答准确率。 |
| `Acc_FP` | 虚假前提问题上的正确拒绝率。 |
| `SR` | 虚假前提下的不支持属性暴露率。 |
| `ORR` | 真实前提下的过度拒答率。 |
| `PFR` | 保守解析后的解析失败率。 |
| `HPS` | 真实前提准确率和虚假前提拒绝率的调和均衡分数。 |

## Citation

```bibtex
@misc{endopremisebench2026,
  title  = {EndoPremiseBench: Diagnosing False-Premise Attribute Answering in Endoscopic VQA},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Code and benchmark pipeline}
}
```
