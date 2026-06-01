# EndoPremiseBench

**诊断内镜 VQA 中的虚假前提属性回答问题。**

[![Paper](https://img.shields.io/badge/Paper-ARR%20submission-B31B1B.svg)](#citation)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-2563EB.svg)](https://luious-lyh.github.io/EndoPremiseBench/)
[![Code](https://img.shields.io/badge/Code-reproducible%20pipeline-2F855A.svg)](#快速开始)
[![Data Policy](https://img.shields.io/badge/Data-bring%20licensed%20copies-6B7280.svg)](#数据说明)

[English](README.md) | [中文](README.zh-CN.md) | [项目主页](https://luious-lyh.github.io/EndoPremiseBench/) | [本地页面](docs/index.html)

<p align="center">
  <img src="assets/readme/benchmark_overview.png" width="95%" alt="EndoPremiseBench construction and evaluation overview">
</p>

EndoPremiseBench 是一个面向内镜 VQA 的诊断型基准，核心问题是：模型是否只在图像真正支持问题前提时回答属性问题，并在问题前提不成立时拒绝继续“顺着题目编答案”。每个样本都是四选一 MCQ，其中包含一个 `Not applicable` 选项。一个可靠模型既要能回答真实前提问题，也要能拒绝不被图像支持的属性前提。

## 最新动态

- **2026-05-26：** 最终论文软件包已整理完成，包含脱敏后的公开代码、更新后的基准图示、补充控制实验工具和面向审稿人的文档。

## 亮点

<p align="center">
  <img src="assets/readme/result_panel.jpg" width="95%" alt="Utility-restraint profile and unsupported-attribute exposure">
</p>

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

构建候选样本：

```bash
python data_building/profile_premise_data.py \
  --config configs/premise_probe_v2.json \
  --output-json results/data_reality_check_v2.json \
  --output-candidates results/premise_candidates_v2.jsonl \
  --output-md tables/data_reality_check_v2.md
```

创建平衡主划分和控制集：

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

运行本地开源 VLM：

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

运行闭源或 API-compatible VLM。密钥只从环境变量读取：

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

评分并汇总输出：

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

构建补充控制实验 manifest 和论文展示资产：

```bash
python controls/build_prompt_sensitivity.py
python controls/build_image_mismatch_manifest.py
python analysis/build_paper_assets.py
```

## 数据说明

请将已获授权的本地数据放在 `data/` 下，或在 `configs/premise_probe_v2.json` 中覆写路径：

```text
data/
  Kvasir-VQA/
  Kvasir-VQA-x1/
  EndoBench/
  EndoBench-Extended/
```

代码会相对于仓库根目录和 `--data-root` 解析路径。请遵守原始数据提供方的数据许可、患者隐私限制和再分发约束。

## 指标说明

| 指标 | 含义 |
| --- | --- |
| `Acc_TP` | 真实前提问题上的回答准确率。 |
| `Acc_FP` | 虚假前提问题上的正确拒绝率。 |
| `SR` | 虚假前提下的不支持属性暴露率。 |
| `ORR` | 真实前提下的过度拒答率。 |
| `PFR` | 保守解析后的解析失败率。 |
| `HPS` | 真实前提准确率和虚假前提拒绝率的调和均衡分数。 |

## 仓库结构

```text
configs/        探针 ontology 和源数据路径配置。
prompts/        MCQ prompt 模板。
data_building/  源数据画像和 benchmark manifest 构建。
inference/      开源模型、text-only 和闭源/API-compatible 推理脚本。
scoring/        答案解析、自测试和指标计算。
analysis/       结果聚合与论文展示资产生成。
controls/       补充控制实验、修复 manifest 和 API 控制运行。
examples/       可直接复制使用的命令示例。
docs/           GitHub Pages 项目主页、审计记录和 release notes。
assets/         README 与项目主页视觉材料。
results/        生成的 manifest、原始输出和评分输出。
tables/         生成的 CSV/Markdown/LaTeX 表格。
figures/        生成的论文图。
```

`results/`、`tables/` 和 `figures/` 默认只保留占位文件，其余生成物被忽略，以保证新 clone 的仓库保持轻量。

## 复现说明

- 正式运行 6,000 条样本前，建议先用 `--limit` 做 smoke test。
- 闭源/API 运行支持分片、断点续跑和跳过已成功输出：
  `--num-shards`、`--shard-index`、`--resume` 和 `--skip-success-output`。
- `docs/software_manifest.tsv` 记录论文软件附件中的文件内容和哈希。
- `docs/code_audit.md` 记录从旧本地目录和两台实验服务器纳入了哪些代码，以及为何排除 raw/log/endpoint 类产物。
- 论文资产脚本假设结果包已经生成，不会重新运行模型。

## Citation

```bibtex
@misc{endopremisebench2026,
  title  = {EndoPremiseBench: Diagnosing False-Premise Attribute Answering in Endoscopic VQA},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Code and benchmark pipeline}
}
```
