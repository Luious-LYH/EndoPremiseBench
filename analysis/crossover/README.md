# Atomic 2x2 crossover analysis

This pipeline constructs and analyzes the source-grounded atomic-evidence
crossover reported in the paper. It does not run VLM inference. The paper's
frozen analysis contains 164 adjudicated blocks: 158 instrument--polyp count
blocks in the primary analysis and 6 location blocks used descriptively.

## Inputs and release boundary

Obtain the source annotations and endoscopy images from their original
providers. The builder expects a test-split Parquet file and local images at
`data/Kvasir-VQA-x1/images/<image_id>.jpg`. Neither the source files nor the
generated probe manifest, image hashes, raw model outputs, or row-level scored
outputs are included in this repository. Generated files may contain
source-derived annotation text or identifiers; keep them in a private,
ignored output directory.

Install the optional builder dependencies:

```bash
python -m pip install -r requirements-crossover.txt
```

## Rebuild and analyze

First build the pre-adjudication manifest (165 blocks / 660 probes):

```bash
python analysis/crossover/build_crossed_atomic_control.py \
  --test-parquet data/Kvasir-VQA-x1/annotations/test.parquet \
  --project-root . \
  --output-dir results/generated/crossover/pre-adjudication
```

Apply the model-output-blind adjudication rule and align each model's raw
outputs to the retained 164-block manifest:

```bash
python analysis/crossover/adjudicate_atomic_manifest.py \
  --blocks results/generated/crossover/pre-adjudication/crossed_blocks_165.jsonl \
  --manifest results/generated/crossover/pre-adjudication/crossed_manifest_660.jsonl \
  --raw-dir results/generated/crossover/raw \
  --output-dir results/generated/crossover/adjudicated
```

The raw directory must contain one 660-row JSONL per model, with the same
probe IDs as the manifest. Analyze the adjudicated 656-row-per-model files:

```bash
python analysis/crossover/analyze_crossed_results.py \
  --manifest results/generated/crossover/adjudicated/crossed_manifest_656.jsonl \
  --model Qwen3-VL-8B=results/generated/crossover/adjudicated/raw/qwen3_vl_8b.jsonl \
  --model Qwen2.5-VL-7B-Kvasir-ft=results/generated/crossover/adjudicated/raw/qwen25_kvasir_ft.jsonl \
  --model MedGemma-4B-Kvasir-ft=results/generated/crossover/adjudicated/raw/medgemma_kvasir_ft.jsonl \
  --model Lingshu-7B=results/generated/crossover/adjudicated/raw/lingshu_7b.jsonl \
  --output-dir results/generated/crossover/analysis \
  --bootstrap-samples 2000 --seed 20260716
```

The analyzer writes aggregate metrics and the paper-facing count-stratum table
to the output directory. It does not write scored row-level JSONL files. The
checked-in `results/atomic_crossover_primary.csv` contains the four model rows
from the final table; no deterministic reference-policy rows are included.

Do not commit files from `results/generated/`. The checked-in aggregate table
is the public result artifact; full manifests and row-level outputs remain
private pending source-data and provider-policy review.
