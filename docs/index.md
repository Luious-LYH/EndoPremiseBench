# EndoPremiseBench Project Page

EndoPremiseBench is a diagnostic benchmark for endoscopic VQA systems that
separates useful visual answering from unsupported premise continuation.

![Benchmark overview](../assets/readme/benchmark_overview.jpg)

## Core Idea

Many VQA questions quietly assume that an object, attribute, or clinical finding
is visible. EndoPremiseBench converts source-supported and source-unsupported
endoscopic questions into a shared MCQ interface with one `Not applicable`
option. A robust system should answer supported questions and reject false
premises.

## Main Readout

![Utility-restraint profile](../assets/readme/result_panel.jpg)

The benchmark reports supported-question accuracy and false-premise rejection
together. Unsupported-attribute exposure is tracked explicitly so a model cannot
look strong by answering confidently under unsupported premises.

## Reproduce

1. Install dependencies with `pip install -r requirements.txt`.
2. Place licensed datasets under `data/` or edit
   `configs/premise_probe_v2.json`.
3. Run the construction scripts in `data_building/`.
4. Run inference with `inference/`.
5. Score and aggregate with `scoring/` and `analysis/`.

The repository does not include source endoscopy images, model weights, raw API
logs, or private review material.

## Release Checklist

- Code paths are repository-relative by default.
- API credentials are read only from environment variables.
- Generated artifacts are ignored by Git unless deliberately added.
- `docs/software_manifest.tsv` records the staged software package inventory.

