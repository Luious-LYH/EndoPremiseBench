# Example Commands

These commands assume you run them from the repository root after placing the
source datasets under `data/`.

## Build Benchmark Manifests

```bash
python data_building/profile_premise_data.py \
  --config configs/premise_probe_v2.json \
  --output-candidates results/premise_candidates_v2.jsonl

python data_building/build_balanced_subset.py \
  --input results/premise_candidates_v2.jsonl \
  --output results/premise_balanced_main_v2.jsonl

python data_building/build_control_manifests.py \
  --input results/premise_balanced_main_v2.jsonl
```

## Score Existing Outputs

```bash
python scoring/parse_and_score.py \
  --input results/model_raw.jsonl \
  --output results/model_scored.jsonl \
  --metrics-output results/model_metrics.json
```

## Closed/API Inference

```bash
export ENDOPREMISE_API_KEY=...
python inference/run_closed_api_premise_inference.py \
  --provider openai-compatible \
  --api-type openai_compatible \
  --base-url https://api.example.com/v1 \
  --api-key-env ENDOPREMISE_API_KEY \
  --model MODEL_NAME \
  --input results/premise_balanced_main_v2.jsonl \
  --output results/api_model_raw.jsonl
```
