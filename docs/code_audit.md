# Code Provenance Audit

This repository is a sanitized, reviewer-facing code package for
EndoPremiseBench. It was assembled from the final paper workspace, the original
local development tree, and the two experiment servers used during the deadline
runs.

## Audited Sources

| Source | Role | Public handling |
| --- | --- | --- |
| Original local development tree | Main benchmark construction, prompt, inference, scoring, and paper table scripts. | Core scripts migrated into `data_building/`, `prompts/`, `inference/`, `scoring/`, and `analysis/`. |
| Local `shared_scripts/inference` | Updated open-weight VLM inference runner. | Public runner migrated into `inference/run_transformers_probe_inference.py`. |
| Local `agent_internal/endo_premise_bench/a_group_supplement_20260524` | Supplemental controls, result freezing, shard merging, parse retry, and ID tracking utilities. | Reproducibility logic migrated into `analysis/` and `controls/`; private launch/log/output files excluded. |
| Main experiment server 84 | Primary source for the 20260524-20260601 supplemental result pipeline. | Read-only audit confirmed the same core scripts or older snapshots; public package keeps the newest sanitized local mirror where it is newer. |
| Secondary experiment server 89 | Secondary repair and isolated rerun provenance for missing/pending control IDs. | Only the repair-manifest construction logic is included, as `controls/build_secondary89_repair_manifests.py`. |

## Included From Server/Agent-Internal Runs

The follow-up migration adds the server-provenance utilities that are needed to
understand and reproduce the paper-facing experiment controls:

| Public path | Purpose |
| --- | --- |
| `analysis/build_paper_assets.py` | Frozen 20260525 paper asset builder for main, question-only, wording, coverage, and artifact inventory tables. |
| `analysis/merge_closed_api_shards.py` | Conservative closed/API shard merger by manifest ID order. |
| `analysis/merge_full10k_by_manifest_order.py` | Row-index based merger for the 10k expanded manifest where duplicate IDs can exist. |
| `analysis/repair_parse_retry.py` | Row-index based parse-retry preparation, split, merge, and finalize workflow. |
| `analysis/track_completed_ids.py` | Recomputes completed, failed, pending, duplicate, and unknown IDs from raw JSONL outputs. |
| `controls/build_prompt_sensitivity.py` | Deterministic prompt-sensitivity subset and prompt template freeze. |
| `controls/build_image_mismatch_manifest.py` | Deterministic same-source image-mismatch control manifest builder. |
| `controls/build_secondary89_repair_manifests.py` | Secondary-server repair manifest builder for failed/pending control IDs. |
| `controls/run_closed_api_controls.py` | Sanitized text-only closed/API control runner for wording and question-only controls. |

The public versions default to the repository root via `EPB_PROJECT_ROOT` (or
`.`), use `results/` for generated outputs, and avoid private machine paths.

## Deliberately Excluded

The following classes of files were audited but not migrated into the public
repository:

- Raw model outputs, scored JSONL files, provider response dumps, and generated
  metrics. These are result artifacts, not source code, and can contain
  provider-specific metadata.
- Server logs, scheduler logs, terminal transcripts, and `screen` orchestration
  scripts. They document a local execution environment but do not improve public
  reproducibility.
- Endpoint pool JSON files, private gateway URLs, credentials, and environment
  files.
- Raw endoscopy images, licensed datasets, model checkpoints, and local caches.
- Large internal progress notes that mix operational state with private run
  diagnostics.

## Current Confidence

The public tree now contains the code needed for benchmark construction,
inference, scoring, paper-facing aggregation, supplemental control construction,
closed/API control execution, shard merging, parse repair, and ID audit. The
remaining excluded files are intentionally non-public artifacts or
server-specific launch material rather than necessary paper code.
