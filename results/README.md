# Paper-facing result aggregates

These files are compact aggregates aligned with the final EndoPremiseBench
manuscript. Percentages are rounded to one decimal, as in the paper.

- `main_results.csv`: the 15 complete primary VLM evaluations.
- `rebuttal_sensitivity.csv`: attribute-wise supported accuracy and the
  sensitivity analysis excluding location and morphology/type items.
- `supplementary_controls.csv`: model-level summaries for question-only,
  wording, prompt, answer-position, image-mismatch, and expanded-Kvasir checks.
- `control_coverage.csv`: run and scored-output counts for those six control
  families.
- `atomic_crossover_primary.csv`: count-stratum results for the adjudicated
  2x2 crossover, with block-bootstrap 95% intervals.

No raw or scored JSONL, image, manifest, provider response, server log, or
deterministic reference-policy row is included. The 100.0% rejection values in
the question-only file are actual model-run outcomes; their paired supported
accuracy is 0.0%. They are not deterministic reference policies. The three
deterministic policies discussed in the paper are intentionally omitted from
these tables.

The expanded-Kvasir result uses the frozen 10,000-row manifest, which contains
9,998 unique probe IDs because two supported items were duplicated once each.
Removing those repeats leaves all displayed one-decimal results unchanged.

The aggregate CSVs are not a redistribution of source images or annotations.
Users must obtain source datasets from their providers and follow the
applicable licenses and privacy terms. See `docs/code_audit.md` and the
repository README for the release boundary and analysis entry points.
