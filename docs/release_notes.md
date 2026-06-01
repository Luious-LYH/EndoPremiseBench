# Release Notes

## 2026-06-01

- Reorganized the public code tree under the final paper workspace.
- Added reviewer-facing README visuals and a lightweight project-page overview.
- Added the staged software manifest at `docs/software_manifest.tsv`.
- Updated the closed/API inference runner with sharding, resume, and external
  success-skip support while keeping public defaults repository-relative.
- Added sanitized supplemental-control and server-provenance utilities from the
  final 84/89 experiment audit, with inclusion/exclusion rationale documented
  in `docs/code_audit.md`.

## Data And Artifact Policy

The public repository is code-first. It excludes licensed endoscopy images,
model checkpoints, raw API/provider logs, and local generated outputs by
default.
