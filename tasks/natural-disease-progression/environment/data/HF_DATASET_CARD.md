---
license: other
task_categories:
  - tabular-classification
tags:
  - clinical
  - restricted
---

# Natural Disease Progression controlled task data

This private release is derived from MIMIC-IV 3.1. It is not open data. Access may be granted only after an administrator verifies that each individual accessor is a PhysioNet credentialed user, has completed the required CITI Data or Specimens Only Research training, and has signed the current MIMIC-IV/PhysioNet Credentialed Health Data Use Agreement. Access cannot be shared within a team.

This release is the development-only longitudinal-v2 snapshot. It contains train data and labels, public validation inputs, validation scorer labels, and train-derived public scoring configuration. It contains no test input or truth, raw patient identifier, controller database, or raw MIMIC table. `working.duckdb` is a build intermediate and is intentionally not distributed.

Runtime use is offline. See the matching code repository's `docs/data-release.md` and `runtime-manifest.json` for provenance and file hashes.
