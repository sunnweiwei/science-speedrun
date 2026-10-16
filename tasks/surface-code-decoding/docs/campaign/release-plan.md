# Surface-code and massspec release implementation

> Historical document: the warm-start proposals below describe the original v1
> release. Current v1.2 requires random initialization, supplies no trained
> agent weights, and allows 8 H100 GPUs × 48 hours. The current
> [runtime contract](../environment/contract.md) supersedes those proposals;
> the original measured results remain provenance.

User authorization, 2026-09-22: complete both tracks' prepared Hugging Face
datasets, materialization, normalized score, science figures, and all missing
Surface-code Harbor infrastructure. Use the strongest currently reproduced
reference as the upper score anchor (not perfect prediction).

## Fixed scope and evidence

- No new accuracy campaign or checkpoint selection. AlphaQubit v1 remains frozen.
- Surface-code: balanced accuracy over all 13 durations, equal groups within
  distance, equal d3/d5. MWPM lower, same-test audited released TN upper.
- Massspec: full 17,556-query Hit@1; DeepSets lower 0.04004329004329004;
  strongest full-test GLACIER plus strict-clean MIST upper 0.4939052175894281.
- Score = clip((performance - lower)/(upper - lower), 0, 1). Retain original
  scientific metrics, unclipped values, saturation and status. Infrastructure
  errors must not masquerade as measured zero quality.
- Public historical tests are isolated at runtime, not represented as secret.
  Private HF grader repositories contain labels; predictor receives no labels.
- Surface-code warm-start checkpoints authorized; massspec remains from-scratch.
- Development data licenses and provenance retained. No test prediction caches
  or unrelated restricted third-party weights in public releases.

## Workstreams

1. Surface implementation: physical data splits, arbitrary-N model inference,
   separated-data training, native Harbor environment/verifier and score tests.
2. Massspec implementation: package existing materialized data, HF downloader,
   source fallback, normalized grader/development reports and regression checks.
3. Figures: real validation examples, editable SVG, rendered PNG and provenance.
4. Integration: publish datasets, pin HF commits, isolated clean replay, audit,
   docs and GitHub PRs. Root agent independently checks delegated changes.

## Published release locations

- Corning/ai4sci-surface-code-decoding (public development and verifier inputs)
- Corning/ai4sci-surface-code-decoding-evaluation (private grader labels)
- Corning/ai4sci-massspecgym-retrieval (public development and verifier inputs)
- Corning/ai4sci-massspecgym-retrieval-evaluation (private grader labels)

## Resource and validation plan

Reuse complete local assets and saved models; CPU for packing/checksums/scores,
GPU only for frozen-model replay. Reserve up to 12 GPU-hours initially for
integration checks; expand only if measured runtime justifies it. Existing
training budget is not renewed. Use Slurm for GPU work. CS has no Docker engine. Full scientific GPU replay therefore uses isolated
Apptainer on CS, while the actual native Docker image and two-service topology
are built and exercised on a GitHub-hosted runner.

Checks progress from unit/format tests to complete dataset verification, HF
round-trip, real relocated inference and full scoring. Completion requires
published pinned assets, usable launch commands, actual model replay, figures,
score anchor verification and pushed PRs. A passing mock is insufficient.

## Completed delivery

- Both public HF data repositories and separately private evaluator repositories
  are published, with pinned revisions, sizes, SHA-256 and exact inventories.
  Real complete downloads and verified reuse passed for both tracks. Surface
  public pin: 18a4b988ad6cd262a6c92b0a7df356af7d0fb434; private pin:
  b07dd3e6a484c3df76f2ed90bbf97c72e446dc36. Massspec public pin:
  4a0c890f9a27685b1cffb5b131a501828abb4d1f; private pin:
  061195f660391d40fa18a422edf175699fe0d771.
- Science figures use actual validation examples and ship editable SVG,
  PNG/PDF, generating code and provenance, with previews on HF.
- Surface-code has a native Harbor task, physically split train/validation/test,
  portable ten-specialist warm starts, arbitrary-N inference, a working
  separated-data optimizer example and a label-isolated trusted verifier.
- Independent Surface-code audit verified all 390 split-condition payloads,
  all 84,512,060 parameters, allowed noise resources and exact score anchors.
- Complete frozen GPU replay (Slurm 12682867) covers 3.25 million test shots.
  P=0.7550636538461539, score=0.858096277311141. Historical P=0.7550607692307693
  and score=0.8580328368056459 remain fixed. There are 25 changed decisions;
  old/new adapters are bit-identical on 75,000 shots in the same-process control
  (12683095). Cross-run numerical cause remains unproven; no result selection.
- Real one-step training and 39-shot shuffled/reversed-order inference passed
  (12683009), including independent model reload, manifest and parameter audit.
  These are implementation checks, not an accuracy-improvement claim.
- Final targeted Python suite: 117 passed, 4 explicitly optional skips; shell
  syntax and diff checks passed. Massspec suite: 44 passed.
- Actual native Docker build and supervisor/grader security checks passed in
  CI run 35697461569: valid model, invalid submission, and infrastructure failure.
  Synthetic fixture accuracy is not used as scientific baseline evidence.
- Massspec full frozen scratch GLACIER/DeepSets rankings were rescored on all
  17,556 queries: Hit@1=0.4032239690134427, score=0.8002008032128514. A fresh
  offline full DeepSets replay reproduced every scientific metric; 28 complete
  rankings differ only below rank 183, with every top-20 ranking unchanged.
- Fixed deployment issues found by real acceptance: externalized massspec HF
  cache; QEC reuse now validates trusted archive inventory without redundant
  extraction/NFS deletion; Docker evidence export preserves read-only artifacts.
- English PRs: existing Surface-code #24 and new massspec #45. Neither is merged.
  Detailed validation scope is in each track's release-acceptance JSON.
- Durable raw evidence remains outside Git under the project artifact root:
  ai4sci-artifacts/benchmark-release-20260922. Git contains code, documentation,
  compact acceptance evidence and figures, not datasets or model weights.
