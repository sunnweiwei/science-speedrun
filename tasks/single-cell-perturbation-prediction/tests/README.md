# Verifier

Two containers built from `tests/Dockerfile` (same runtime as the dev image):

- **main** — mounts the public test inputs (`$TASK_DATA_DIR/test`) at `/benchmark` and SE-600M at
  `/opt/se600m`, copies nothing else; runs `bash /workspace/submission/infer.sh` **twice** per
  context as uid 10001 with no network, one GPU, and fresh ephemeral `HOME`/`TMPDIR` per pass
  (**Pass 1** on the full `requests.jsonl`, **Pass 2** on a deterministic shuffled proper subset of
  `requests.jsonl` in a fresh process); validates both `predictions.h5ad` and
  `predictions_subset.h5ad` (storage safety, gene order, cell counts, finiteness); seals both
  outputs + SHA-256 hashes into the `sealed` volume; waits for the grader.
- **grader** — mounts the same public inputs and the private responses (`$TASK_PRIVATE_DIR`);
  checks provenance hashes; enforces the two-pass per-request query-independence gate via
  `query_independence.compare_prediction_files` (`atol=1e-4, rtol=1e-4`, rejecting cross-query
  shrinkage / batch-coupled models with `score = reward = 0.0`); scores every context with the SSI
  V3 dual-skill metric (`population_metrics_v2.py`, `calibrated_reward_v2.py:aggregate_contexts_v3`)
  and retains v2 WMSE-only and v1 `cellscore.py` (cell-eval 0.8.2) diagnostics; writes
  `reward.json`, `report.json`, `agent_feedback.json` into the sealed results, which main copies to
  `/logs/verifier`.

V3/V4 primary score per context (`near`, `mid`, `far`):
1. `amplitude_skill = (N - WMSE) / (N - P)`: cross-fitted weighted MSE of the per-target mean
   `log1p(CP10k)` profile (final-subset targets; the target's readout gene, resolved through its
   Ensembl ID, is excluded), with squared min-max `|t|` weights (Scanpy `t-test_overestim_var`,
   controls excluded) from one half of the real cells and the truth mean from the other half
   (5 seeded splits × 2 directions), calibrated between generic perturbation mean `N` and
   same-target repeat `P`.
2. `identification_skill = (NIR - 0.5) / (repeat_NIR - 0.5)`: Euclidean nearest-interventional
   retrieval rate (`NIR`, half-credit ties, all roster target genes excluded from the retrieval
   panel), calibrated between target-blind chance `0.5` and split-half repeatability `repeat_NIR`.
3. **Pre-clip average, then zero floor**: `signed_context = 0.5 * amplitude_skill + 0.5 * identification_skill`,
   `context_score = max(0.0, signed_context)` (signed skills are averaged **before** clipping at zero).
4. Overall `score = reward = clip(mean(context_score), 0.0, 1.0)`, matching `ssi_index_v3.cell()`
   within `< 1e-12`.

The reference bundle (targets, readout gene indices, per split/direction weights and truth means,
`N`, `P`, gap bootstrap interval) is built once from the private responses and cached under
`<private>/<ctx>/wmse_v2/`; with a read-only private mount, build it first:
`python tests/population_metrics_v2.py --private P --benchmark B`.

`reward.json`: `score` and `reward` (identical V3 task score), `v3_<ctx>`, `v3_<ctx>_signed`,
`v3_<ctx>_amplitude_skill`, `v3_<ctx>_identification_skill`, v2 numbers (`v2_reward`, `v2_<ctx>`,
`v2_<ctx>_raw_skill`, `v2_<ctx>_nir`, `v2_<ctx>_repeat_nir`), and v1 numbers (`v1_reward`,
`v1_leaderboard_reward`, `v1_<ctx>_final`, `v1_<ctx>_leaderboard`). A context whose Pass 1
inference fails gets `invalid_submission` (`context_score = 0.0`, kept in the denominator) without
hiding the others; any submission that violates two-pass `query_independence` receives top-level
`score = 0.0, reward = 0.0`; scorer, reference or infrastructure failures raise and leave no reward.

## Running locally

```
export TASK_DATA_DIR=/path/to/DATA_ROOT/dev TASK_PRIVATE_DIR=/path/to/DATA_ROOT/private
docker compose -f tests/docker-compose.yaml -f local-override.yaml up --abort-on-container-exit
```
where `local-override.yaml` mounts the package at `/workspace/submission` in `main`. Unit tests:
`bash tests/run_unit_tests.sh vcz-verifier:latest`.

`agent_feedback.json` holds the v1 leaderboard-subset scores only. Under TASK_STANDARD.md the
controller scores snapshots of `/workspace/submission` without telling the agent, so nothing is
returned by default; a runner that does return feedback should return only this file.
`report.json` and the final-subset scores stay with the operator.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The V4 primary score (`score` / `reward` in `reward.json`, matching `ssi_index_v3.cell()` within `< 1e-12`) evaluates zero-shot transcriptomic perturbation prediction across three held-out cellular contexts (`near`, `mid`, `far`) by combining two experimentally calibrated skills per context before zero-flooring:

$$\text{amplitude\_skill}_c = \frac{N_c - \text{WMSE}_c}{N_c - P_c}, \qquad \text{identification\_skill}_c = \frac{\text{NIR}_c - 0.5}{\text{repeat\_NIR}_c - 0.5}$$

$$\text{signed\_context}_c = \frac{1}{2}\left(\text{amplitude\_skill}_c + \text{identification\_skill}_c\right), \qquad \text{context\_score}_c = \max\left(0.0,\; \text{signed\_context}_c\right), \qquad S = \operatorname{clip}_{[0,1]}\!\left(\frac{1}{3}\sum_{c} \text{context\_score}_c\right)$$

- **Calibrated Weighted MSE (`amplitude_skill`)**: Directly adapts the negative-to-positive experimental anchor calibration framework of **Shift Bioscience (*Nature Biotechnology* 2026, DOI: `10.1038/s41587-026-03307-w`, official commit `15361aec8392fb45a75c8c9da88ca5e6510fca97`)**, using Scanpy `t-test_overestim_var` squared min–max $|t|$-weights cross-fitted across 5 seeded split-halves × 2 directions (`population_metrics_v2.py`) so measurement noise cannot correlate gene weights with target means, while excluding each knockdown target's direct cis-readout gene resolved via stable Ensembl ID.
- **Calibrated Nearest-Interventional Retrieval (`identification_skill`)**: Directly motivated by **Ahlmann-Eltze, Huber & Anders (*Nature Methods* 2025, DOI: `10.1038/s41592-025-02772-6`)**, which demonstrated that deep perturbation models frequently collapse toward the generic post-perturbation mean (`N`) and match or trail simple linear/mean baselines on aggregate distance metrics while failing to distinguish *which* gene was perturbed. Calibrating Euclidean `NIR` (with all roster target genes masked out) between target-blind chance (`0.5`) and biological replicate repeatability (`repeat_NIR`) directly measures target-discriminative specificity on the same `[0, 1]` anchor scale.
- **Pre-clip signed averaging**: Averaging signed `amplitude_skill` and `identification_skill` *before* applying `max(0.0, ...)` prevents a noisy high-variance guess with disastrous WMSE (`amplitude_skill << 0`) from scoring positively solely through slight retrieval ordering, and conversely penalizes a shrunk constant-like predictor whose `identification_skill <= 0`.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **No single-cell optimal-transport / MMD / E-distance gate in the primary scalar**: On pooled CRISPRi scRNA-seq screens with technical batch structure and variable UMI depth, full cell-cloud distribution distances are dominated by library-size and basal-cluster variance rather than perturbation-specific transcriptional shifts (*Nature Biotechnology* 2026, `10.1038/s41587-026-03307-w`; *Ahlmann-Eltze et al., Nature Methods* 2025). Calibrating pseudobulk `log1p(CP10k)` responses against split-half experimental repeats isolates reproducible biological signal.
2. **Two-pass per-request query independence (`query_independence.py`) instead of single-request container restarts**: Restarting Python/CUDA and reloading the 11.5 GB `SE-600M` checkpoint 900 times (`3 contexts × 300 requests`) would exceed reasonable GPU wall time without changing legitimate predictions. Running **Pass 1 (full 300-request manifest)** and **Pass 2 (a deterministic shuffled proper subset in a fresh isolated process with ephemeral `HOME`/`TMPDIR`)** and enforcing `np.allclose(..., atol=1e-4, rtol=1e-4)` on per-request mean `log1p(CP10k)` profiles blocks transductive cross-query shrinkage, batch-mean centering, and target-panel adaptation while preserving fast batched GPU inference for compliant models seeded per `request_id`.

### 3. Diagnostic Signals Retained in report.json

`report.json` preserves the complete multi-resolution audit trail without altering `score`:

- **Per-context V3 & V2 components**: `WMSE`, `negative_loss` (`N`), `repeat_loss` (`P`), `gap_lower95`/`gap_upper95` (5,000-draw paired target bootstrap CI), `amplitude_skill`, `NIR`, `repeat_NIR`, `identification_skill`, `signed_context`, `context_score`, `per_target_loss`, and `per_target_nir`.
- **Two-pass Query Independence audit (`query_independence`)**: `status`, `passed`, `checked_requests`, `changed_requests`, `max_difference`, and any offending `request_id`s.
- **Full `cell-eval` 0.8.2 Generalist panel & Q1–Q3 biological strata (`v1_*`)**: `des` (`overlap_at_N`), `pds` (`discrimination_score_l1`), `mae`, `pearson_delta`, `spearman_lfc`, `auprc_de`, and `spearman_effect` disaggregated across `leaderboard`/`final` subsets, effect-pattern strata (`conserved-strong`, `context-variable`, `weak`), and `seen` vs. `withheld` targets.
