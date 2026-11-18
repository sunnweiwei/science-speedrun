# Validation and independent replay

`test_release_contract.py` verifies full-suite weighting/normalization, reordered
IDs, malformed output rejection, checksum failures, label-free inputs, validation
subsets, inventory integrity, safe extraction, role mounts and failure status.
Use the pinned runtime:

```bash
python -m pytest -q tests/test_release_contract.py
bash -n tests/test.sh environment/entrypoint.sh environment/data/materialize.sh
```

The real task baseline is not a constant fixture. After materialization, replay
the immutable model on relocated input-only data:

```bash
# reference-model.tar.gz from the private evaluation repository (hf_release.json "reference_model")
bash "$TASK_DATA_DIR/reference-model/infer.sh" \
  "$TASK_DATA_DIR/verifier" /tmp/qec-predictions
PYTHONPATH=environment/baseline/alphaqubit python -m qecbench.scoring --input "$TASK_DATA_DIR/verifier" \
  --predictions /tmp/qec-predictions --labels "$TASK_PRIVATE_DIR" \
  --config environment/baseline/alphaqubit/score_config.json --output /tmp/qec-report.json
```

The direct commands audit numerical behavior on a trusted host. They are not the
untrusted submission sandbox. Production evaluation uses the separate Harbor
verifier environment. To run the native Docker topology locally:

```bash
export SURFACE_CODE_SUBMISSION="$TASK_DATA_DIR/reference-model"
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml down --volumes
# Main must start alongside the grader. Do not use --abort-on-container-exit:
# the grader exits after publishing results while main is still collecting them.
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml up --build -d
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml wait main
mkdir -p ./verifier-result
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml cp main:/logs/verifier/report.json ./verifier-result/report.json
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml cp main:/logs/verifier/reward.json ./verifier-result/reward.json
docker compose -f tests/docker-compose.yaml -f tests/docker-compose.local.yaml down --volumes
```

Copy individual reports rather than the entire log volume, which also contains
the deliberately read-only relocated submission. Infrastructure errors have a
report but intentionally no `reward.json`.

`main_verifier.py` runs as root only to freeze artifacts, launch UID 10001,
terminate child processes and seal outputs. The inference child has no labels,
network, root privilege, or permission to write reward files. `grader.py` has
labels but never imports submitted code. The tokenized result handshake avoids
using stale reward files. Infrastructure failures intentionally produce no
reward. Invalid models or outputs produce `score=0` with a failure status.

The schema permits arbitrary sample ordering, duration ordering, N and supported
subsets. Full official scoring keeps all 130 conditions; subset scores are
explicit development diagnostics, enabled only with
`--allow-subset-diagnostic --validation`. They do not redefine the benchmark.

Container checks on hosts without Docker must be identified by their actual
engine (for example, Apptainer with isolated read-only binds), and cannot be
reported as a completed native Docker/Harbor engine run. See the release
validation evidence for the actual commands and results.

The native Docker CI builds both current images, downloads and verifies the
pinned public development HF archive, and starts the development image offline.
It also resolves the production Compose topology to check the single-GPU
reservation. Its verifier checks use small synthetic CPU fixtures with
`docker-compose.cpu-smoke.yaml`, whose `!reset` tag deliberately removes that
reservation. These exercise isolation and scoring plumbing, not scientific
accuracy or GPU inference speed. Production/local replay retains exactly one
GPU; it never inherits all host GPUs.

The attention-reference and microbatch-equivalence unit tests explicitly request
JAX `highest` matmul precision only for the compared operations. Their original
FP32 tolerances are unchanged. Production training/inference retain their chosen
precision; GPU default/TF32 differences are not hidden by a global setting.

The bitwise interrupted-versus-continuous optimizer resume test is CPU-only;
run it with `JAX_PLATFORMS=cpu`. A5000 investigation found tiny optimizer-moment
differences after GPU JIT recompilation even at `highest` precision, while RNG,
sampler state, counters, parameters and EMA matched. GPU runs explicitly skip
this deterministic CPU contract; its exact equality checks are retained.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The benchmark evaluates quantum error-correction (QEC) memory decoding on the published Google Quantum AI Sycamore surface-code experiments (*Google Quantum AI*, "Suppressing quantum errors by scaling a surface code logical qubit," *Nature* **614**, 676–681, 2023; upstream archive [Zenodo 6804040](https://zenodo.org/records/6804040)) following the learned neural-decoder protocol of **AlphaQubit** (*Bausch et al.*, "Learning high-accuracy error decoding for quantum processors," *Nature* **635**, 834–840, 2024, `doi:10.1038/s41586-024-08148-8`).

- **130 Hardware Memory Conditions & Even/Odd Split**: Following Bausch et al. (2024), all 130 surface-code conditions (4 distance-3 patches + 1 distance-5 patch $\times$ 2 memory bases $X, Z$ $\times$ 13 odd stabilizer round counts $r \in \{1, 3, \dots, 25\}$) partition the 50,000 hardware shots per condition into even-indexed calibration/training/validation shots (`19,880` train + `5,120` validation) and `25,000` odd-indexed held-out test shots (`3,250,000` total test shots).
- **Keyed HMAC-SHA256 Sample IDs & Temporal Shot-Order Permutation**: In V4, `environment/data/build_release.py` replaces unkeyed hashes with keyed `HMAC-SHA256` digests (`hmac.new(secret_key, f"{folder}:{index}", sha256).hexdigest()[:32]` projected to collision-checked `uint64`) and permutes test rows per condition using a keyed PRNG seeded from `secret_key`. Neither `sample_ids` nor array row indices in `verifier-inputs.tar.gz` reveal original shot indices or temporal drift order; the reverse `(sample_id -> folder, source_shot_index)` mapping resides exclusively in `grader/private_sample_id_map.json` (`TASK_PRIVATE_DIR`).
- **Balanced Logical Success (`P`)**: Each condition's binary observable-flip predictions (`probability >= 0.5`) define a logical failure fraction; averaging uniformly across the 13 durations per patch/basis group, across groups within each code distance (`8` groups for $d=3$, `2` groups for $d=5$), and equally across distances yields:
  $$P = 1 - \frac{1}{2}\left(\text{fail}_{d=3} + \text{fail}_{d=5}\right)$$
  Both `reward.json["score"]` and `reward.json["reward"]` emit `P` directly (matching `harness/ssi_index_v3.py:surface_code()` to $< 10^{-12}$).

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No Log-Fidelity Fit Gate on Primary Score**: Exponential logical-error-per-round ($\varepsilon_d$) curve fitting requires logical fidelity $F = 1 - 2 P_{\text{fail}} > 0$ ($P_{\text{fail}} < 0.5$) at every round count. High-noise or under-trained decoders can cross $P_{\text{fail}} \ge 0.5$ at $r=25$ while still producing well-defined per-shot binary decisions; gating the primary reward on curve-fit convergence would create discontinuous cliff penalties unrelated to true logical error rate.
- **No Arbitrary Anchor Clipping or Calibration Penalty in `reward.json`**: While MWPM ($L = 0.7160463461538461$) and published tensor-network decoding ($U = 0.7615159615384615$) provide physical reference anchors, `P` itself is already bounded on $[0, 1]$ with clear physical meaning (logical survival probability balanced across $d=3$ and $d=5$). Over-constraining `reward.json` with Brier/ECE thresholds would penalize sharp deterministic decoders such as MWPM or Lookup/Union-Find hybrids.

### 3. Diagnostic Signals Retained in report.json

`report.json` preserves rich secondary scientific diagnostics without contaminating the single scalar V3/V4 reward:
- **Anchor-Normalized Reference Score**: `score` and `score_unclipped` relative to the frozen MWPM ($L$) and tensor-network ($U$) anchors (`score_config.json`).
- **Per-Distance & Per-Condition Error Breakdown**: `failure_by_distance` (`d3`, `d5`), `balanced_logical_failure`, and `per_condition` exact error/shot counts across all 130 hardware conditions.
- **Physical Scaling Fits (`scientific_metrics`)**: Fitted logical error per round ($\varepsilon_3, \varepsilon_5$) and error-suppression factor $\Lambda_{3/5} = \varepsilon_3 / \varepsilon_5$ when log-fidelity fits are defined (`status: "available"` vs `"unavailable"`).
- **Probability Calibration (`calibration`)**: Pooled Brier score, negative log-likelihood (NLL), and 10-bin expected calibration error (ECE).
