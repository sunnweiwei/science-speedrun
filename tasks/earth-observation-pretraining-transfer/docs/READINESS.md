# Delivery readiness

As of 2026-10-09, this track is ready for review and merge under the agreed
best-effort container policy. Docker/Harbor execution is not supported on the
development cluster and is not a merge gate. This records source and CPU
validation, not an end-to-end container acceptance or a new scientific result.

## Completed delivery work

- Model-independent task name and directory; OlmoEarth-v1 Large remains the baseline.
- Task authors, offline agent/verifier configuration and a default 48-hour external
  development budget in `task.toml`. Providers can configure their run budget.
- Task-root build contexts, development-only read-only mounts, writable research
  mount, and a separate private-data verifier mount. Missing host paths fail
  instead of silently creating empty data directories.
- Standard CPU-only `environment/data/materialize.sh` entry point. The operator
  guide documents preparation and separation of development and private views.
- A publisher that copies a candidate, checks CPU inference on at most two
  development inputs, validates the output and atomically exchanges directories.
  Previous submissions are retained. Unsupported exchange leaves the old package
  intact; a submission directory must not itself be a bind mount.
- Missing/malformed packages, ineligible provenance, and malformed predictions
  receive reward 0 with a reason. Infrastructure and ambiguous process failures
  fail without a new reward. Existing rewards cannot be overwritten. Deferred
  scoring fingerprints include the new contract checker.

## Verification performed

Twelve focused CPU tests passed in the existing pinned Python runtime:

```bash
cd earth-observation-pretraining-transfer
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -p test_delivery.py -v
python -m unittest discover -s tests -p test_staged_verifier.py -v
python -m unittest discover -s tests -p test_screening.py -v
python -m unittest discover -s tests -p test_seed_comparison.py -v
```

The five delivery tests check replacement and rollback preservation, unsupported
filesystem exchange, malformed predictions versus missing data, zero reward for
invalid packages, and replay-error routing. The staged-verifier test exercises
real CPU probes on synthetic data, single-stage/deferred score agreement,
label-free replay, tamper rejection and ineligible provenance. Six development
tests cover test-row exclusion, source preservation, cached-feature identity and
paired comparisons. No model training or benchmark experiment was run.

Additional checks passed: TOML parsing and policy fields, YAML parsing and mount
separation for both Compose manifests, Python AST parsing, shell syntax,
Dockerfile COPY source existence, and publisher/replay CLI imports from a
reconstructed workspace. YAML parsing does not establish Docker Compose runtime
compatibility. No Docker build, container launch or Harbor run was claimed.

## Remaining non-blocking limitations

- The controller must enforce the declared offline policy, allocate GPUs,
  snapshot the published submission and collect verifier output. Docker namespace
  settings follow the weather verifier but have not been exercised here.
- Unexpected inference exits remain unscored until an operator identifies whether
  the cause is the model or infrastructure. Do not turn arbitrary runtime failures
  into model-quality measurements.
- The starter evidence remains one adaptively selected seed: 77.034884% AWF
  validation accuracy and 21.459503 LFMC validation MAE. AWF has no independent
  test split; public LFMC test was exposed during earlier development. No new
  final reward, independent-seed replication or untouched-test claim is made.
- Smaller-card feasibility remains unestablished. Runtime/cost measurements are
  documented with the baseline, including their exclusions.
- Historical external-LFMC acquisition code references an absent
  `results/external-lfmc/metadata-selection-policy.json`. It is outside the
  delivered frozen-probe workflow and is not copied to the agent workspace.
