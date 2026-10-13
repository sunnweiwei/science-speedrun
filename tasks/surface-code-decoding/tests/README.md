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
