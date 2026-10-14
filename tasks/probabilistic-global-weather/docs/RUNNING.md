# Run the track on a standalone Linux host

These commands do not require Slurm or our cluster paths. Run them from the
repository root. The Docker commands document the intended standalone path;
actual Docker/GPU/namespace execution has not been validated. Current validation
uses the Dockerfile-derived Apptainer runtime. See [validation evidence](READINESS.md).

## Prerequisites

- Repository read access for the task code. The public prepared HF data require no authentication.
- Python 3.10+ and `huggingface_hub>=1.0,<2` on the operator host.
- Linux x86-64, Git, Docker Engine with Compose v2, and NVIDIA Container Toolkit.
  The host must permit the Bubblewrap user namespaces used by the verifier;
  a container configuration check alone cannot establish this.
- Four NVIDIA GPUs compatible with the pinned JAX/CUDA runtime, 32 CPU cores
  and 384 GiB RAM. No specific GPU model is required. Reference timing was
  measured on RTX PRO 6000 96-GB GPUs; memory fit and throughput on other
  hardware must be checked. The same wall-clock limits apply.
- 1 TB for the default windowed evaluation, including prepared inputs and the
  configured runtime reserve,
  with separate space for any training data or build/download caches beyond that
  reserve. Retaining all predictions
  or running separate forecast/score phases requires 8 TiB. The full prepared
  resource download is approximately 220.4 GB (evaluation-only: 54.8 GiB);
  ordinary setup does not rebuild historical climatologies.
- Network access during image builds and HF data download. Training
  and independent evaluation run without network access.

Clone the released track from main:

```bash
gh repo clone T0-RSI/ai4sci-tasks
cd ai4sci-tasks
git checkout main
export WEATHER_TASK_DIR="$(pwd)/probabilistic-global-weather"
export TASK_DATA_DIR=/absolute/path/on/large-disk/wn2-data
export WEATHER_RUN_DIR=/absolute/path/on/large-disk/wn2-run
mkdir -p "$TASK_DATA_DIR" "$WEATHER_RUN_DIR/training" \
  "$WEATHER_RUN_DIR/submission" "$WEATHER_RUN_DIR/verifier"
```

Use an empty run directory for a new training run. The starter creates its own
`training/run` directory; resume the same run with `--resume`.

## Optional: check upstream availability for a maintainer rebuild

```bash
python3 "$WEATHER_TASK_DIR/environment/data/check_sources.py"
```

Skip this check when downloading the prepared HF release. It is only needed
when rebuilding resources from upstream. The lightweight check reads only three
upstream metadata objects, needs Python
3.9+ and no GPUs, and exits nonzero if a source is unreachable or violates
the declared metadata policy. Passing it does not verify the underlying data chunks.

ARCO is a rolling archive. The source check permits changes only to its global
`last_updated`, `valid_time_stop` and `valid_time_stop_era5t` attributes. All
remaining metadata, including variable definitions, grid shapes, chunking and
encodings, must retain the frozen schema identity; stable ERA5 coverage must
extend through 2022. Other archives retain exact metadata checks. This handles
normal [ARCO updates](https://github.com/google-research/arco-era5#overview)
without silently accepting schema changes. Metadata checks do not prove all
historical data chunks are unchanged; downloaded files retain their inventories.

## Build both images

```bash
docker build -f "$WEATHER_TASK_DIR/environment/Dockerfile" \
  -t weather-starter:local "$WEATHER_TASK_DIR"
docker build -f "$WEATHER_TASK_DIR/tests/Dockerfile" \
  -t weather-verifier:local "$WEATHER_TASK_DIR"
```

The verifier image includes the cyclone tracker binaries and both pinned Python
environments, so use it for full resource preparation. Dependency installation
requires access to the image registry, package indexes and upstream Git repositories.

## Prepare data once, without GPUs

```bash
python3 -m pip install 'huggingface_hub>=1.0,<2'
bash "$WEATHER_TASK_DIR/environment/data/materialize.sh" "$TASK_DATA_DIR"
```

This anonymously downloads the prepared HF release at the checked-in commit,
verifies the archives and every extracted file, and restores the required layout.
It includes the frozen IBTrACS snapshot. Use `--scope development` or
`--scope evaluation` when provisioning just one runtime; default is `all`.
Rerun the same command to resume. To rehash an installed release offline:

```bash
bash "$WEATHER_TASK_DIR/environment/data/materialize.sh" "$TASK_DATA_DIR" --verify-only
```

See [data preparation](../environment/data/MATERIALIZE.md) for storage, source terms
and the separate maintainer-only upstream rebuild workflow.

## Check configuration and GPU visibility

```bash
docker compose -f "$WEATHER_TASK_DIR/environment/docker-compose.yaml" config --quiet
docker compose -f "$WEATHER_TASK_DIR/tests/docker-compose.yaml" config --quiet
docker run --rm --gpus '"device=0,1,2,3"' weather-starter:local \
  /opt/weather-python/bin/python -c 'import jax; print(jax.devices()); assert len(jax.devices("gpu")) == 4'
```

The task runner normally supplies GPU allocations and submission/log mounts on
top of the Compose definitions. For standalone use, the explicit commands below
supply those settings. Do not assume plain `docker compose up` trains the model.
The commands select GPU indices 0–3; adjust them to the GPUs allocated to you.

## Train and export

```bash
docker run --rm --network=none --gpus '"device=0,1,2,3"' \
  --cpus=32 --memory=384g --shm-size=64g --security-opt seccomp=unconfined \
  --mount "type=bind,src=$TASK_DATA_DIR/training-2017-2020,dst=/workspace/data/training,readonly" \
  --mount "type=bind,src=$WEATHER_RUN_DIR/training,dst=/workspace/research" \
  --mount "type=bind,src=$WEATHER_RUN_DIR/submission,dst=/workspace/submission" \
  weather-starter:local /opt/weather-python/bin/python /workspace/baseline/starter/train.py \
  --data /workspace/data/training --output /workspace/research/run \
  --submission /workspace/submission --seed 0
```

This command trains the default 580-update recipe and exports `infer.sh` and
weights to the host submission directory. It is a substantial GPU job, not a
smoke test. The recipe allows 24 hours of training plus 30 minutes of packaging;
its 87.51 GPU-hour cost estimate combines measured prefix and continuation runs.
The container's default user may own generated files as root on the host.

## Run the independent evaluator

```bash
docker run --rm --network=none --gpus '"device=0,1,2,3"' \
  --cpus=32 --memory=384g --shm-size=64g --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined --security-opt systempaths=unconfined --cap-add SYS_ADMIN \
  --mount "type=bind,src=$TASK_DATA_DIR/evaluation-resources,dst=/benchmark,readonly" \
  --mount "type=bind,src=$WEATHER_RUN_DIR/submission,dst=/workspace/submission,readonly" \
  --mount "type=bind,src=$WEATHER_RUN_DIR/verifier,dst=/logs/verifier" \
  weather-verifier:local bash /tests/test.sh
```

The verifier runs the submitted model independently on eight dates, 56 members
and 15 days, then writes `reward.json`, `status.json` and the detailed `replay/`
reports under the host verifier directory. A failed run can write reward zero;
always inspect `status.json` as well. Use a new empty verifier directory for a
new evaluation. The AppArmor, system-path and `SYS_ADMIN` options let Bubblewrap create the model's
namespace inside Docker on hosts that restrict unprivileged user namespaces (Ubuntu 24.04 by
default); without them `inference.log` shows `bwrap: Failed to make / slave: Permission denied` and
the run scores 0 (observed 2026-09-26). With GPUs attached, the NVIDIA runtime's per-GPU
`/proc/driver/nvidia/gpus/*` mounts additionally stop the namespace from mounting its own procfs
(`bwrap: Can't mount proc on /newroot/proc: Operation not permitted`); `environment/weathernext/wn2/test.sh` detaches them
before inference (observed and fixed 2026-09-30). The 12-hour evaluation budget is separate from
training.

The default evaluator uses lossless state continuation and a configurable disk
allowance, without repeating earlier forecast steps. Existing trained starter
packages need only an inference-entry upgrade, not another training run:

```bash
python3 "$WEATHER_TASK_DIR/wn2/upgrade_inference.py" \
  --source /existing/trained-package --output /new/trained-package-v3
```

Use that new directory as the submission mount. Pass
`WEATHER_EVALUATION_DISK_ALLOWANCE_GB=1000` and
`WEATHER_EVALUATION_DISK_RESERVE_GB=25` into the verifier container. Lower allowances
produce shorter windows; 250 GB is a useful stock-model starting point, subject
to the planner's exact input/package calculation. See the [complete workflow and
accounting limits](../wn2/WINDOWED_EVALUATION.md). Smaller windows may need a longer
evaluation budget and matching runner timeout. Training is unaffected.

Scores, prediction/replay receipts, tracks and tracker logs survive cleanup.
A failed window retains its available artifacts; successful windows cannot be
rescored without new inference. Full retained outputs and split phases require
explicit `--legacy-case-mode` and the larger legacy storage provision.

## Apptainer image construction without Slurm

On a host with Apptainer and permitted fakeroot builds, render the same
Dockerfiles and build directly. This avoids the site-specific Slurm wrapper:

```bash
export WEATHER_IMAGE_DIR=/absolute/path/on/large-disk/weather-images
mkdir -p "$WEATHER_IMAGE_DIR"
for component in environment tests; do
  python3 "$WEATHER_TASK_DIR/slurm/render_apptainer.py" \
    "$WEATHER_TASK_DIR/$component/Dockerfile" \
    "$WEATHER_IMAGE_DIR/$component.def" --context "$WEATHER_TASK_DIR"
  apptainer build --fakeroot --mksquashfs-args '-processors 4' \
    "$WEATHER_IMAGE_DIR/$component.sif" "$WEATHER_IMAGE_DIR/$component.def"
done
```

For full data preparation, use `tests.sif` with the `--image` command in the
data guide. Apptainer build and namespace support depend on host configuration;
the presence of these commands is not evidence of a fresh build on your host.
