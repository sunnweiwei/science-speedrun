# Large WorldCover and moisture starter

This baseline trains an OlmoEarth-v1 Large encoder entirely from random
initialization: 6,000 WorldCover updates followed by 1,000 updates predicting
NDMI mean/standard deviation alongside WorldCover. The exported encoder has
307,770,368 parameters and produces 1,024-dimensional frozen features for
both AWF and LFMC. Training heads are omitted from the package.

[Recorded results and limitations](RESULTS.md) describe the measured
validation performance. The commands below run inside the prepared agent
environment. The pinned upstream runtime is at `/opt/olmoearth`, and its Python
is already on `PATH`. The controller prepares and mounts the source data
read-only at `/workspace/data/19837`; the agent does not install the runtime
or download datasets.

## Recipe

| Setting | Shared configuration |
|---|---|
| Encoder | Official v1 Large; width 1,024, 24 layers, 16 heads, MLP ratio 4, drop path 0.1 |
| Initialization | Scratch; initialization seed 12536, data seed 3622 |
| Source data | 19,837 geographically filtered samples from two pinned public pretraining shards |
| Cache | Two deterministic 16×16 crops per sample; 39,674 rows; crop seed 42017 |
| Inputs | Sentinel-2 L2A and Sentinel-1; patch size 4, twelve temporal slots, observed-band masks |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8, weight decay 0.002; gradient clipping 1 |
| Batch and precision | Global batch 64, microbatch 16, accumulation 4; FP32 parameters, BF16 encoder autocast, FP32 heads/loss |
| Augmentation/readout | Joint flips/90-degree rotations; native spatial pooling with per-sample observed-modality correction |

| Phase | WorldCover pretraining | Moisture adaptation |
|---|---|---|
| Updates | 6,000 | 1,000 additional; 7,000 cumulative |
| Input order | Source order, oldest first | Reverse observations and validity masks together, newest first |
| Loss | `1.0 × WorldCover soft CE` | `0.1 × WorldCover soft CE + 1.0 × moisture MSE` |
| Spectral-moment loss | Weight 0 | Weight 0 |
| Encoder/head peak LR | 3e-5 / 1e-3 | 1e-5 / 1e-3 |
| Schedule | Cosine horizon 6,000, warmup 300, final ratio 0.1 | New cosine horizon 1,000, warmup 100, final ratio 0.1 |
| Heads | Linear WorldCover and inactive spectral heads | Retain old heads; add a linear 1,024→2 moisture head |
| Transition | Fresh scratch initialization | Restore encoder, old heads, sampler and RNG; initialize only moisture head with seed 12536; reset AdamW |
| Selected checkpoint | Parent at update 6,000 | Exactly local update 1,000 |

WorldCover supplies eleven-class fractions per patch, weighted by recognized
pixel coverage. Unknown codes have zero weight. Moisture targets come from
the same pretraining imagery: recover physical Sentinel-2 B8A/B11 values and
compute NDMI `(B8A-B11)/(B8A+B11)`. Use observed, finite, nonnegative pixels
whose band sum exceeds 0.001 times the sum of the two normalization-band means.
Average valid pixel ratios within each 4×4 patch and temporal slot, then compute
the temporal mean and population standard deviation. They require at least one
and two valid slots, respectively.

Moisture loss is standardized squared error, weighted by WorldCover known
coverage times the summed tree/shrub/mangrove fractions (classes 10, 20, 95).
Per-channel target statistics are fitted on valid pretraining targets only,
with standard-deviation floor 0.01. The denominator covers the complete
accumulated batch. Crops without woody moisture supervision remain in training
for WorldCover. All heads are linear.

Inputs use upstream COMPUTED normalization; pretraining Sentinel-1 is already
in dB. Synthetic month positions and the downstream adapter remain unchanged.
AWF/LFMC labels never train the encoder or auxiliary heads. WorldCover maps and
moisture targets are training-only dependencies; inference uses only imagery
and observation masks.

## Prepare the derived cache and targets

Run these blocks in one shell. Use fresh cache/run directories and unused
package names for each independent reproduction; the defaults below assume
unused output paths. Preparation is CPU-only and writes derived data under
`/workspace/research`; the mounted source dataset remains unchanged.

```bash
set -euo pipefail
OLMO_STARTER=/workspace/baseline/starter
OLMO_UPSTREAM=/opt/olmoearth
OLMO_RESEARCH=/workspace/research/starter
OLMO_CACHE="$OLMO_RESEARCH/worldcover-cache"
OLMO_MOISTURE="$OLMO_RESEARCH/moisture-targets"
OLMO_RUNS="$OLMO_RESEARCH/runs"
OLMO_EVAL=/workspace/research/validation
OLMO_PACKAGES="$OLMO_EVAL/packages"

mkdir -p "$OLMO_RESEARCH"
export PYTHONHASHSEED=0 OLMO_INFERENCE_HASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CUDA_VISIBLE_DEVICES='' python "$OLMO_STARTER/prepare_alignment.py" \
  --source /workspace/data/19837 --output "$OLMO_CACHE" \
  --crop-size 16 --crops-per-sample 2 --seed 42017 --workers 2
CUDA_VISIBLE_DEVICES='' python "$OLMO_STARTER/prepare_moisture.py" \
  --cache-manifest "$OLMO_CACHE/manifest.json" --output "$OLMO_MOISTURE" \
  --vegetation-classes 10 20 95 --min-sum-fraction 0.001 \
  --target-std-floor 0.01 --chunk-rows 128
```

The cache and target manifests record source identities, normalization, crop
mapping, masks, target statistics and preparation costs. Preserve them with
the reproduction. The supplied source data already applies the declared
geographic exclusions; retain those exclusions when changing the recipe.

## Train both phases

Training requires one GPU. Preserve the environment's assigned
`CUDA_VISIBLE_DEVICES`; `cuda:0` selects its first visible GPU.

```bash
OLMO_INIT_SEED=12536
OLMO_DATA_SEED=3622
OLMO_NAME="starter-large-seed$OLMO_INIT_SEED"
OLMO_RANDOM="random-large-seed$OLMO_INIT_SEED"
OLMO_PHASE1="$OLMO_RUNS/$OLMO_NAME-worldcover"
OLMO_PHASE2="$OLMO_RUNS/$OLMO_NAME-moisture"

python "$OLMO_STARTER/train_alignment.py" \
  --upstream "$OLMO_UPSTREAM" --cache-manifest "$OLMO_CACHE/manifest.json" \
  --output "$OLMO_PHASE1" --variant large \
  --init-seed "$OLMO_INIT_SEED" --data-seed "$OLMO_DATA_SEED" \
  --steps 6000 --stop-at 6000 --warmup 300 \
  --global-batch 64 --microbatch 16 --lr 3e-5 --head-lr 1e-3 \
  --weight-decay 0.002 --max-grad-norm 1 --final-lr-ratio 0.1 \
  --worldcover-weight 1 --moments-weight 0 --moment-std-floor 0.05 \
  --save-every 1000 --log-every 10 --device cuda:0

OLMO_PARENT_SHA256="$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' \
  "$OLMO_PHASE1/checkpoints/step006000.json")"

python "$OLMO_STARTER/train_alignment.py" \
  --upstream "$OLMO_UPSTREAM" --cache-manifest "$OLMO_CACHE/manifest.json" \
  --output "$OLMO_PHASE2" --variant large \
  --initialize-from "$OLMO_PHASE1/checkpoints/step006000.pt" \
  --parent-sha256 "$OLMO_PARENT_SHA256" \
  --init-seed "$OLMO_INIT_SEED" --data-seed "$OLMO_DATA_SEED" \
  --head-init-seed "$OLMO_INIT_SEED" \
  --moisture-manifest "$OLMO_MOISTURE/manifest.json" \
  --temporal-order newest-first \
  --steps 1000 --stop-at 1000 --warmup 100 \
  --global-batch 64 --microbatch 16 --lr 1e-5 --head-lr 1e-3 \
  --weight-decay 0.002 --max-grad-norm 1 --final-lr-ratio 0.1 \
  --worldcover-weight 0.1 --moments-weight 0 --moisture-weight 1 \
  --moment-std-floor 0.05 --save-every 500 --log-every 10 --device cuda:0
```

Phase two verifies the parent checkpoint's full digest, completion receipt and
archived training source. Keep the phase-one run and its `launches` directory
available through export. Each reproduction derives the parent digest from its
own saved checkpoint. Keep the same source files across both phases and export;
ordinary resumes enforce exact recipe/source identity.

## Export research packages

```bash
CUDA_VISIBLE_DEVICES='' python "$OLMO_STARTER/export_alignment.py" \
  --checkpoint "$OLMO_PHASE2/checkpoints/step001000.pt" \
  --upstream "$OLMO_UPSTREAM" --output "$OLMO_PACKAGES/$OLMO_NAME"
CUDA_VISIBLE_DEVICES='' python "$OLMO_STARTER/export.py" \
  --upstream "$OLMO_UPSTREAM" --variant large --seed "$OLMO_INIT_SEED" \
  --output "$OLMO_PACKAGES/$OLMO_RANDOM"
```

The first package contains the selected 7,000-update encoder and provenance
for both phases. The second reconstructs its architecture-matched random
initialization with the same seed and builder order. Packages are usable after
relocation and follow the [encoder contract](../../contract.md).

Use the same trained package for both downstream tasks. Follow the
[evaluation guide](../../evaluation/README.md) to evaluate these research
packages against matched random, raw features and trivial predictors on the
complete train/validation populations. Keep inference batch size 16, hash seed 0
and GPU type fixed across paired models. Checkpoint and hyperparameter selection
use validation; final LFMC test scoring follows selection.

## Publish the selected package

After choosing a candidate using validation, publish it with:

```bash
python "$OLMO_STARTER/publish.py" \
  --candidate "$OLMO_PACKAGES/$OLMO_NAME" \
  --check-input /workspace/data/eval/awf/inputs.h5 \
  --submission /workspace/submission
```

The publisher copies the candidate beside the submission, checks required files,
and runs CPU inference on at most two unlabeled development rows with a
five-minute timeout. It validates IDs, feature shape, dtype and finiteness before
using Linux `renameat2(RENAME_EXCHANGE)` to replace the complete directory in one
operation. The previous complete package remains in the printed sibling path
for rollback. Failed checks or unsupported directory exchange leave the current
submission intact. Use a single publisher; the controller snapshots a complete
published directory before evaluation. Keep `/workspace/submission` inside the
workspace filesystem rather than making that directory itself a bind mount.
This portability check is not a model-quality evaluation.
