# Single-checkpoint reference implementation

The task uses one FFN, one SENSE and one SHAPE checkpoint. No trained weights or
precomputed predictions are supplied to the agent. The verifier always starts
from the raw image. [RESULTS.md](RESULTS.md) records the selected starter’s
measured development results and the formal entry’s validation scope.

The wrapper exposes `train-volume`, `train-shape`, `export`, `publish`, and `predict`.
Arguments are explicit; `run.sh <command> --help` lists them. For example:

```bash
set -euo pipefail
BASE=/workspace/baseline/pathfinder-fast/run.sh
TRAIN_SECONDS=${TRAIN_SECONDS:-10800}
# Phase 1: historical legacy training, random initialization.
"${PYTHON:-python3}" -m torch.distributed.run --standalone \
  --local-addr=127.0.0.1 --nproc-per-node=4 --no-python \
  bash "$BASE" train-volume --stage ffn --ffn-policy legacy --batch 16 \
  --seed 42 --workers 0 --lr 0.0001 --eval-every 250 --steps 6750 \
  --ffn-recurrent-steps 4 --ffn-head-init-gain 1 --ffn-positive-weight 1 \
  --manifest /workspace/data/training/volume/manifest.json \
  --output /workspace/research/ffn-legacy --seconds "$TRAIN_SECONDS"
# Require the selected step; do not substitute an earlier time-limited model.
"${PYTHON:-python3}" - <<'CHECK'
import torch
checkpoint = torch.load("/workspace/research/ffn-legacy/last.pt",
                        map_location="cpu", weights_only=False)
assert checkpoint["step"] == 6750, "Legacy training did not reach step 6750"
CHECK
# Phase 2: one production-rule update, with a fresh optimizer.
bash "$BASE" growth-reference \
  --manifest /workspace/data/training/volume/manifest.json \
  --checkpoint /workspace/research/ffn-legacy/last.pt \
  --output /workspace/research/ffn-reference.json \
  --device cuda --samples 16 --shape 129 --max-calls 4096
"${PYTHON:-python3}" -m torch.distributed.run --standalone \
  --local-addr=127.0.0.1 --nproc-per-node=4 --no-python \
  bash "$BASE" train-volume --stage ffn --ffn-policy production --batch 16 \
  --seed 42 --workers 0 --lr 0.00003 --eval-every 500 --steps 1 \
  --ffn-recurrent-steps 8 --ffn-positive-weight 2 \
  --init-checkpoint /workspace/research/ffn-legacy/last.pt \
  --ffn-validation-reference /workspace/research/ffn-reference.json \
  --manifest /workspace/data/training/volume/manifest.json \
  --output /workspace/research/ffn-corrected --seconds "$TRAIN_SECONDS"
FFN_CHECKPOINT=/workspace/research/ffn-corrected/last.pt
# Validate this explicit candidate before the export/publish commands below.
bash "$BASE" train-volume --stage sense \
  --manifest /workspace/data/training/volume/manifest.json \
  --output /workspace/research/sense --seconds "$TRAIN_SECONDS"
bash "$BASE" train-shape --root /workspace/data/runtime \
  --manifest /workspace/data/training/shape/manifest.json \
  --output /workspace/research/shape --seconds "$TRAIN_SECONDS"
bash "$BASE" export --ffn "$FFN_CHECKPOINT" --sense "$SENSE_CHECKPOINT" \
  --shape "$SHAPE_CHECKPOINT" --output /workspace/submission-next
bash "$BASE" publish --staged /workspace/submission-next
```

The selected workflow is the best measured historical corrected FFN: legacy
training through step 6750, followed by one production-rule optimizer update.
Both phases use four GPUs and 16 crops per GPU (total batch 64), seed 42,
raw/255 input without additional normalization, and AdamW with weight decay
0.01. Phase 1 uses four recurrent steps, positive weight 1 and learning rate
1e-4. Phase 2 starts a fresh optimizer at 3e-5, uses eight recurrent steps and
positive weight 2, and retains corrected movement, seed revalidation and
disconnectedness rules. Inference continues to use the corrected production
rules. Use fresh output directories. `TRAIN_SECONDS=10800` is a per-stage time
cap, not a guarantee that phase 1 reaches its requested step on every device.

The measured selected checkpoint SHA256 is
`ffdaeac9ddcd7eb27ece07d19135b9dfd8186fa50b7493c2327bcab5a2257bf7`;
its legacy parent SHA256 is
`38d90421dce62b1e35d57d730800a872a58efb44c41b80877061e7af825cc53e`.
The selected checkpoint is phase-2 step 1, not the legacy parent itself and not
the final checkpoint of the subsequent repair-training run. Its complete local
reconstruction proxy nERL is 0.07227, versus 0.03500 for the corrected-from-start
16300-step candidate. Fixed single-window results are 0.07079 versus 0.04064;
paired eight-window results are 0.03502 versus 0.01905. These are different
validation scopes; do not compare numbers across scopes. The corrected-from-start
batch-16/weight-4 recipe and 6000-step ablations remain experiments, not the
selected starter. See RESULTS.md for measured evaluation limits.

The commands reproduce the two-stage recipe with the current trainer; they do
not promise byte-identical historical weights. Hardware, source revisions and
numerical precision can change the result. No trained weights are bundled.
`last.pt` is used here only to address the explicitly requested endpoint, not as
an automatic acceptance rule. Check its recorded step, complete fixed nonempty
seed growth and fresh reconstruction validation before export. A new run must
be evaluated against the selected reference before replacing it. Do not extend
phase 2 and assume that its final or lowest-loss model is better.

Training checkpoints are written atomically and include optimizer state,
per-rank random-generator state and the next global sample index.
`--init-checkpoint` deliberately loads model weights only for phase 2.
Production training normalizes recurrent loss by the configured step count;
production validation uses complete fixed seed growth. BCE alone does not
select the production model. The seed-growth reference uses only visible
validation data and is generated from phase 1 of this run.

Select checkpoints only with visible validation data. Export strips optimizer
state, verifies the full architecture and writes the package atomically to a
fresh destination. The publish command atomically exchanges the staged directory with
`/workspace/submission`, retaining the old package at the staging path. Both
paths must be on the same filesystem with Linux atomic-exchange support; an
unsupported filesystem fails without deleting either package.
The fixed search threshold is 0.8; no extra ensemble options are used.

These commands retain the selected legacy-to-production training workflow. The
measured FFN has complete fixed single/eight-window validation. A fresh replay
of this recipe and the complete training-to-formal-evaluation container route
remain acceptance gates.

A fresh development prediction uses the visible validation image:

```bash
mkdir /workspace/research/run
cp -r /workspace/data/validation/image /workspace/research/run/image
cp -r /workspace/submission /workspace/research/run/model
cp -r /workspace/data/runtime/point-ops /workspace/research/run/point-ops
bash "$BASE" predict --root /workspace/research/run \
  --runtime /workspace/data/runtime \
  --limits /workspace/baseline/pathfinder-fast/limits.json
```

This command uses every exposed GPU and needs complete inputs; see
`/workspace/evaluation/README.md` for its runtime. It creates a fresh
FFN prediction and all downstream outputs under `run/prediction`; it accepts no
prediction/context shortcut. Use the visible suite with `evaluate_physical.py`
for development scoring. The same limits and algorithms feed the formal
verifier, which independently stages the image/weights and isolates prediction.
