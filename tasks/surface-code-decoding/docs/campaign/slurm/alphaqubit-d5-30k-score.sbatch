#!/usr/bin/env bash
#SBATCH --job-name=alphaqubit-d5-30k-score
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${QEC_TASK_ROOT:?}"
: "${QEC_ASSET_ROOT:?}"
export PYTHONPATH="${QEC_TASK_ROOT}" JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
qec_python="${QEC_TASK_ROOT}/alphaqubit-runtime/.venv/bin/python"
qec_followup="${QEC_ASSET_ROOT}/alphaqubit-d5-30k"
# Unchanged d3 predictions were copied and hashed before submission.
"${qec_python}" -m baselines.score_neural \
  --predictions "${qec_followup}/predictions/finetune" \
  --data-root "${QEC_ASSET_ROOT}/data/sycamore" \
  --classical-root "${QEC_ASSET_ROOT}/classical-v1" \
  --output "${qec_followup}/scores-finetune.json" \
  --bootstrap-samples 2000 --seed 20260919
"${qec_python}" - <<'PY'
import json
import os
from pathlib import Path
from baselines.alphaqubit_artifact import atomic_json, sha256_file

root = Path(os.environ['QEC_ASSET_ROOT'])
followup = root / 'alphaqubit-d5-30k'
decision = json.loads((followup / 'decision.json').read_text())
original = root / 'alphaqubit-v1/scores-finetune.json'
if sha256_file(original) != decision['parent_score_sha256']:
    raise ValueError('The frozen v1 score changed')
for relative, digest in decision['retained_d3_prediction_sha256'].items():
    if sha256_file(followup / relative) != digest:
        raise ValueError(f'Unchanged d3 prediction was modified: {relative}')
before = json.loads(original.read_text())
after = json.loads((followup / 'scores-finetune.json').read_text())
# Shared bootstrap RNG draws can change after the d5 inputs change. Check the
# deterministic results, while retaining each run's independently sampled CI.
before_d3 = {k: v for k, v in before['summary']['macro']['d3'].items() if k != 'ci95'}
after_d3 = {k: v for k, v in after['summary']['macro']['d3'].items() if k != 'ci95'}
before_counts = {k: v for k, v in before['summary']['per_condition'].items() if v['distance'] == 3}
after_counts = {k: v for k, v in after['summary']['per_condition'].items() if v['distance'] == 3}
if before_d3 != after_d3 or before_counts != after_counts:
    raise ValueError('The follow-up unexpectedly changed d3 results')
atomic_json(followup / 'comparison.json', {
    'interpretation': 'Post-test follow-up on the same public test fold; not a fresh independent confirmation.',
    'checkpoint_selection': 'Even validation only; unchanged optimizer, LR, EMA and selection rule.',
    'd3_verification': 'Prediction hashes, condition counts and deterministic macro results unchanged; bootstrap intervals are recomputed.',
    'before': before['summary']['macro'], 'after': after['summary']['macro'],
    'score_sha256': sha256_file(followup / 'scores-finetune.json'),
})
PY
