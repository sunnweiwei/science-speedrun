#!/usr/bin/env bash
# B2 inference: `state tx infer` on virtual copies of the basal cells (one per requested cell,
# labelled with the target) -> count-scale predictions on the input panel.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
in=$1; out=$2
work=$(mktemp -d "${TMPDIR:-/tmp}/b2-XXXXXX")
python "$here/predict.py" pad --input "$in" --work "$work"
# arc-state checkpoints contain numpy scalars; torch>=2.6 defaults torch.load to weights_only=True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m state tx infer --model-dir "$here/runs/zeroshot" --checkpoint "$here/runs/zeroshot/checkpoints/final.ckpt" \
  --adata "$work/padded.h5ad" --pert-col gene --control-pert non-targeting \
  --output "$work/pred.h5ad" --quiet
python "$here/predict.py" finish --input "$in" --work "$work" --output "$out"
rm -rf "$work"
