#!/usr/bin/env bash
# package.sh CHECKPOINT OUT_DIR
# Assemble a self-contained submission from a trained reference checkpoint: this baseline's model
# code, predict/perturb entry points and the weights as last.pt. Unit tests are left out.
set -euo pipefail
CKPT=$1; OUT=$2; HERE=$(cd "$(dirname "$0")" && pwd)
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT/reference"
cp "$HERE"/{predict.py,perturb.py,prediction_contract.py,infer.sh,perturb.sh} "$OUT/"
cp -r "$HERE/reference/alphagenome_50m" "$OUT/reference/"
cp "$HERE/reference/__init__.py" "$OUT/reference/" 2>/dev/null || true
rm -rf "$OUT/reference/alphagenome_50m/tests" "$OUT/reference/alphagenome_50m/__pycache__"
cp "$CKPT" "$OUT/last.pt"
chmod 755 "$OUT/infer.sh" "$OUT/perturb.sh"
params=$(python - "$OUT/last.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
print(sum(v.numel() for v in state.values()))
PY
)
cat > "$OUT/README.md" <<README
# alphagenome-reference (task baseline)

The task's reference model (\`reference/alphagenome_50m\`, ${params} parameters in \`last.pt\`),
trained on the train split with \`run_ddp.sh\`. \`infer.sh\` writes contract-v4 track predictions and
\`perturb.sh\` the A/B perturbation outputs. Source checkpoint: \`$(basename "$CKPT")\`
(sha256 $(sha256sum "$CKPT" | cut -c1-16)…). This is the baseline to beat.
README
echo "packaged $CKPT -> $OUT ($params parameters)"
