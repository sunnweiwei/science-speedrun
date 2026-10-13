#!/usr/bin/env bash
# package.sh CHECKPOINT ADAPTER OUT_DIR
# Assemble a self-contained submission from a trained (calibrated) reference GRU: the model code,
# weights, the train-fit input adapter and an infer.sh that rebuilds the input cache offline.
set -euo pipefail
CKPT=$1; ADAPTER=$2; OUT=$3
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$OUT"
cp "$HERE/longitudinal.py" "$OUT/longitudinal.py"
cp "$CKPT" "$OUT/model.pt"
cp "$ADAPTER" "$OUT/train.adapter.json"
cat > "$OUT/infer.sh" <<'INFER'
#!/usr/bin/env bash
# infer.sh INPUT_DIR OUTPUT_DIR
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd); IN=$1; OUT=$2
CACHE=$(mktemp -d)
DEVICE=$(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
python3 "$HERE/longitudinal.py" cache --input "$IN" --adapter "$HERE/train.adapter.json" --cache "$CACHE/input.pt"
python3 "$HERE/longitudinal.py" predict --cache "$CACHE/input.pt" --checkpoint "$HERE/model.pt" \
  --device "$DEVICE" --output "$OUT/predictions.parquet"
rm -rf "$CACHE"
INFER
chmod 755 "$OUT/infer.sh"
cat > "$OUT/README.md" <<'README'
# temporal-gru (unchanged task baseline)

The task's reference temporal GRU (`baseline/temporal-gru/longitudinal.py --kind temporal`),
trained on `data/train` and calibrated on a train-internal fold. `infer.sh` rebuilds the input
cache with the packaged train-fit adapter and writes keyed predictions. This is the baseline to
beat, not an architecture-development result.
README
echo "packaged $CKPT -> $OUT"
