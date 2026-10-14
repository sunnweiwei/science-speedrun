#!/usr/bin/env bash
# Download or build a task's data, then write $ENVS_DIR/TASK.env (TASK_DATA_DIR, TASK_PRIVATE_DIR).
#   prepare.sh TASK DATA_ROOT
# Each branch runs the command documented in the task's environment/data/MATERIALIZE.md (or
# README). Private Hugging Face repositories need HF_TOKEN in the environment; it is passed to the
# download step only and never reaches an agent container.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/config.sh"
task=$1; mkdir -p "$2"; root=$(cd "$2" && pwd); t="$TASKS_DIR/$task"
[ -d "$t" ] || { echo "Unknown task $task" >&2; exit 1; }
image="${PREPARE_IMAGE:-speedrun-prepare:1}"
$DOCKER image inspect "$image" >/dev/null 2>&1 || $DOCKER build -q -f "$here/Dockerfile.prepare" -t "$image" "$here"
inbox() {  # inbox "pip pins" "command"  — run in the prepare image with the task and the data root
  $DOCKER run --rm ${HF_TOKEN:+-e HF_TOKEN} -v "$t:/task:ro" -v "$root:$root" -w /task "$image" bash -c \
    "uv venv -q /tmp/v && uv pip install -q --python /tmp/v/bin/python $1 && export PATH=/tmp/v/bin:\$PATH && $2"
}
case "$task" in
  genome-sequence-to-function)            # prepared release from HF, verified; docker-based
    bash "$t/environment/data/materialize.sh" release "$root"
    data=$root; private=$root ;;
  single-cell-perturbation-prediction)    # prepared release (dev/ + private/) and SE-600M
    inbox "huggingface_hub>=1.0" "VCZ_MATERIALIZE_PYTHON=/tmp/v/bin/python bash environment/data/fetch_release.sh $root"
    data=$root/dev; private=$root/private ;;
  mass-spectrometry-molecule-retrieval)   # prepared public view + private grader labels
    inbox "huggingface_hub==0.36.2" "bash environment/data/materialize.sh $root/public $root/labels"
    data=$root/public; private=$root/labels ;;
  pde-surrogate)                          # agent side: train subtree only; verifier: both subtrees
    inbox "huggingface_hub==0.36.2 h5py==3.16.0 numpy" \
      "python environment/data/materialize.py $root/agent --from-release --subtree train && python environment/data/materialize.py $root/verifier --from-release"
    data=$root/agent; private=$root/verifier ;;
  timeseries-forecasting)                 # GiftEval + pretraining corpus (275 GB), validation cases
    inbox "huggingface_hub numpy pandas pyarrow datasets" \
      "python environment/data/materialize.py $root && python environment/evaluation/make_cases.py --data $root --manifest environment/data/split.json --split valid --input $root/valid/input --truth $root/valid/truth"
    data=$root; private=$root ;;
  surface-code-decoding)                  # development + verifier roles; grader role needs HF_TOKEN
    inbox "huggingface-hub==0.35.3" \
      "python environment/data/materialize.py --output $root --roles development verifier && python environment/data/materialize.py --output $root --roles grader"
    data=$root; private=$root/grader ;;
  natural-disease-progression)            # credentialed HF release (MIMIC-IV terms), then re-split
    rev=f7da2d06295f1ed614de0ceb5619b3f72dbaedd7
    inbox "huggingface_hub>=1.0" "hf download ScienceRSI/natural-disease-progression --repo-type dataset --revision $rev --local-dir $root/release"
    $DOCKER build -q -f "$t/tests/Dockerfile" --target grader -t ndp-grader "$t"
    $DOCKER run --rm --network none -v "$root/release:/source:ro" -v "$t:/task:ro" -v "$root:/out" ndp-grader \
      python /task/environment/data/build_heldout_split.py --source /source --output /out/runtime
    data=$root/runtime/development; private=$root/runtime/private ;;
  protein-folding-prediction)             # FoldBench release (73 GB, pinned HF commit, every file checksummed)
    inbox "huggingface_hub[hf_xet]>=1.0" "bash environment/data/materialize.sh $root"
    data=$root; private=$root ;;
  tokamak-magnetic-control)               # no data: the machine description is vendored into the images
    data=$root; private=$root ;;
  *) echo "No prepare recipe for $task; follow its environment/data/MATERIALIZE.md" >&2; exit 1 ;;
esac
mkdir -p "$ENVS_DIR"
printf 'TASK_DATA_DIR=%s\nTASK_PRIVATE_DIR=%s\n' "$data" "$private" > "$ENVS_DIR/$task.env"
echo "[prepare] $task ready: $ENVS_DIR/$task.env"; cat "$ENVS_DIR/$task.env"
echo "[prepare] add the data root to MOUNT_ROOTS if it is outside EVAL_ROOT: $root"
