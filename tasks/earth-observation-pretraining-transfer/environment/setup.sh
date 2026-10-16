#!/bin/bash
# CPU-only, pinned official training/evaluation runtime.
set -euo pipefail
: "${1:?Usage: setup.sh RUNTIME_DIRECTORY}"
OLMO_RUNTIME=$(realpath -m "$1")
OLMO_REV=0497dfbb6711ded4e6bf10cf089fc1e4d58c186b
export CUDA_VISIBLE_DEVICES=''
command -v uv >/dev/null || { echo 'Install uv 0.8.22 first' >&2; exit 2; }
mkdir -p "$OLMO_RUNTIME"
if [[ ! -d "$OLMO_RUNTIME/olmoearth-upstream/.git" ]]; then
  git clone https://github.com/allenai/olmoearth_pretrain.git "$OLMO_RUNTIME/olmoearth-upstream"
  git -C "$OLMO_RUNTIME/olmoearth-upstream" checkout "$OLMO_REV"
fi
[[ $(git -C "$OLMO_RUNTIME/olmoearth-upstream" rev-parse HEAD) == "$OLMO_REV" ]] || { echo 'Wrong upstream revision' >&2; exit 2; }
cd "$OLMO_RUNTIME/olmoearth-upstream"
export UV_CACHE_DIR="$OLMO_RUNTIME/uv-cache" UV_LINK_MODE=copy
uv sync --locked --extra training --extra beaker --extra eval --python 3.12.11 --no-editable
uv pip freeze --python .venv/bin/python > "$OLMO_RUNTIME/requirements-observed.txt"
printf '%s\n' "$OLMO_REV" > "$OLMO_RUNTIME/upstream-revision.txt"
