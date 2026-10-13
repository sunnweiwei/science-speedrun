#!/usr/bin/env bash
# Build-time only. Identical native-model runtime in development and verification.
set -euo pipefail
runtime_root=${MASSSPECGYM_RUNTIME_ROOT:-/opt}
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root="$runtime_root/ms-pred"
glacier_python="$runtime_root/glacier/bin/python"
source_commit=ed8311f22958cb37f055b663b5f56c5c77a2ee33
mkdir -p "$runtime_root"
if [[ ! -f "$source_root/SOURCE_COMMIT" ]]; then
  source_checkout=$(mktemp -d)
  trap 'rm -rf "$source_checkout"' EXIT
  git clone --filter=blob:none --no-checkout https://github.com/coleygroup/ms-pred.git "$source_checkout/ms-pred"
  mkdir -p "$source_root"
  git -C "$source_checkout/ms-pred" archive "$source_commit" src pyproject.toml README.md LICENSE \
    | tar -x -C "$source_root"
  printf '%s\n' "$source_commit" > "$source_root/SOURCE_COMMIT"
elif [[ $(cat "$source_root/SOURCE_COMMIT") != "$source_commit" ]]; then
  echo 'Unexpected GLACIER source revision' >&2
  exit 1
fi
export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
uv python install --install-dir "$runtime_root/python" 3.11.14
managed_python=$(uv python find --managed-python --no-project 3.11.14)
if [[ ! -x "$glacier_python" ]]; then
  uv venv --python "$managed_python" "$runtime_root/glacier"
fi
"$glacier_python" -c 'import pathlib, sys; root=pathlib.Path(sys.argv[1]).resolve(); assert pathlib.Path(sys.base_prefix).resolve().is_relative_to(root), "Existing runtime uses an external Python; choose a fresh runtime root"' "$runtime_root"
uv pip install --python "$glacier_python" --index-strategy unsafe-best-match \
  -r "$script_root/glacier-requirements.txt"
uv pip install --python "$glacier_python" --no-deps --no-build-isolation -e "$source_root"
"$glacier_python" -c 'import torch, dgl, torch_scatter, pytorch_lightning; from ms_pred.glacier.joint_model import JointModel; from ms_pred.magma.run_magma import magma_augmentation; print("native GLACIER runtime imports passed")'
chmod -R a+rX "$runtime_root/glacier" "$runtime_root/python" "$source_root"
