#!/usr/bin/env bash
# Build the task's native DGL for its pinned Python, torch ABI and CUDA architectures.
# The source checkout (including submodules) is pinned in both Dockerfiles.
set -euo pipefail
source_dir=${1:?DGL source directory required}
out=${2:?wheel output directory required}
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
export UV_PYTHON_INSTALL_DIR="$build_dir/python" UV_CACHE_DIR="$build_dir/cache"
uv python install 3.11.14
uv venv --python "$(uv python find --managed-python --no-project 3.11.14)" "$build_dir/venv"
export PATH="$build_dir/venv/bin:$PATH"
uv pip install --python "$build_dir/venv/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 cmake==3.31.6 setuptools==75.8.0 wheel==0.45.1 \
    Cython==0.29.37 numpy==1.26.4 scipy==1.17.1 networkx==3.6.1 requests==2.34.2 \
    tqdm==4.70.1 psutil==7.2.2 pyyaml==6.0.3 pydantic==2.14.3
# DGL's wheel setup finds the compiled libraries under the checkout's build directory.
cmake -S "$source_dir" -B "$source_dir/build" -DUSE_CUDA=ON -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="80 86 90" -DCUDA_ARCH_PTX=90 -DBUILD_TORCH=ON \
    -DTORCH_PYTHON_INTERPS="$build_dir/venv/bin/python" -DBUILD_TYPE=release
cmake --build "$source_dir/build" --parallel "${DGL_BUILD_JOBS:-12}"
mkdir -p "$out"
cd "$source_dir/python"
python setup.py bdist_wheel --dist-dir "$out"
test -s "$out/dgl-2.5-cp311-cp311-linux_x86_64.whl"
