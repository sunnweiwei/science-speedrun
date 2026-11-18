# Rebuild point-ops/pathfinder_fps.so for sm_86 and sm_90 (H100) in the task's own runtime: Python 3.11.13 and
# the hash-locked torch 2.6.0 cu124 wheels, with the CUDA 12.4 toolkit. Compilation only; no GPU is used.
# Build context: requirements.lock and build_point_ops.py from this directory's parent folders, and src/ holding
# the four files of runtime/vendor/pointnext/openpoints/cpp/pointnet2_batch/src from the prepared release.
#   docker build -f fps-build.Dockerfile -t fps-build CONTEXT
#   docker create --name fps fps-build && docker cp fps:/build/point-ops/pathfinder_fps.so . && docker rm fps
# nvcc output is not bit-for-bit reproducible; receipts/fps-sm86-sm90-rebuild.json records the released binary.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda
FROM python:3.11.13-slim-bookworm@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47
RUN apt-get update && apt-get install -y --no-install-recommends g++ libc6-dev ninja-build && rm -rf /var/lib/apt/lists/*
COPY --from=cuda /usr/local/cuda-12.4 /usr/local/cuda-12.4
ENV CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$PATH
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock
COPY build_point_ops.py /build/build_point_ops.py
COPY src /build/vendor/pointnext/openpoints/cpp/pointnet2_batch/src
WORKDIR /build
RUN CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST="8.6;9.0" MAX_JOBS=4 python build_point_ops.py --root /build > /build/build.log 2>&1 \
    || (cat /build/build.log; exit 1)
