# Development and verifier images for the tokamak magnetic-control task.
#   docker build --target development -t tokamak-dev      -f environment/Dockerfile .
#   docker build --target verifier    -t tokamak-verifier -f environment/Dockerfile .
# Build context: the task directory (tokamak-magnetic-control/).
# Stage order matters: a build without --target produces the LAST stage, so the
# development image (what an agent gets) is last and the verifier, which carries
# tests/sealed, can only be produced by asking for it explicitly
# (tests/docker-compose.yaml does; environment/docker-compose.yaml selects
# development).

FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
# FreeGSNKE 3.1.0 (LGPL-3.0) provides the free-boundary evolutive solver.
RUN pip install --no-cache-dir "freegsnke==3.1.0" "gymnasium==1.2.*" \
    && pip freeze > /opt/requirements-base.lock.txt

# ---------------------------------------------------------------- verifier
FROM base AS verifier
RUN pip install --no-cache-dir "onnxruntime>=1.18" \
    && pip freeze > /opt/requirements-verifier.lock.txt
COPY environment/tokagym /opt/tokamak-task/tokagym
COPY environment/data /opt/tokamak-task/data
COPY environment/evaluation /opt/tokamak-task/evaluation
COPY tests /tests
ENV PYTHONPATH=/opt/tokamak-task TOKAGYM_DATA_DIR=/opt/tokamak-task/data/machine_configs
# One thread per process: each episode is a single-threaded worker with its own CPU, so replays are
# bitwise deterministic and never oversubscribe the verifier's CPUs.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN mkdir -p /logs/verifier /workspace/submission

# ---------------------------------------------------------------- development
FROM base AS development
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir "onnx>=1.16" "onnxruntime>=1.18" \
    && pip freeze > /opt/requirements-development.lock.txt
WORKDIR /workspace
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/tokagym /workspace/tokagym
COPY environment/data /workspace/data
COPY environment/evaluation /workspace/evaluation
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
ENV PYTHONPATH=/workspace TOKAGYM_DATA_DIR=/workspace/data/machine_configs
RUN mkdir -p /workspace/submission /workspace/research

