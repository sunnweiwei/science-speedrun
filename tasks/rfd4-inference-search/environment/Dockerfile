FROM python:3.12-slim-bookworm AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.9.25 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates libxrender1 libxext6 zlib1g-dev && rm -rf /var/lib/apt/lists/*
# AF3 runs in its own locked environment. Single-sequence prediction skips the search databases.
RUN git init /opt/af3 && git -C /opt/af3 fetch -q --depth=1 https://github.com/google-deepmind/alphafold3.git f6a5aecf6eea2b5de4f846df86921630c5036b0b && git -C /opt/af3 checkout -q --detach FETCH_HEAD \
    && cd /opt/af3 && UV_PROJECT_ENVIRONMENT=/opt/af3env CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --no-cache --frozen --no-dev --no-editable && /opt/af3env/bin/build_data
COPY environment/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache --torch-backend cu126 -r /tmp/requirements.txt
ENV HF_HUB_OFFLINE=1 WANDB_MODE=disabled MPLBACKEND=Agg XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
WORKDIR /workspace
RUN mkdir -p /workspace/submission /workspace/research
COPY environment/evaluation /workspace/evaluation
RUN --mount=type=bind,source=environment/references,target=/published,ro \
    for f in s15/summary.csv s15/base_schedule_summary.csv monomer_beam/scaling_150slot_summary.csv monomer_beam/matched_30slot_summary.csv; do \
      if [ -f "/published/$f" ]; then install -D -m 644 "/published/$f" "/workspace/references/$f"; fi; \
    done

FROM runtime AS verifier
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap && rm -rf /var/lib/apt/lists/*
COPY environment/baseline /workspace/baseline
COPY tests/test.sh tests/rfd4_private.py /tests/
COPY environment/references/overall /opt/rfd4-overall

FROM runtime AS development
COPY instruction.md environment/contract.md /workspace/
COPY environment/data/README.md /workspace/data/README.md
COPY environment/baseline /workspace/baseline
COPY environment/references/agent.md /workspace/references/README.md
COPY environment/references/upstream /workspace/references/upstream
COPY environment/references/overall/operating_points.csv environment/references/overall/comparison_field.csv environment/references/overall/quality-throughput*.png /workspace/references/overall/
