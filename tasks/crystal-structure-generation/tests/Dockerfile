FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 AS generator-runtime

ARG UV_VERSION=0.8.15
ARG OMG_REVISION=ed80cb86fd69d4f4d14164c6635b0a8d4939b34f
ARG LEMAT_REVISION=58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl g++ gcc git libc6-dev make && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY runtime/omatg-generator-runtime.patch /tmp/omatg-generator-runtime.patch
RUN git init /opt/omg && \
    git -C /opt/omg remote add origin https://github.com/FERMat-ML/OMatG.git && \
    git -C /opt/omg fetch --depth 1 origin "${OMG_REVISION}" && \
    git -C /opt/omg checkout --detach FETCH_HEAD && \
    git -C /opt/omg apply --check /tmp/omatg-generator-runtime.patch && \
    git -C /opt/omg apply /tmp/omatg-generator-runtime.patch && \
    python -m pip install --no-cache-dir \
      safetensors==0.6.2 \
      torch-scatter==2.1.2 \
      -f https://data.pyg.org/whl/torch-2.8.0+cu128.html && \
    python -m pip install --no-cache-dir -e /opt/omg && \
    useradd --create-home --uid 10001 agent

# END SHARED GENERATOR RUNTIME: keep identical in development and final replay.

FROM generator-runtime AS scorer-runtime
COPY runtime/lemat-eight-gpu.patch /tmp/lemat-eight-gpu.patch
COPY runtime/lemat-small-batch.patch /tmp/lemat-small-batch.patch
COPY runtime/lemat-substitution-novelty.patch /tmp/lemat-substitution-novelty.patch
RUN git init /opt/lemat && \
    git -C /opt/lemat remote add origin https://github.com/LeMaterial/lemat-genbench.git && \
    git -C /opt/lemat fetch --depth 1 origin "${LEMAT_REVISION}" && \
    git -C /opt/lemat checkout --detach FETCH_HEAD && \
    git -C /opt/lemat apply --check /tmp/lemat-eight-gpu.patch && \
    git -C /opt/lemat apply /tmp/lemat-eight-gpu.patch && \
    git -C /opt/lemat apply --check /tmp/lemat-small-batch.patch && \
    git -C /opt/lemat apply /tmp/lemat-small-batch.patch && \
    git -C /opt/lemat apply --check /tmp/lemat-substitution-novelty.patch && \
    git -C /opt/lemat apply /tmp/lemat-substitution-novelty.patch && \
    uv sync --project /opt/lemat --frozen --no-dev && \
    mkdir -p /opt/lemat/results_final && \
    chmod 1777 /opt/lemat/results_final && \
    rm -rf /opt/lemat/data/.cache && \
    ln -s /scorer-assets/huggingface/hub /opt/lemat/data/.cache && \
    rm -rf /root/.cache && \
    ln -s /scorer-assets/home/.cache /root/.cache

ENV PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CRYSTALGEN_LEMAT_GPU_WORKERS=8 \
    PYTHONPATH=/tests

FROM scorer-runtime AS grader
WORKDIR /tests
COPY . /tests
CMD ["sleep", "infinity"]

FROM generator-runtime AS main
ENV PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/tests

WORKDIR /tests
COPY . /tests
RUN chmod 755 /tests/test.sh
CMD ["/tests/test.sh"]
