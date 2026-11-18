# Both the agent and verifier use the same DGL source and wheel build.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5 AS dgl-wheel
COPY --from=ghcr.io/astral-sh/uv:0.9.14@sha256:fef8e5fb8809f4b57069e919ffcd1529c92b432a2c8d8ad1768087b0b018d840 /uv /usr/local/bin/uv
ADD https://github.com/dmlc/dgl.git#743e65fa3ba9781bef443bfdbca7d87983e1071f /build/dgl
COPY environment/runtime/build-dgl.sh /build/build-dgl.sh
RUN bash /build/build-dgl.sh /build/dgl /wheels

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential libcairo2 libgomp1 bubblewrap && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.14
COPY environment/runtime /opt/massspecgym-runtime
RUN --mount=type=bind,from=dgl-wheel,source=/wheels,target=/wheels bash /opt/massspecgym-runtime/install.sh

RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      numpy==1.24.4 \
      scipy==1.14.1 \
      rdkit==2023.9.4 \
      myopic-mces==1.0.0 \
      pulp==2.9.0 && \
    useradd --create-home --uid 10001 agent

ENV PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /tests
COPY tests /tests
COPY environment/evaluation/benchmark_score.py environment/evaluation/score_bounds.json /tests/
RUN chmod 755 /tests/test.sh && \
    mkdir -p /logs/verifier /sealed /verifier-input && chmod 755 /logs && chmod 700 /sealed /verifier-input
CMD ["/tests/test.sh"]
