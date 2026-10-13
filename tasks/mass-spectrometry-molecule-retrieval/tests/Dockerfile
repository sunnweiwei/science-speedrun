FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential libcairo2 libgomp1 bubblewrap && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.14
COPY environment/runtime /opt/massspecgym-runtime
RUN bash /opt/massspecgym-runtime/install.sh

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
    mkdir -p /logs/verifier /sealed && chmod 755 /logs && chmod 700 /sealed
CMD ["/tests/test.sh"]
