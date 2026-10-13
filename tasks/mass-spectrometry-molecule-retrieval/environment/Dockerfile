FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential libcairo2 libgomp1 bubblewrap && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.14
COPY environment/runtime /opt/massspecgym-runtime
RUN bash /opt/massspecgym-runtime/install.sh

# curl and huggingface_hub serve the operator-side data materialization, which
# runs in this image with network access before the agent budget starts.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      numpy==1.24.4 \
      scipy==1.14.1 \
      rdkit==2023.9.4 \
      myopic-mces==1.0.0 \
      pulp==2.9.0 \
      huggingface_hub==1.32.0 && \
    useradd --create-home --uid 10001 agent

ENV PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
# The agent's workspace (TASK_STANDARD.md); the datasets are mounted under /workspace/data.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
COPY environment/data/check_resources.py /workspace/data/check_resources.py
COPY environment/evaluation /workspace/evaluation
COPY environment/references /workspace/references
COPY environment/baseline /workspace/baseline
COPY environment/entrypoint.sh /usr/local/bin/task-entrypoint
RUN mkdir -p /workspace/submission /workspace/research && \
    chmod -R a+rX /workspace/baseline/starter /workspace/evaluation /workspace/references /workspace/data && \
    chmod 755 /usr/local/bin/task-entrypoint && \
    chmod 777 /workspace/submission /workspace/research

ENTRYPOINT ["/usr/local/bin/task-entrypoint"]
CMD ["sleep", "infinity"]
