FROM python:3.12.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git bubblewrap libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.22
RUN git clone https://github.com/allenai/olmoearth_pretrain.git /opt/olmoearth && cd /opt/olmoearth && git checkout 0497dfbb6711ded4e6bf10cf089fc1e4d58c186b
WORKDIR /opt/olmoearth
RUN uv sync --locked --extra training --extra beaker --extra eval --no-editable --python 3.12
ENV PATH="/opt/olmoearth/.venv/bin:$PATH" WANDB_MODE=disabled HF_HUB_OFFLINE=1 \
    PYTHONHASHSEED=0 OLMO_INFERENCE_HASHSEED=0 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
# Build context: the earth-observation-pretraining-transfer task directory, as declared in docker-compose.yaml.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md environment/data/MATERIALIZE.md environment/data/sources.json /workspace/data/
COPY environment/baseline /workspace/baseline
COPY docs/LICENSE-OlmoEarth /workspace/baseline/starter/LICENSE-OlmoEarth
COPY environment/evaluation /workspace/evaluation
COPY tests/replay.py tests/namespace.py /workspace/evaluation/
# Keep the existing raw-feature helper's sibling-starter import working.
RUN ln -s baseline/starter /workspace/starter && \
    mkdir -p /workspace/submission /workspace/research
# Prepared data and writable research storage are supplied by the operator.
CMD ["sleep", "infinity"]
