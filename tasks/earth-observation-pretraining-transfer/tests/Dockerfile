# Standalone verifier build; no locally named training image is required.
FROM python:3.12.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git bubblewrap libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.8.22
RUN git clone https://github.com/allenai/olmoearth_pretrain.git /opt/olmoearth && cd /opt/olmoearth && git checkout 0497dfbb6711ded4e6bf10cf089fc1e4d58c186b
WORKDIR /opt/olmoearth
RUN uv sync --locked --extra training --extra beaker --extra eval --no-editable --python 3.12
ENV PATH="/opt/olmoearth/.venv/bin:$PATH" WANDB_MODE=disabled HF_HUB_OFFLINE=1
ENV OLMO_INFERENCE_HASHSEED=0
WORKDIR /workspace
COPY tests /tests
COPY environment/evaluation /evaluation
COPY environment/baseline/starter /starter
RUN mkdir -p /data/eval/awf /data/eval/lfmc && touch /data/eval/awf/inputs.h5 /data/eval/lfmc/inputs.h5
# Harbor starts the verifier with its own command and then runs /tests/test.sh; an entrypoint that
# grades at start-up would exit before the submission is uploaded.
CMD ["sleep", "infinity"]
