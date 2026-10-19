# Build context: fruit-fly-neuron-reconstruction/. Data and models stay outside the image.
FROM python:3.11.13-slim-bookworm@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 gcc libc6-dev git time bubblewrap && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COPY upstream/pathfinder/runtime/requirements.lock /opt/pathfinder/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /opt/pathfinder/requirements.lock
COPY upstream/common.py /opt/pathfinder/upstream/common.py
COPY upstream/pathfinder /opt/pathfinder/upstream/pathfinder
RUN rm -rf /opt/pathfinder/upstream/pathfinder/receipts
RUN mkdir -p /workspace/submission /workspace/research /logs/verifier
RUN git config --system --add safe.directory /runtime/ffn \
    && git config --system --add safe.directory /runtime/connectomics \
    && git config --system --add safe.directory /workspace/data/runtime/ffn \
    && git config --system --add safe.directory /workspace/data/runtime/connectomics
WORKDIR /workspace
RUN printf '{"protocol":"single-checkpoint-fast/v1"}\n' > /opt/pathfinder/container-verifier.json
RUN chmod 444 /opt/pathfinder/container-verifier.json


FROM runtime AS verifier
COPY tests/test.sh /tests/test.sh
COPY tests/limits.json /opt/pathfinder/limits.json
CMD ["sleep", "infinity"]

FROM runtime AS development
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
COPY environment/data/README.md /workspace/data/README.md
COPY environment/evaluation /workspace/evaluation
COPY environment/contract.md /workspace/contract.md
COPY instruction.md /workspace/instruction.md
# Editable copy of the reference implementation next to the baseline commands; run.sh uses the
# identical trusted copy under /opt/pathfinder unless PATHFINDER_CODE points here.
COPY upstream/common.py /workspace/baseline/pathfinder-fast/common.py
COPY upstream/pathfinder /workspace/baseline/pathfinder-fast/pathfinder
# Experiment receipts and test reports are maintainer records, never agent inputs.
RUN rm -rf /workspace/baseline/pathfinder-fast/pathfinder/receipts /opt/pathfinder/upstream/pathfinder/receipts \
    /workspace/baseline/pathfinder-fast/pathfinder/figures /opt/pathfinder/upstream/pathfinder/figures \
    && find /workspace/baseline/pathfinder-fast/pathfinder /opt/pathfinder/upstream/pathfinder -maxdepth 1 \( -name "*.md" -o -name "test_*.py" -o -name "malecns-*.json" -o -name "*.sbatch" -o -name "verify_single.py" \) -delete
CMD ["sleep", "infinity"]
