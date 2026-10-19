# Build context: fruit-fly-brain/. Linux x86_64 runtime; mount data/models separately.
FROM python:3.11.13-slim-bookworm@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 gcc libc6-dev && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib
COPY upstream/pathfinder/runtime/requirements.lock /opt/pathfinder/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /opt/pathfinder/requirements.lock
COPY upstream/common.py /opt/pathfinder/upstream/common.py
COPY upstream/pathfinder /opt/pathfinder/upstream/pathfinder
WORKDIR /workspace
CMD ["sleep", "infinity"]
