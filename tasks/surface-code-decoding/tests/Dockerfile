FROM python:3.11.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.14
WORKDIR /opt/qec
COPY environment/baseline/alphaqubit/alphaqubit-runtime/pyproject.toml environment/baseline/alphaqubit/alphaqubit-runtime/uv.lock /opt/qec/
RUN uv sync --frozen --no-dev --python /usr/local/bin/python && useradd --uid 10001 --create-home agent
ENV PATH="/opt/qec/.venv/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH=/opt/task HF_HUB_OFFLINE=1 PIP_NO_INDEX=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY environment/baseline/alphaqubit/qecbench /opt/task/qecbench
COPY environment/baseline/alphaqubit/baselines /opt/task/baselines
COPY environment/baseline/alphaqubit/score_config.json /opt/task/score_config.json
COPY tests /opt/task/tests
# Harbor runs /tests/test.sh in a separate verifier image and does not upload tests/ there.
RUN chmod 755 /opt/task/tests/test.sh && ln -s /opt/task/tests /tests \
    && mkdir -p /logs/verifier /sealed /workspace/submission && chmod 700 /sealed
WORKDIR /opt/task
CMD ["sleep", "infinity"]
