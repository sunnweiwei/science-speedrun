FROM python:3.11.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates libgomp1 git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.9.14
WORKDIR /opt/qec
COPY environment/baseline/alphaqubit/alphaqubit-runtime/pyproject.toml environment/baseline/alphaqubit/alphaqubit-runtime/uv.lock /opt/qec/
RUN uv sync --frozen --no-dev --python /usr/local/bin/python
ENV PATH="/opt/qec/.venv/bin:/usr/local/bin:/usr/bin:/bin" PYTHONPATH=/workspace/baseline/alphaqubit HF_HUB_OFFLINE=1 PIP_NO_INDEX=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# The agent's workspace (TASK_STANDARD.md). The verified baseline is the task's Python project
# (environment/baseline/alphaqubit, whose qecbench/, baselines/ and score_config.json the verifier image shares).
COPY environment/baseline/alphaqubit /workspace/baseline/alphaqubit
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
COPY environment/evaluation /workspace/evaluation
COPY environment/references /workspace/references
COPY environment/data/check_resources.py /opt/qec/check_resources.py
COPY environment/entrypoint.sh /opt/qec/entrypoint.sh
RUN chmod 755 /opt/qec/entrypoint.sh && mkdir -p /workspace/submission /workspace/research
WORKDIR /workspace
ENTRYPOINT ["/opt/qec/entrypoint.sh"]
CMD ["sleep", "infinity"]
