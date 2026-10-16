FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

# Model runtime (/opt/venv: FAIRChem 2.16, phonopy 3.5.1, phono3py 3.30.0, torch-dftd) and the
# historical MP2020 scoring runtime (/opt/scoring-venv: Python 3.11, pymatgen 2023.5.10).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv==0.9.14
WORKDIR /opt/am
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra uma --extra data --no-install-project && rm -rf /root/.cache
COPY scoring/pyproject.toml scoring/uv.lock /opt/am-scoring/
RUN cd /opt/am-scoring && UV_PROJECT_ENVIRONMENT=/opt/scoring-venv uv sync --frozen --no-install-project && \
    rm -rf /root/.cache && chmod -R a+rX /opt/uv-python /opt/scoring-venv /opt/venv

WORKDIR /workspace
COPY contract.md /workspace/contract.md
COPY data/README.md /workspace/data/README.md
COPY baseline /workspace/baseline
# Historical reference recipes remain inspectable; published fitted offsets are not training inputs.
RUN rm -f /workspace/baseline/uma-mixed-replay/element-references.yaml
COPY references /workspace/references
COPY evaluation /workspace/evaluation
RUN mkdir -p /workspace/submission /workspace/research /workspace/data/pool /workspace/data/val \
      /workspace/data/models /workspace/data/baselines

CMD ["sleep", "infinity"]
