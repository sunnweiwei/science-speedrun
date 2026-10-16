FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

# Same two runtimes as the agent image, so validation runs and the verifier execute identical code.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv==0.9.14
WORKDIR /opt/am
COPY environment/pyproject.toml environment/uv.lock ./
RUN uv sync --frozen --extra uma --extra data --no-install-project && rm -rf /root/.cache
COPY environment/scoring/pyproject.toml environment/scoring/uv.lock /opt/am-scoring/
RUN cd /opt/am-scoring && UV_PROJECT_ENVIRONMENT=/opt/scoring-venv uv sync --frozen --no-install-project && \
    rm -rf /root/.cache && chmod -R a+rX /opt/uv-python /opt/scoring-venv /opt/venv && \
    useradd --create-home --uid 10001 agent && mkdir -p /benchmark /labels

ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1
WORKDIR /tests
COPY tests /tests
COPY environment/evaluation /evaluation
RUN chmod 755 /tests/test.sh && chmod -R a+rX /evaluation
CMD ["/tests/test.sh"]
