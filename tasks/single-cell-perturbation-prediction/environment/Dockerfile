FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Development image. arc-state (STATE: SE + ST), the official cell-eval scorer, and the
# single-cell stack, all pinned. Runtime network access is disabled by the task: nothing
# here may be fetched later, so everything the agent needs is installed or mounted.
ARG STATE_COMMIT=9bbfe78a434a55205e4de834e1ea99f85f7a3add
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    python -m pip install --no-cache-dir --disable-pip-version-check \
      anndata==0.12.19 numpy==2.4.6 scipy==1.17.1 h5py==3.14.0 pandas==2.3.3 \
      cell-eval==0.8.2 pdex==0.3.0 scanpy polars pyarrow pytest \
      "arc-state @ git+https://github.com/ArcInstitute/state.git@${STATE_COMMIT}" && \
    python -m pip freeze > /opt/requirements.lock && \
    apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 agent
ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
COPY environment/evaluation /workspace/evaluation
COPY environment/entrypoint.sh /usr/local/bin/vcz-entrypoint
# The release entries are mounted read-only under /workspace/data; the frozen SE-600M release is
# one of them and is also reachable at /opt/se600m, the path the contract gives packages.
RUN mkdir -p /workspace/submission /workspace/research /workspace/data && \
    ln -s /workspace/data/se600m /opt/se600m && \
    chmod 755 /usr/local/bin/vcz-entrypoint && \
    chown -R agent:agent /workspace
ENV SE600M_DIR=/opt/se600m
ENTRYPOINT ["/usr/local/bin/vcz-entrypoint"]
CMD ["sleep", "infinity"]
