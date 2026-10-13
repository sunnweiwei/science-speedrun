FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Verifier image: identical runtime to the development image (a submission's infer.sh runs
# here as uid 10001 with no network) plus the trusted grader.
ARG STATE_COMMIT=9bbfe78a434a55205e4de834e1ea99f85f7a3add
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    python -m pip install --no-cache-dir --disable-pip-version-check \
      anndata==0.12.19 numpy==2.4.6 scipy==1.17.1 h5py==3.14.0 pandas==2.3.3 \
      cell-eval==0.8.2 pdex==0.3.0 scanpy polars pyarrow pytest \
      "arc-state @ git+https://github.com/ArcInstitute/state.git@${STATE_COMMIT}" && \
    apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 agent
ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /tests
COPY . /tests
RUN chmod 755 /tests/test.sh
CMD ["/tests/test.sh"]
