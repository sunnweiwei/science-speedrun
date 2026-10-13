FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install \
      numcodecs==0.15.1 numpy==1.26.4 pandas==2.2.3 pyarrow==19.0.1 \
      pysam==0.23.3 zarr==2.18.7 pyyaml==6.0.2 pytest==8.3.5 \
      scipy==1.14.1 scikit-learn==1.5.2

WORKDIR /workspace
COPY contract.md /workspace/contract.md
COPY data/README.md /workspace/data/README.md
COPY baseline /workspace/baseline
COPY references /workspace/references
COPY evaluation /workspace/evaluation
RUN mkdir -p /workspace/submission /workspace/research \
      /workspace/data/reference /workspace/data/public /workspace/data/metadata \
      /workspace/data/labels /workspace/data/scoring /workspace/data/targets \
      /workspace/data/variant_evals \
    && chmod 755 /workspace/baseline/*/*.sh \
      /workspace/baseline/alphagenome-reference/reference/alphagenome_50m/*.sh

CMD ["sleep", "infinity"]
