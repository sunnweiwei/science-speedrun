FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      numcodecs==0.15.1 numpy==1.26.4 pandas==2.2.3 pyarrow==19.0.1 \
      pysam==0.23.3 zarr==2.18.7 pyyaml==6.0.2 pytest==8.3.5 \
      scipy==1.14.1 scikit-learn==1.5.2 && \
    useradd --create-home --uid 10001 agent && \
    mkdir -p /benchmark/reference /benchmark/public \
      /labels/public /labels/labels /labels/scoring/test \
      /labels/targets/test

ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/src

WORKDIR /tests
COPY tests /tests
COPY environment/data/src /src
COPY environment/evaluation /evaluation
RUN chmod 755 /tests/test.sh
CMD ["/tests/test.sh"]
