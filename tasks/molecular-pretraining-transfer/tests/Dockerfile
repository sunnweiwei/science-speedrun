# Verifier images (TASK_STANDARD.md). Build context: tests/. "main" fine-tunes the frozen package (as the
# unprivileged user agent) and seals the predictions; "grader" scores them against the private labels and never
# runs package code.
# BEGIN SHARED RUNTIME: keep identical in environment/Dockerfile and tests/Dockerfile (tests/test_task.py checks it).
# Official Uni-Mol2 code: Uni-Mol (deepmodeling/Uni-Mol @ 90f52c4, unimol2/) on Uni-Core (dptech-corp/Uni-Core @ ace6fae).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel@sha256:a55ff10111eb11f998884327d37361592e632899edd24fce99886b69289e33e6 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST="8.0;9.0"
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "typing-extensions>=4.16" "numpy<2" rdkit==2024.3.5 lmdb==1.5.1 pandas==2.2.2 \
      scikit-learn==1.5.2 tensorboardX==2.6.2.2 tokenizers==0.20.3 ml_collections==0.1.1 numba==0.60.0
RUN git clone https://github.com/dptech-corp/Uni-Core /opt/Uni-Core && cd /opt/Uni-Core \
 && git checkout ace6fae1c8479a9751f2bb1e1d6e4047427bc134 \
 && python setup.py install --enable-cuda-ext && python -c "import unicore; print(\"unicore ok\")"
RUN git clone https://github.com/deepmodeling/Uni-Mol /opt/Uni-Mol && cd /opt/Uni-Mol \
 && git checkout 90f52c41299a1a582da0f9765e9f87aa21faa16a
# Uni-Core hard-codes a 90 s NCCL timeout, too short for one slow rank on a shared host; raise it to 30 min.
RUN U=$(python -c "import os, unicore; print(os.path.dirname(unicore.__file__))") \
 && sed -i "s/timeout=datetime.timedelta(seconds=90)/timeout=datetime.timedelta(seconds=1800)/" $U/distributed/utils.py \
 && grep -q "seconds=1800" $U/distributed/utils.py
# END SHARED RUNTIME

FROM runtime AS main
RUN useradd --create-home --uid 10001 agent
COPY . /tests
RUN chmod 755 /tests/test.sh && chmod -R a+rX /tests
WORKDIR /tests
CMD ["/tests/test.sh"]

FROM runtime AS grader
COPY . /tests
WORKDIR /tests
CMD ["sleep", "infinity"]
