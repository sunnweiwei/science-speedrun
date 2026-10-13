FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361 AS base

RUN python -m pip install --no-cache-dir duckdb==1.4.4 pyarrow==23.0.1 numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 pytest==8.3.5 lightgbm==4.6.0 joblib==1.5.3

# Thread pools follow task.toml's cpus = 10 (see environment/Dockerfile).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    OMP_NUM_THREADS=10

RUN useradd --create-home --uid 10001 agent

FROM base AS main
WORKDIR /tests
COPY tests/common.py tests/main_verifier.py tests/test.sh /tests/
COPY environment/data/label_rules.py /data/label_rules.py
RUN python -c "import joblib, lightgbm, torch"
RUN chmod 755 /tests/test.sh && mkdir -p /logs/verifier /sealed /workspace/submission
CMD ["sleep", "infinity"]

FROM base AS grader
WORKDIR /tests
COPY tests/common.py tests/grader.py /tests/
COPY environment/evaluation /evaluation
COPY environment/data/label_rules.py /data/label_rules.py
CMD ["python", "/tests/grader.py", "--help"]

FROM grader AS unit-tests
COPY tests /tests
COPY . /task
RUN chmod 755 /tests/run_unit_tests.sh
CMD ["/tests/run_unit_tests.sh"]
