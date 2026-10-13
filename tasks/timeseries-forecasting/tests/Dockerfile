# Independent verifier environment. Build context = the task directory (timeseries-forecasting/),
# which tests/docker-compose.yaml sets: this file copies environment/... as well as tests/.
#   docker build -f tests/Dockerfile -t timeseries-forecasting-verifier .
# cuda12.6 rather than cuda13.0: the cu130 build needs driver >= 580, which H100 hosts on driver 570 lack.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
# the image's Python is Ubuntu 24.04's system interpreter (PEP 668), hence PIP_BREAK_SYSTEM_PACKAGES
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Bubblewrap isolates inference from the labels when the container may create user namespaces
# (tests/docker-compose.yaml grants that); the `runner` uid is the fallback boundary when it may
# not. Both are in tests/isolation.py.
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 runner
COPY environment/runtime/requirements.txt environment/runtime/constraints.txt /opt/runtime/
RUN grep -vE "^(torch==|#)" /opt/runtime/requirements.txt > /opt/runtime/req-notorch.txt && \
    pip install --no-cache-dir -c /opt/runtime/constraints.txt -r /opt/runtime/req-notorch.txt
COPY tests/main_verifier.py tests/isolation.py tests/test.sh /tests/
COPY environment/evaluation/score.py environment/evaluation/make_cases.py environment/evaluation/ts_io.py \
     environment/evaluation/seasonal_naive_reference.json environment/evaluation/score_bounds.json /tests/evaluation/
COPY environment/data/split.json environment/data/split_smoke.json /tests/
# /private is where the compose overlay mounts the evaluation series and /tests holds the staging code and
# the split: both root-only, so a package running as the runner uid can regenerate no label from them.
RUN chmod 755 /tests/test.sh && mkdir -p /private && chmod 700 /private /tests
ENV PIP_NO_INDEX=1
