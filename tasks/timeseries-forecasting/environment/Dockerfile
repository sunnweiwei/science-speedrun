# Development environment for the timeseries-forecasting task. Build context = the task directory
# (timeseries-forecasting/), which environment/docker-compose.yaml sets; this file copies
# instruction.md and environment/... .
#   docker build -f environment/Dockerfile -t timeseries-forecasting-dev .
# cuda12.6 rather than cuda13.0: the cu130 build needs driver >= 580, which H100 hosts on driver 570 lack.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
# the image's Python is Ubuntu 24.04's system interpreter (PEP 668), hence PIP_BREAK_SYSTEM_PACKAGES
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends git bubblewrap && rm -rf /var/lib/apt/lists/*
COPY environment/runtime/requirements.txt environment/runtime/constraints.txt /opt/runtime/
RUN grep -vE "^(torch==|#)" /opt/runtime/requirements.txt > /opt/runtime/req-notorch.txt && \
    pip install -c /opt/runtime/constraints.txt -r /opt/runtime/req-notorch.txt
# The agent's workspace (TASK_STANDARD.md): documents, data docs (the datasets are mounted next to them),
# verified baselines, unverified references, development evaluation. The external-target replay packages
# load forbidden checkpoints and stay with the maintainers (docs/validation/released_models).
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md environment/data/split.json environment/data/split_smoke.json environment/data/sources.lock.json /workspace/data/
COPY environment/evaluation/ts_io.py environment/evaluation/score.py environment/evaluation/make_cases.py environment/evaluation/make_example.py \
     environment/evaluation/README.md environment/evaluation/seasonal_naive_reference_valid.json /workspace/evaluation/
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
COPY environment/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /workspace/submission /workspace/research && \
    cp /workspace/evaluation/ts_io.py /workspace/baseline/starter/ts_io.py
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
