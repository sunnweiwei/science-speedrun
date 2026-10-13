# Development environment for the pde-surrogate task. Build context = the task directory
# (pde-surrogate/), which environment/docker-compose.yaml sets; this file copies instruction.md
# and environment/... .
#   docker build -f environment/Dockerfile -t pde-surrogate-dev .
# cuda12.6 rather than cuda13.0: the cu130 build needs driver >= 580, which H100 hosts on driver 570 lack.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
# the image's Python is Ubuntu 24.04's system interpreter (PEP 668), hence PIP_BREAK_SYSTEM_PACKAGES
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# torch comes from the base image; the rest is pinned, and constraints.txt keeps the image's torch build
COPY environment/runtime/requirements.txt environment/runtime/constraints.txt /opt/runtime/
RUN grep -vE "^(torch==|#)" /opt/runtime/requirements.txt > /opt/runtime/req-notorch.txt && \
    pip install -c /opt/runtime/constraints.txt -r /opt/runtime/req-notorch.txt
# The Well's source at the commit the CNextU-net baseline uses: its Hydra configs are not in the wheel
RUN git init -q /opt/the_well && \
    git -C /opt/the_well fetch -q --depth 1 https://github.com/PolymathicAI/the_well.git ad50de0e8861380d7c4d62c0503c654710f026e7 && \
    git -C /opt/the_well checkout -q --detach FETCH_HEAD && rm -rf /opt/the_well/.git
# The agent's workspace (TASK_STANDARD.md): documents, data docs (the datasets are mounted next to
# them), verified baselines, unverified references, development evaluation. The replay packages for
# released checkpoints and their fine-tune configs stay with the maintainers (docs/validation/released_models):
# their only purpose is loading assets the task forbids.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md environment/data/AUDIT.md environment/data/split.json environment/data/split_smoke.json environment/data/sources.lock.json /workspace/data/
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
COPY environment/evaluation /workspace/evaluation
COPY environment/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY environment/data/check_agent_mount.py /opt/pde-surrogate/check_agent_mount.py
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /workspace/submission /workspace/research
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
