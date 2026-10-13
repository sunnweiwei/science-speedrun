FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime@sha256:0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361

COPY requirements.txt /tmp/task-requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      -r /tmp/task-requirements.txt \
    && python -c "import duckdb, matplotlib, numpy, pandas, pyarrow, seaborn, shap, sklearn, statsmodels, torch, torchvision, tqdm, umap"

# Thread pools follow task.toml's cpus = 10. Without this PyTorch sizes them by the host's
# cores, and under a CPU quota on a large host the reference GRU ran ~15x slower (104 threads).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    OMP_NUM_THREADS=10

RUN useradd --create-home --uid 10001 agent

WORKDIR /workspace
COPY contract.md /workspace/contract.md
RUN mkdir -p /workspace/baseline /workspace/references /workspace/data \
      /workspace/submission /workspace/research \
      /workspace/data/train /workspace/data/public/validation \
      /workspace/data/trusted/validation
COPY data/README.md data/label_rules.py /workspace/data/
COPY baseline /workspace/baseline
COPY references /workspace/references
COPY evaluation /workspace/evaluation
COPY entrypoint.sh /usr/local/bin/task-entrypoint
RUN chmod 755 /usr/local/bin/task-entrypoint \
      /workspace/references/scaffold/infer.sh \
      /workspace/references/scaffold/train.py \
      /workspace/baseline/temporal-gru/package.sh \
      /workspace/references/delphi/train_demo.sh \
      /workspace/references/delphi/smoke_train.sh \
    && chown -R agent:agent /workspace

USER agent
ENTRYPOINT ["/usr/local/bin/task-entrypoint"]
CMD ["sleep", "infinity"]
