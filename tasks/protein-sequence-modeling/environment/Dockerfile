FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 10001 agent

WORKDIR /workspace
# The agent's workspace (TASK_STANDARD.md): contract, data README (the datasets are mounted next to
# it), verified baselines, measured-run references, development evaluation.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
COPY environment/evaluation /workspace/evaluation
COPY environment/baseline /workspace/baseline
COPY environment/references /workspace/references
COPY environment/entrypoint.sh /usr/local/bin/task-entrypoint
# submission holds the currently published candidate; research holds the loop's
# own history and checkpoints, which must never leak into the replay package.
# Normalize permissions explicitly: COPY preserves the build context's mode
# bits, so a checkout with a restrictive umask would otherwise bake unreadable
# files into the image and the result would differ between builders.
RUN mkdir -p /workspace/submission /workspace/research \
 && chmod -R a+rX /workspace \
 && chmod 755 /usr/local/bin/task-entrypoint /workspace/baseline/trivial/infer.sh \
 && chown -R agent:agent /workspace
# The agent owns its whole workspace: it is expected to modify the starter, and
# publishing stages a directory beside /workspace/submission before renaming it
# into place, which needs write access to /workspace itself.
ENTRYPOINT ["/usr/local/bin/task-entrypoint"]
CMD ["sleep", "infinity"]
