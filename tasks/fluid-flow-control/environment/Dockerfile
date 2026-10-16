# Build context: task root (set explicitly in both native Compose definitions).
FROM clagemann/hydrogym-nvhpc-26.1_cuda-12.9_hopper_blackwell@sha256:5e8deb0b0efce2e2fb47934f0ad7dd2c2b4a430d2b9b72d47bfc2dd3aff39b03 AS runtime
USER root
ARG HYDROGYM_REVISION=4ab9854dea3d84e38a59c25e0f5835a00cf8225f
ARG TDMPC2_REVISION=e9f59321933cbc8e11a002b842adc7d4ffae8ff1
ARG HYDROGYM_AUTHOR_REVISION=20a483ecc6ea711f6b380fff05f212ca4d7e2c67
RUN source /etc/profile && module purge && module load MAIA/1.0-NVHPC-26.1 && \
    git init /opt/hydrogym && \
    git -C /opt/hydrogym remote add origin https://github.com/dynamicslab/hydrogym.git && \
    git -C /opt/hydrogym fetch --depth 1 origin ${HYDROGYM_REVISION} && \
    git -C /opt/hydrogym checkout --detach FETCH_HEAD && \
    /home/easybuild/venvs/hydrogym_gpu/bin/python -m pip install --no-deps /opt/hydrogym && \
    /home/easybuild/venvs/hydrogym_gpu/bin/python -m pip install --no-deps stable-baselines3==2.7.1 && \
    /home/easybuild/venvs/hydrogym_gpu/bin/python -m pip install torchrl==0.13.3 tensordict==0.13.0
RUN git init /opt/references/tdmpc2 && \
    git -C /opt/references/tdmpc2 remote add origin https://github.com/nicklashansen/tdmpc2.git && \
    git -C /opt/references/tdmpc2 fetch --depth 1 origin ${TDMPC2_REVISION} && \
    git -C /opt/references/tdmpc2 checkout --detach FETCH_HEAD
RUN git init /opt/references/hydrogym-author && \
    git -C /opt/references/hydrogym-author remote add origin https://github.com/cl126162/TheHydroGymProject.git && \
    git -C /opt/references/hydrogym-author fetch --depth 1 origin ${HYDROGYM_AUTHOR_REVISION} && \
    git -C /opt/references/hydrogym-author checkout --detach FETCH_HEAD
RUN source /etc/profile && module purge && module load MAIA/1.0-NVHPC-26.1 && \
    /home/easybuild/venvs/hydrogym_gpu/bin/python -m pip install pyarrow==19.0.1 netCDF4==1.7.2 \
    pyvista==0.46.4 vtk==9.5.2 pooch==1.9.0 scooby==0.11.2
ENV PATH=/home/easybuild/venvs/hydrogym_gpu/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HYDROGYM_AUTHOR_SOURCE=/opt/references/hydrogym-author \
    HYDROGYM_UPSTREAM_SOURCE=/opt/hydrogym \
    PYTHONPATH=/opt/hydrogym \
    HYDROGYM_EVALUATOR_CODE=/opt/hydrogym-task/evaluation \
    HF_HOME=/tmp/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
COPY environment/evaluation /opt/hydrogym-task/evaluation
COPY environment/data /opt/hydrogym-task/data
WORKDIR /workspace
# Unprivileged accounts for submitted policy processes (see evaluation/protocol.py): one per case
# that can run at the same time (evaluate_native.replay_cases gives each running case its own uid,
# 10001-10016), because closing a policy kills every process of its account. The base
# image's home is 0700, so without search permission on it the demoted policy cannot reach the
# Python toolchain and venv under /home/easybuild (every policy fails with "python: not found").
RUN useradd --create-home --uid 10001 hydrogym-policy && \
    for i in $(seq 2 16); do useradd --create-home --uid $((10000 + i)) hydrogym-policy-$i; done && \
    chmod o+x /home/easybuild && \
    mkdir -p /workspace/submission /logs/verifier

FROM runtime AS verifier
COPY tests/test.sh /tests/test.sh
CMD ["sleep", "infinity"]

FROM runtime AS development
# The agent's workspace (TASK_STANDARD.md). The release is mounted at /workspace/data/release; /data is a
# link to it because the evaluator and starter default to --data /data.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
# The whole directory: the evaluator in /workspace/evaluation imports its release code and reads its
# descriptors from the sibling data/ directory.
COPY environment/data /workspace/data
COPY environment/evaluation /workspace/evaluation
COPY environment/baseline /workspace/baseline
COPY environment/references/README.md /workspace/references/README.md
# Only the curated runnable subset of the specialist campaign goes into the image; its dated history stays in Git.
COPY environment/references/specialists/research_train.py environment/references/specialists/research_predict.py environment/references/specialists/public_learner.py environment/references/specialists/export_policy.py /workspace/references/specialists/
COPY environment/references/specialists/recipes /workspace/references/specialists/recipes
COPY environment/references/specialists/RESEARCH_BASELINES.md /workspace/references/specialists/README.md
COPY environment/references/specialists/verified /workspace/references/results
RUN mkdir -p /workspace/submission /workspace/research /workspace/data/release && ln -s /workspace/data/release /data
CMD ["sleep", "infinity"]
