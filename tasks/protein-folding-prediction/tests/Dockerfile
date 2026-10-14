FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

ARG TRANSFORMERS_REVISION=ef32577f55da19a4989cd7b22e004dc43a4998cb
ARG ESM_REVISION=917af90b624535eed1e072d343c717e3ec11fef4
ARG FOLDBENCH_REVISION=4273f6877d82bd0b2fa476d1b2f34d121cbccc70
ARG ESMFOLD2_CCD_REVISION=1ebf0e3481a5184eb6171d40615c79e384b48796
ARG PROTENIX_REVISION=c3bfc365b3e1341a11935eddfe7bfdc308092147
ARG OPENDDE_REVISION=f607bb3c9ff299c0627ac20f5ef8e25d716ed46f
ARG DPLM_REVISION=8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d

ENV HF_HOME=/data/hf \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl g++ gcc git libc6-dev make && \
    rm -rf /var/lib/apt/lists/*

RUN git init /opt/transformers && \
    git -C /opt/transformers remote add origin https://github.com/huggingface/transformers.git && \
    git -C /opt/transformers fetch --depth 1 origin ${TRANSFORMERS_REVISION} && \
    git -C /opt/transformers checkout --detach FETCH_HEAD && \
    python -m pip install --no-cache-dir -e /opt/transformers

RUN git init /opt/esm && \
    git -C /opt/esm remote add origin https://github.com/Biohub/esm.git && \
    git -C /opt/esm fetch --depth 1 origin ${ESM_REVISION} && \
    git -C /opt/esm checkout --detach FETCH_HEAD && \
    python -m pip install --no-cache-dir \
        biopython==1.85 biotite==1.5.0 brotli==1.1.0 \
        cloudpathlib==0.21.1 einops==0.8.1 msgpack-numpy==0.4.8 \
        pytest==8.4.1 rdkit==2026.3.4 scipy==1.15.3 zstd==1.5.7.3 && \
    python -m pip install --no-cache-dir --no-deps \
        --ignore-requires-python -e /opt/esm

RUN HF_HUB_DISABLE_XET=1 python -c \
      "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='biohub/ESMFold2', filename='ccd.pkl', revision='${ESMFOLD2_CCD_REVISION}', local_dir='/opt/esmfold2-assets')" && \
    echo "9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5  /opt/esmfold2-assets/ccd.pkl" | sha256sum -c -

RUN for spec in \
      "Protenix https://github.com/bytedance/Protenix.git ${PROTENIX_REVISION}" \
      "OpenDDE https://github.com/aurekaresearch/OpenDDE.git ${OPENDDE_REVISION}" \
      "dplm https://github.com/bytedance/dplm.git ${DPLM_REVISION}"; do \
        set -- ${spec}; \
        git init "/opt/references/$1"; \
        git -C "/opt/references/$1" remote add origin "$2"; \
        git -C "/opt/references/$1" fetch --depth 1 origin "$3"; \
        git -C "/opt/references/$1" checkout --detach FETCH_HEAD; \
    done

RUN git init /opt/foldbench && \
    git -C /opt/foldbench remote add origin https://github.com/BEAM-Labs/FoldBench.git && \
    git -C /opt/foldbench fetch --depth 1 origin ${FOLDBENCH_REVISION} && \
    git -C /opt/foldbench checkout --detach FETCH_HEAD && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/2.3.2 \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    MAMBA_ROOT_PREFIX=/opt/micromamba micromamba create -y \
      -n foldbench -c conda-forge -c aivant \
      python=3.9 openstructure=2.8.0 libtiff=4.4.0 pip && \
    /opt/micromamba/envs/foldbench/bin/python -m pip install --no-cache-dir \
      biopython==1.85 networkx numpy pandas parallelbar scipy tqdm && \
    ln -s /opt/micromamba/envs/foldbench /opt/foldbench-env

# END SHARED RUNTIME: intentionally identical in the two independent build contexts.

ENV HF_HOME=/tmp/huggingface
WORKDIR /workspace
COPY . /tests
CMD ["sleep", "infinity"]
