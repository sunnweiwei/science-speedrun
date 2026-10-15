FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# gcc/g++: `prody` publishes no cp311 wheel for this base image, so pip builds the 2.6.1
# sdist, which compiles C extensions. Without a compiler the image does not build at all.
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Foldseek 10-941cd33 (GPL-3.0), pinned by sha256: the grader's novelty gate (tests/novelty.py) searches every
# designable design against the fit corpus in TM-align mode; agents can run the same check in development.
RUN curl -fsSL -o /tmp/foldseek.tar.gz \
      https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz && \
    echo "e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7  /tmp/foldseek.tar.gz" | sha256sum -c - && \
    tar --no-same-owner -xzf /tmp/foldseek.tar.gz -C /opt && rm /tmp/foldseek.tar.gz
ENV FOLDSEEK=/opt/foldseek/bin/foldseek

# Fixed image dependencies; the agent never installs these.
#
# numpy is 2.x, not the 1.26.4 this image used to pin: `biotite==1.4.0`, which atomworks
# requires and which RFdiffusion3's whole data pipeline is built on, ships wheels built
# against the numpy 2 ABI. Holding numpy at 1.26 would make `import biotite` fail at
# runtime. Verified compatible: the 23 tests in tests/ pass unchanged under numpy 2.4.6,
# and gemmi 0.7.5 imports and parses against it.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      "numpy>=2.1,<3" gemmi==0.7.5 && \
    useradd --create-home --uid 10001 agent

# Inverse folding: SOURCE ONLY, from the ORIGINAL LigandMPNN repository, now VENDORED
# at environment/evaluation/ligandmpnn (revision 26ec57ac, MIT) rather than cloned at build time -
# so the build does not depend on a third-party clone succeeding and the exact scoring
# code is in the repository. See that directory's VENDORED.md for what was and was not
# taken. Foundry ships a re-implementation in `models/mpnn`, and upstream itself says to
# use the old repositories for benchmarking until it stabilises - correctly, as its
# `ligand_mpnn` was measured returning byte-identical sequences with and without a bound
# ligand. See tests/pipeline.py for that measurement.
#
# Complex prediction is RoseTTAFold3, whose source is VENDORED in
# environment/evaluation/rf3 (nothing to clone here either) and whose parameters
# are a public download. AlphaFold 3 is gone: its parameters cannot be redistributed,
# which is precisely what kept designability off the agent's own scoring path.
COPY environment/evaluation/ligandmpnn /opt/ligandmpnn

# LigandMPNN vendors a copy of openfold that predates NumPy 2 and uses the aliases
# `np.int`, `np.object` and `np.bool`, all REMOVED in NumPy 2. This image pins
# numpy>=2.1 (biotite 1.4.0 needs the numpy-2 ABI), and `run.py` imports sc_utils ->
# openfold at module scope, so without this every scorer call dies at import with
# `AttributeError: module 'numpy' has no attribute 'int'` - on every track. Six
# occurrences across three files, none in LigandMPNN's own model code.
RUN sed -i 's/\bnp\.int\b/int/g; s/\bnp\.object\b/object/g; s/\bnp\.bool\b/bool/g' \
      /opt/ligandmpnn/openfold/np/residue_constants.py \
      /opt/ligandmpnn/openfold/data/templates.py \
      /opt/ligandmpnn/openfold/np/relax/utils.py && \
    ! grep -rn 'np\.int\b\|np\.object\b\|np\.bool\b' /opt/ligandmpnn --include=*.py

# Reference design systems: SOURCE ONLY, pinned by revision. No generator weights are
# supplied - vendoring a finished design system's checkpoints would reduce this task to
# selecting among them.
ARG RFD3_REVISION=b02eed6a6bdf8f44d14a80cc36e3da13c9f2291c
ARG BOLTZGEN_REVISION=a3149cf18eeb58648d1abbb27539bd73f746cdda
ARG DISCO_REVISION=82b594f838eb61dd8c78ae3a403ccb8f00cf7abf
ARG PROTEINA_COMPLEXA_REVISION=33b76585801eb902802cc7410c85bc37ae81b346
ARG BOLTZDESIGN1_REVISION=31aeedd7f187ba88dd9e92514070aee13381cb93
RUN for spec in \
      "rfd3 https://github.com/RosettaCommons/foundry.git ${RFD3_REVISION}" \
      "boltzgen https://github.com/HannesStark/boltzgen.git ${BOLTZGEN_REVISION}" \
      "disco https://github.com/DISCO-design/DISCO.git ${DISCO_REVISION}" \
      "proteina_complexa https://github.com/NVIDIA-BioNeMo/Proteina-Complexa.git ${PROTEINA_COMPLEXA_REVISION}" \
      "boltzdesign1 https://github.com/yehlincho/BoltzDesign1.git ${BOLTZDESIGN1_REVISION}"; do \
        set -- ${spec}; \
        git init "/opt/references/$1"; \
        git -C "/opt/references/$1" remote add origin "$2"; \
        git -C "/opt/references/$1" fetch --depth 1 origin "$3"; \
        git -C "/opt/references/$1" checkout --detach FETCH_HEAD; \
        rm -rf "/opt/references/$1/.git"; \
    done

# RFdiffusion3's own training stack. The starter vendors the rfd3 and foundry SOURCE
# (environment/baseline/rfd3-small/rfd3, pinned at the same revision as /opt/references/rfd3) but
# not its dependencies, so they are installed here while the build still has a network -
# the runtime does not. No generator weights are installed: `foundry install rfd3` would
# fetch a finished checkpoint, which is exactly what this task does not supply.
#
# The dependency list mirrors `starter/rfd3/pyproject.toml` (its `dependencies` plus the
# `rfd3` extra) rather than installing `rc-foundry` from PyPI. Two reasons: PyPI's
# release is not the pinned revision, and installing it would put a second copy of
# `foundry` in site-packages competing with the vendored source on PYTHONPATH. `torch`
# is omitted because the base image supplies it (>=2.2.0 is all that is asked) and
# `ipykernel` because it is a notebook convenience.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      "atomworks[ml]==2.2.1" \
      "lightning>=2.5.0" "hydra-core>=1.3.0" "rootutils>=1.0.7" "environs>=11.0.0" \
      "wandb>=0.15.10" "rich>=13.9.4" "jaxtyping>=0.2.17" "beartype>=0.18.0" \
      "typer>=0.20.0" "loralib>=0.1.1" "einops>=0.8.0" "einx>=0.1.0" \
      "opt_einsum>=3.4.0" "dm-tree>=0.1.6" "zstandard" "pandas>=2.2" \
      "assertpy" "toolz" "pydantic>=2.8" && \
    python -c "import torch, numpy, gemmi, atomworks, lightning; \
print('torch', torch.__version__, 'numpy', numpy.__version__, 'gemmi', gemmi.__version__)"

# LigandMPNN's own imports. `run.py` does `from prody import *` at module scope, so
# without these the scorer dies on its first call with ModuleNotFoundError.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      "prody>=2.4" "ml_collections>=0.1.1"

# The scoring parameters. BOTH sets are public downloads pinned by sha256, which is the
# whole reason designability can now ship: LigandMPNN's .pt files (~17 MB) and the RF3
# checkpoint (~2.8 GB). The script is re-runnable on any cluster, so an operator without
# this image can reproduce /assets exactly.
COPY environment/fetch_designability_weights.sh /usr/local/bin/fetch-designability-weights
RUN chmod 755 /usr/local/bin/fetch-designability-weights && \
    bash /usr/local/bin/fetch-designability-weights /assets

ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LIGANDMPNN_ROOT=/opt/ligandmpnn \
    PROTEIN_DESIGN_ASSETS=/assets \
    PROTEIN_DESIGN_REFERENCES=/opt/references \
    PROJECT_ROOT=/workspace/baseline/rfd3-small/rfd3 \
    PYTHONPATH=/workspace/baseline/rfd3-small/rfd3/src:/workspace/baseline/rfd3-small/rfd3/models/rfd3/src:/workspace/evaluation/rf3/src \
    PROTEIN_DESIGN_DATA=/data \
    PROTEIN_DESIGN_WORK=/workspace/research

# END SHARED RUNTIME

# Build context is the TASK ROOT so the grading code can be shared with the
# development image instead of being duplicated in git.
WORKDIR /workspace
# Pretrained assets the agent DECLARES rather than ships (contract.md). It must exist
# and sit at the SAME path in development and verification, because common.py resolves
# every declared asset under it. Empty by default; an operator bind-mounts a populated
# host directory over it for the predictor_inversion and hybrid paradigms, which are
# REFUSED without at least one declared asset. See docker-compose.yaml.
RUN mkdir -p /pretrained

COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
# The index builder is the one part of materialization the agent may legitimately rerun:
# it turns structures into the two parquet tables RFD3 trains from, offline, and an agent
# that wants a different corpus or a different clustering has to be able to rebuild them.
# The rest of materialize.py stays operator-only - it needs the network.
COPY environment/data/build_training_index.py /workspace/data/build_training_index.py
COPY environment/evaluation /workspace/evaluation
COPY environment/references /workspace/references
COPY environment/baseline/rfd3-small /workspace/baseline/rfd3-small
COPY tests/common.py tests/geometry.py tests/pipeline.py tests/grader.py tests/novelty.py \
     tests/residues.py /workspace/evaluation/lib/
COPY environment/entrypoint.sh /usr/local/bin/protein-design-entrypoint
RUN mkdir -p /workspace/submission /workspace/research /workspace/evaluation/lib && \
    chmod 755 /usr/local/bin/protein-design-entrypoint && \
    chown -R agent:agent /workspace
# The development corpus is mounted at the standard /workspace/data (TASK_STANDARD.md).
# /data is the path this task's scripts and docs have always used, kept as a link so both
# resolve to the same tree.
RUN ln -s /workspace/data /data
ENTRYPOINT ["/usr/local/bin/protein-design-entrypoint"]
CMD ["sleep", "infinity"]
