FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# `curl` is needed by fetch_designability_weights.sh below; the base image ships neither
# it nor git.
# gcc/g++: needed to build the `prody` sdist (no cp311 wheel); see environment/Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Foldseek 10-941cd33 (GPL-3.0), pinned by sha256: the grader's novelty gate (tests/novelty.py) searches every
# designable design against the fit corpus in TM-align mode; agents can run the same check in development.
RUN curl -fsSL -o /tmp/foldseek.tar.gz \
      https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz && \
    echo "e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7  /tmp/foldseek.tar.gz" | sha256sum -c - && \
    tar --no-same-owner -xzf /tmp/foldseek.tar.gz -C /opt && rm /tmp/foldseek.tar.gz
ENV FOLDSEEK=/opt/foldseek/bin/foldseek

# numpy is 2.x, not 1.26.4: the grader now imports `rf3`, which reaches atomworks and
# `biotite==1.4.0`, whose wheels are built against the numpy 2 ABI. The 23 unit tests
# pass unchanged under numpy 2.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      "numpy>=2.1,<3" gemmi==0.7.5 && \
    useradd --create-home --uid 10001 agent

# Inverse folding: SOURCE ONLY, from the ORIGINAL LigandMPNN repository - not foundry's
# re-implementation; see tests/pipeline.py for the measurement that decided it. VENDORED
# at environment/evaluation/ligandmpnn (revision 26ec57ac, MIT) rather than cloned, so the grader
# image is built from code in this repository; see that directory's VENDORED.md.
# `run.py` does `from prody import *` at module scope, so prody must be installed below
# or the scorer dies on its first call.
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

# Complex prediction is RoseTTAFold3, and the grader imports `rf3` directly, so the
# verifier needs the same vendored source and dependency stack as the agent image. That
# is the cost of the scorer being a real model rather than a mounted binary. The list
# mirrors the agent image's; `rc-foundry` is deliberately NOT installed from PyPI,
# because a second `foundry` in site-packages would compete with the vendored source.
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      "atomworks[ml]==2.2.1" \
      "lightning>=2.5.0" "hydra-core>=1.3.0" "rootutils>=1.0.7" "environs>=11.0.0" \
      "rich>=13.9.4" "jaxtyping>=0.2.17" "beartype>=0.18.0" \
      "typer>=0.20.0" "einops>=0.8.0" "einx>=0.1.0" \
      "opt_einsum>=3.4.0" "dm-tree>=0.1.6" "zstandard" "pandas>=2.2" \
      "assertpy" "toolz" "pydantic>=2.8" \
      "prody>=2.4" "ml_collections>=0.1.1"

# The vendored RF3/foundry source. The build context is the task root (see
# docker-compose.yaml), so these are the same trees the agent image ships.
#
# TWO copies, because they are two packages with a real dependency between them: `rf3`
# imports `foundry` at 29 sites (trainers, samplers, data transforms), and `foundry`
# lives in the starter's vendored tree while the predictor now lives under
# environment/evaluation/. Dropping either one breaks `import rf3` at module scope, and
# therefore every scorer call on every track.
COPY environment/baseline/rfd3-small/rfd3 /opt/foundry
COPY environment/evaluation/rf3 /opt/rf3

# The scoring parameters: LigandMPNN's .pt files (~17 MB) and the RF3 checkpoint
# (~2.8 GB), all public and pinned by sha256. Baked in, so the verifier mounts nothing.
COPY environment/fetch_designability_weights.sh /usr/local/bin/fetch-designability-weights
RUN chmod 755 /usr/local/bin/fetch-designability-weights && \
    bash /usr/local/bin/fetch-designability-weights /assets

ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    LIGANDMPNN_ROOT=/opt/ligandmpnn \
    PROTEIN_DESIGN_ASSETS=/assets \
    PROJECT_ROOT=/opt/foundry \
    PYTHONPATH=/opt/foundry/src:/opt/rf3/src

# Pretrained assets the agent DECLARES rather than ships (contract.md). It must exist
# and sit at the SAME path in development and verification, because common.py resolves
# every declared asset under it. Empty by default; an operator bind-mounts a populated
# host directory over it for the predictor_inversion and hybrid paradigms, which are
# REFUSED without at least one declared asset. See docker-compose.yaml.
RUN mkdir -p /pretrained

COPY tests/ /tests/
RUN chmod +x /tests/test.sh /tests/run_unit_tests.sh
WORKDIR /tests
