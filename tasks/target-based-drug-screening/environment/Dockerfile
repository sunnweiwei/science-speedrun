FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir numpy==2.2.6 lmdb==1.6.2 rdkit==2025.3.3

WORKDIR /workspace
# Harbor's build context is the task root, matching protein-folding-prediction.
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
# Only working model code and provenance enter the agent image.
COPY environment/baseline/drugclip/train.py environment/baseline/drugclip/train.sh \
     environment/baseline/drugclip/model.py environment/baseline/drugclip/unimol_encoder.py \
     environment/baseline/drugclip/data.py environment/baseline/drugclip/common.py \
     environment/baseline/drugclip/training.py environment/baseline/drugclip/validation.py \
     environment/baseline/drugclip/predict.py environment/baseline/drugclip/infer.sh \
     environment/baseline/drugclip/requirements.txt environment/baseline/drugclip/README.md \
     environment/baseline/drugclip/RESULTS.md environment/baseline/drugclip/TRAINING.md \
     environment/baseline/drugclip/model_manifest.example.json \
     environment/baseline/drugclip/pretrained_manifest.example.json /workspace/baseline/drugclip/
COPY environment/baseline/drugclip/reproduction/README.md \
     environment/baseline/drugclip/reproduction/development_bedroc.csv \
     environment/baseline/drugclip/reproduction/gpu_verification.json \
     environment/baseline/drugclip/reproduction/independent_replay.json \
     environment/baseline/drugclip/reproduction/independent_replay_v2.json /workspace/baseline/drugclip/reproduction/
COPY environment/references/minimal-starter/train.py environment/references/minimal-starter/model.py \
     environment/references/minimal-starter/data.py environment/references/minimal-starter/common.py \
     environment/references/minimal-starter/predict.py environment/references/minimal-starter/infer.sh \
     environment/references/minimal-starter/requirements.txt environment/references/minimal-starter/README.md \
     environment/references/minimal-starter/model_manifest.example.json /workspace/references/minimal-starter/
COPY environment/evaluation/README.md environment/evaluation/metrics.py environment/evaluation/aggregation.py environment/evaluation/dev.py /workspace/evaluation/
COPY environment/publish.py /workspace/publish.py
COPY environment/record_environment.py /workspace/record_environment.py
RUN mkdir -p /workspace/submission /workspace/research
CMD ["sleep", "infinity"]
