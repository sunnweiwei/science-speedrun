FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY environment/requirements.txt /opt/bci/requirements.txt
RUN pip install --index-url https://pypi.org/simple --no-cache-dir -r /opt/bci/requirements.txt
WORKDIR /workspace
COPY instruction.md /workspace/instruction.md
COPY environment/contract.md /workspace/contract.md
COPY environment/data/README.md /workspace/data/README.md
COPY environment/baseline/ /workspace/baseline/
COPY environment/references/ /workspace/references/
COPY environment/evaluation/ /workspace/evaluation/
RUN mkdir -p /workspace/research /workspace/submission
CMD ["sleep","infinity"]
