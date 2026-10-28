FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends util-linux procps && rm -rf /var/lib/apt/lists/* && useradd -u 10001 -m runner
COPY environment/requirements.txt /opt/bci/requirements.txt
RUN pip install --index-url https://pypi.org/simple --no-cache-dir -r /opt/bci/requirements.txt
COPY environment/evaluation/ /opt/bci/evaluation/
COPY tests/main_verifier.py tests/test.sh /tests/
RUN chmod 755 /tests/test.sh && mkdir -p /private /logs/verifier && chmod 700 /private /tests /logs/verifier
WORKDIR /tests
CMD ["sleep","infinity"]
