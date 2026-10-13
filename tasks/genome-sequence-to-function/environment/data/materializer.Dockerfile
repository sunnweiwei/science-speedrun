FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates gcc g++ libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agbench
COPY environment/data/pyproject.toml ./
COPY environment/data/src ./src
RUN python -m pip install --no-cache-dir '.[test]'

ENV SSL_CERT_FILE=/usr/local/lib/python3.12/site-packages/certifi/cacert.pem \
    CURL_CA_BUNDLE=/usr/local/lib/python3.12/site-packages/certifi/cacert.pem

ENTRYPOINT ["agbench"]
