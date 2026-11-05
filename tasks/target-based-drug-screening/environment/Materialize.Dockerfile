FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends mmseqs2 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY data/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY data /opt/target-based-drug-screening-data
ENV PYTHONPATH=/opt/target-based-drug-screening-data
WORKDIR /workspace
CMD ["bash"]

