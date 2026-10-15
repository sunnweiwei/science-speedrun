FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 10001 agent
WORKDIR /tests
COPY . /tests
RUN chmod 755 /tests/test.sh
CMD ["/tests/test.sh"]
