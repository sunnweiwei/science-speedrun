FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir numpy==2.2.6 lmdb==1.6.2 rdkit==2025.3.3
WORKDIR /workspace
# Both compose overlays explicitly use the task root as the build context.
COPY tests/grade.py /tests/grade.py
COPY tests/test.sh tests/reaggregate.py /tests/
COPY environment/evaluation/score.py environment/evaluation/input_view.py environment/evaluation/metrics.py environment/evaluation/aggregation.py /tests/
COPY environment/publish.py /tests/publish.py
COPY environment/record_environment.py /tests/record_environment.py
# /private is root-only: demoted inference cannot traverse the benchmark mount.
RUN mkdir -p /private /workspace/submission /logs/verifier && \
    chmod 700 /private /tests /logs/verifier && chmod 755 /tests/test.sh
CMD ["sleep", "infinity"]
