FROM python:3.12.14-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ARG WEATHERNEXT_REVISION=89c4b2a77a1c57b328b909c575550fd2e5aadc9c
ARG XARRAY_JAX_REVISION=0291d0c8e91ca04b00f5f01ea29a53c0e575eeee
ARG WEATHERBENCHX_REVISION=88af2fde7a78f13b77496a2abb8ca88905fa5e95
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PIP_NO_INDEX=1
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap ca-certificates curl git libspatialindex-dev && rm -rf /var/lib/apt/lists/*
RUN python -m venv --copies /opt/weather-python
ENV PATH=/opt/weather-python/bin:${PATH}
COPY environment/weathernext/wn2/model-requirements.lock.txt /opt/wn2-build/model-requirements.txt
RUN PIP_NO_INDEX=0 python -m pip install --no-cache-dir -r /opt/wn2-build/model-requirements.txt
RUN git init /opt/xarray-jax && git -C /opt/xarray-jax remote add origin https://github.com/google-deepmind/xarray_jax.git && git -C /opt/xarray-jax fetch --depth 1 origin ${XARRAY_JAX_REVISION} && git -C /opt/xarray-jax checkout --detach FETCH_HEAD && PIP_NO_INDEX=0 python -m pip install --no-cache-dir --no-deps -e /opt/xarray-jax
RUN git init /opt/weathernext && git -C /opt/weathernext remote add origin https://github.com/google-deepmind/weathernext.git && git -C /opt/weathernext fetch --depth 1 origin ${WEATHERNEXT_REVISION} && git -C /opt/weathernext checkout --detach FETCH_HEAD && PIP_NO_INDEX=0 python -m pip install --no-cache-dir --no-deps -e /opt/weathernext
RUN /usr/local/bin/python -m venv --copies /opt/wn2-eval
COPY environment/weathernext/wn2/requirements.lock.txt /opt/wn2-build/eval-requirements.txt
RUN PIP_NO_INDEX=0 /opt/wn2-eval/bin/python -m pip install --no-cache-dir -r /opt/wn2-build/eval-requirements.txt
RUN git init /opt/weatherbenchX && git -C /opt/weatherbenchX remote add origin https://github.com/google-research/weatherbenchX.git && git -C /opt/weatherbenchX fetch --depth 1 origin ${WEATHERBENCHX_REVISION} && git -C /opt/weatherbenchX checkout --detach FETCH_HEAD
ENV WEATHER_MODEL_PYTHON=/opt/weather-python/bin/python WEATHER_EVAL_PYTHON=/opt/wn2-eval/bin/python WEATHER_WBX_SOURCE=/opt/weatherbenchX
# END SHARED RUNTIME: model and CPU evaluator have separate pinned environments.

RUN apt-get update && apt-get install -y --no-install-recommends g++ make libnetcdf-dev xutils-dev && rm -rf /var/lib/apt/lists/*
COPY tests/build_tracker.py /opt/wn2-build/build_tracker.py
COPY environment/weathernext/paper-comparison/tracker-source.lock.json /opt/wn2-build/tracker-source.lock.json
RUN /usr/local/bin/python /opt/wn2-build/build_tracker.py --lock /opt/wn2-build/tracker-source.lock.json --output /opt/tempestextremes/bin
ENV WEATHER_TRACKER_BIN=/opt/tempestextremes/bin

WORKDIR /workspace
COPY environment/weathernext/wn2 /tests/wn2
COPY environment/weathernext/paper-comparison /tests/paper-comparison
COPY tests/test.sh /tests/test.sh
RUN ln -sf ../wn2/protocol.json /tests/paper-comparison/weather-protocol.json && ln -sf ../wn2/precipitation-protocol.json /tests/paper-comparison/precipitation-protocol.json
CMD ["sleep", "infinity"]
