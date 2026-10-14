# Visible development evaluation

The four seasonal 2021 cases are visible for model selection. They are never
training data or normalization inputs. Test truth is absent from this container.

The development runner uses the same `wn2-inference/v2` submission entry point
as the formal test. It requests eight members at 12-hour leads through ten days,
the available development truth horizon, and scores native weather plus 12-hour
precipitation. It is a development diagnostic, not full test acceptance.

```bash
export WEATHER_MODEL_PYTHON=/opt/weather-python/bin/python
export PYTHONPATH=/opt/weatherbenchX
export OPENBLAS_NUM_THREADS=2
/opt/wn2-eval/bin/python /workspace/evaluation/development.py \
  --package /workspace/submission --data /workspace/data/development \
  --scales /workspace/data/training/statistics/hres_025deg_6h/std.nc \
  --output /workspace/research/development-run-001
```

Choose a new output directory for each run and use one allocated GPU. The runner
relocates the model and renames case IDs before offline, context-only inference.
`scores.json` contains per-variable weather and precipitation results and the
normalized weather reward. Docker operators must allow Bubblewrap namespaces.
