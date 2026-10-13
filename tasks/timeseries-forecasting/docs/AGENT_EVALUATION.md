# Agent evaluation

## Status

An agent pilot is out of scope for this contribution (agreed with the
maintainers): the contributor's machine has no Docker, so neither the
development image nor the verifier image has been built here. Everything an
agent run needs is in place and was exercised outside Docker:

- the data mount (`materialize.py` receipt, every file of the lock verified), the split
  manifests, the validation cases (`/workspace/data/valid`) and the starter pipeline
  (pack → train → publish → infer) on the real corpus;
- the verifier path (`tests/main_verifier.py`) on the smoke split and the full
  test split for the reference packages and the task-trained baseline
  (`docs/validation/`, `docs/baseline_score.json`);
- the unit tests (`tests/run_unit_tests.sh`).

## What a pilot should check first

1. `docker compose -f environment/docker-compose.yaml build` and
   `-f tests/docker-compose.yaml build` with `TASK_DATA_DIR` set (both
   unbuilt here; the Dockerfiles copy from the task directory, which the
   overlays set as context; pins in `environment/runtime/requirements.txt`).
2. The agent's `/data` view must not contain `gifteval/` (`entrypoint.sh`
   refuses to start); the verifier's overlay mounts only `gifteval/`.
3. A 2-hour smoke run with `compute-profiles.toml` profile `smoke`, the
   `fast` corpus tier and `split_smoke.json`, then `tests/test.sh` on the
   produced package with `TIMESERIES_SPLIT_MANIFEST=/tests/split_smoke.json`.
4. Inference wall-clock on the full test split against the 4-hour limit; the
   6.4M-parameter starter needs about 10 minutes on an A100 across the 82 calls
   (see `docs/validation/task_trained_baseline.json`), and `reward.json` should say
   `"isolation": ["bwrap"]` (or `["uid"]` where the container has no user namespaces).

## Expected failure modes to watch

- Loading the corpus with `datasets.load_dataset()` (wrong layout; use
  `load_from_disk` or `ts_io.read_arrow_dataset`).
- Forecasting variates jointly but writing them in the wrong axis order
  (`quantiles` is `(N, D, 9, H)`).
- Quantile heads that cross (the grader sorts and reports the fraction).
- Training on `/workspace/data/valid` contexts (forbidden; the manifest must say so).
- Non-deterministic inference (TF32, cuDNN autotuning); the verifier reruns
  one cohort and compares bit for bit.
