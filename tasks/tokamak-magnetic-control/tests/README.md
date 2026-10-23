# Verifier and tests

`test.sh` is the Harbor verification entry: it replays `/workspace/submission` on
all cases with the sealed perturbation file `sealed/perturbations.json` (`sealed-v0.2`:
FAIR-MAST-grounded ranges plus the plant mechanisms named in `instruction.md`) and copies
the resulting `reward.json` (`reward` = success fraction) to `/logs/verifier/reward.json`. Build the verifier
image from the task directory:

```bash
docker build --target verifier -t tokamak-verifier -f environment/Dockerfile .
docker run --rm --network none --cpus 32 \
  -v /abs/submission:/workspace/submission:ro -v /abs/logs:/logs/verifier \
  --entrypoint bash tokamak-verifier /tests/test.sh
```

`sealed/` is copied into the verifier image only; the development image does not
contain it. Anyone with the Git repository can read it, so "sealed" means hidden
from the agent's container, not secret.

## CPU-only unit tests (no simulator)

```bash
docker run --rm --network none -v "$PWD/tests:/tests:ro" tokamak-verifier \
  python -m unittest discover -s /tests -p 'test_*.py'
```

- `test_metrics.py`: shape/X-point metric definitions, windows, the success tolerances,
  steady-state window, `tolerance_time_fraction`, the verifier's aggregation (every episode in
  the denominator, latency gate, invalid package), the bounded rerun policy and the step fallback
  ladder (scripted supervisor and environment).
- `test_protocol.py`: protocol failure handling with synthetic policies
  (out-of-bounds, wrong size, NaN, crash, bad observation).

## Simulator smoke (minutes, CPU)

```bash
docker run --rm --network none --cpus 4 tokamak-verifier \
  python /opt/tokamak-task/evaluation/smoke_env.py --cases mastu_diverted_hold --steps 10
```

Fallback ladder end to end (FreeGSNKE, two cases x two seeds): `TOKAGYM_TEST_FAULTS` makes chosen
steps fail, e.g. `mastu_diverted_hold:1001:30:exception:0` (recovered at level 1, same trajectory as
an undisturbed run) or `...:timeout:1` (recovered through the linearised step). It is a testing hook
and is never set by `test.sh`.

Full-horizon sealed replays of the references are recorded in the task README; the v0.1
tables (seeds 1001-1005, 500-step episodes, sealed-v0.1) are kept there for history.
