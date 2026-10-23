# Hold-current anchor: results

## v0.3 suite, `tokamak-shape-metrics/v4` (sealed-v0.3, seeds 1001-1015, 105 episodes)

`harbor_verify.py` with the sealed file (the replay `tests/test.sh` runs), 2026-10-23: `plasma_lost` in 105/105
episodes, `reward` 0.0, `tolerance_time_fraction` 0.0, `policy_compute_ms` 0.021 (measured after the
replay), no solver failure and no rerun; 8 steps needed the fallback ladder. Wall time
107 min with 38 workers (three waves).

## v0.3 (`tokamak-shape-metrics/v3`, sealed seeds 1001-1005)

`tests/test.sh`, 2026-10-05: `plasma_lost` in 35/35 episodes, `reward` 0.0, `policy_compute_ms` 0.045.
