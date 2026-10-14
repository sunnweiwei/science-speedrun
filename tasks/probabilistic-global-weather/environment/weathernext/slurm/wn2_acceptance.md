# Unified submission acceptance

Run the declared eight-date, 56-member, 15-day protocol through the portable
submission entry point. The same forecast stores feed all metric families.

For the native task, `tests/test.sh` runs inference and scoring in the verifier
container. The default windowed pipeline provisions 1 TB and supports
lossless continuation; see [windowed evaluation](../wn2/WINDOWED_EVALUATION.md). The split workflow below retains all forecasts and requires 8 TiB.
GPU/CPU operators can split the phases to release GPUs before the
longer CPU diagnostics. Use the same immutable package, code and output directory:

```bash
# GPU allocation: four GPUs, pinned model runtime, isolated inference.
python wn2/verify_track.py --legacy-case-mode --phase forecast --keep-predictions \
  --package /absolute/submission --benchmark /absolute/evaluation-resources \
  --output /absolute/acceptance --budget-seconds 43200

# After the forecast phase succeeds, run in the verifier image on CPUs.
python wn2/verify_track.py --legacy-case-mode --phase score \
  --package /absolute/submission --benchmark /absolute/evaluation-resources \
  --output /absolute/acceptance --score-workers 4 --budget-seconds 129600
```

The split reserves at most 12 hours for forecasts and 36 hours for CPU scoring,
within the same 48-hour execution envelope; scheduler queue time is excluded.

Use the evaluator Python and WeatherBench-X environment for both launchers.
For host-side isolated inference, set `WEATHER_RUNTIME_ROOT` to an unpacked,
validated starter image; the GPU driver is mounted separately. For CPU scoring,
the verifier image supplies the model-side spectral transform and the compiled
TempestExtremes tracker. Keep paths consistent across mounts: the acceptance
output references the read-only observed-history resource directory.

The forecast phase produces `forecast-status.json`, not a reward or a complete
acceptance status. The score phase requires its complete replay receipts, runs
all four metric families and then produces `status.json` and `reward.json`.
It cannot accept arbitrary precomputed prediction stores as independent replay.
A failed phase retains its error record and logs; it is not a partial pass.

A one-member smoke check or a research-driver score table does not replace full
entry-point acceptance. Historical 32-date scripts are not the current verifier.

## Distributed CPU scoring

When a CPU cluster is available, `slurm/wn2_parallel_score.py` distributes the
same frozen scorer commands by date and metric family. Run it in the verifier
image with `--source`, `--benchmark`, `--output` and `--package` referring to the
same immutable source, resources, completed forecast output and submission.
It requires a successful `forecast-status.json` and unchanged replay receipts.

| Endpoint | Task indices | CPU per task | Initial concurrency |
| --- | --- | --- | --- |
| `core` | 0–7 (date) | 4 | 8 |
| `pooling` | 0–7 (date) | 8 | 4 |
| `spectra` | 0–7 (date) | 8 | 4 |
| `track` | 0–447 (date × 56 + member) | 2 | 16 |
| `cyclones` | 0–7, after tracking completes | 2 | 8 |

The first four arrays can overlap, using at most 128 CPUs. These starting
limits bound shared-storage contention and can be adjusted from measured
throughput. Scoring uses no GPUs. Each worker receives `--endpoint NAME` and
`--index INDEX`. The spectral worker uses the model Python environment on CPU;
the other workers use the evaluator Python environment.

After all eight core tasks succeed, `--endpoint core-summary` emits
`core-result.json` without declaring full acceptance. After all core, pooling,
spectral and cyclone tasks succeed, `--endpoint finalize` checks the complete
endpoint set and emits the final reward and status. Do not run the monolithic
score phase against the same output while these workers are active.
