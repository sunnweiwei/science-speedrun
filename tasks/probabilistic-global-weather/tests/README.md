# Independent WN2 verifier

Harbor transfers `/workspace/submission` to this separate environment and runs
`bash /tests/test.sh`. The entry point executes the canonical WN2 verifier,
relocates each model and supplies fresh IDs and context-only inputs to an offline
Bubblewrap process. Scoring starts after that process exits. The test truth is
mounted only in the outer trusted scoring environment, not in model inference.

`/opt/weather-python` holds the pinned model runtime; `/opt/wn2-eval` holds the
separate NumPy/xarray/WeatherBench-X CPU evaluator. The default runs all eight cases
at once (one per GPU, eight GPUs) in consecutive windows within a 2 TB allowance, resumes
saved model state, and scores every endpoint with one process per lead. Forecast arrays are deleted only after window scoring and
cross-window precipitation/cyclone capture. Scores, tracks, logs and receipts
remain. Set `WEATHER_EVALUATION_DISK_ALLOWANCE_GB`,
`WEATHER_EVALUATION_DISK_RESERVE_GB`, `WEATHER_EVALUATION_CONCURRENT_CASES` and
`WEATHER_EVALUATION_SCORE_WORKERS` to change planning. See
[windowed evaluation](../wn2/WINDOWED_EVALUATION.md) for accounting, package upgrades
without training, and legacy opt-out requirements.
Invalid/incomplete output fails explicitly;
`/logs/verifier/reward.json` contains a numeric reward. Detailed case/status files
remain under `/logs/verifier/replay/replay-XXX`. A zero after setup failure is not measured
forecast quality. Replay directories must be new.

Both Dockerfiles build from the task root, as specified by the native Compose
files. Set `TASK_PRIVATE_DIR` to the [prepared layout](../environment/data/MATERIALIZE.md).
The verifier Compose configuration permits namespace syscalls in the trusted
outer container; the inner model namespace drops all capabilities. Actual Docker
execution remains unverified. On this HPC host, validated inference uses the
unpacked model image and host Bubblewrap; nested Apptainer checks also require
its root mount to be made private in a fresh user/mount namespace and `/dev/full`.

The canonical isolation implementation is `environment/weathernext/wn2/namespace_inference.py`, copied
into the verifier image under `/tests/wn2/`. The standalone invocation is in
[RUNNING.md](../docs/RUNNING.md).
