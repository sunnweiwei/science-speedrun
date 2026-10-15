# Verifier

| file | role |
|---|---|
| `docker-compose.yaml` | `main` (GPUs, no network): fine-tune data from `TASK_DATA_DIR/public/downstream`, test molecules without labels from `TASK_PRIVATE_DIR/verifier/test-inputs`. `grader` (no GPU, no network): labels from `TASK_PRIVATE_DIR/private/labels`. The two share the `sealed` volume. |
| `test.sh` → `main_verifier.py` | validates and freezes `/workspace/submission`, then runs `protocol/check_encoder.py` and the 24 fine-tunes (`protocol/run_suite.py`) as the unprivileged `agent` user. It seals the predictions (hashes + READY) or writes `valid: 0` and INVALID, then copies the grader's `reward.json` / `report.json` to `/logs/verifier`. |
| `grader.py` | checks the snapshot and scores each task against the labels (`protocol/tasks.py`); never runs package code |
| `protocol/` | byte-identical to `environment/evaluation/protocol/` (checked by `test_task.py`) |
| `test_task.py`, `run_unit_tests.sh` | unit tests (standard library) |

A replay outside Harbor needs an override that mounts a package at `/workspace/submission` and a host directory at
`/logs/verifier`; `docs/VALIDATION.md` records the acceptance replays.
