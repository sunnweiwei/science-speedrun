# Verifier

Two containers built from `tests/Dockerfile` (same runtime as the dev image):

- **main** — mounts the public test inputs (`$TASK_DATA_DIR/test`) at `/benchmark` and SE-600M at
  `/opt/se600m`, copies nothing else; runs `bash /workspace/submission/infer.sh` once per context
  as uid 10001 with no network and one GPU; validates each `predictions.h5ad` (gene order, cell
  counts, finiteness); seals predictions + hashes into the `sealed` volume; waits for the grader.
- **grader** — mounts the same public inputs and the private responses (`$TASK_PRIVATE_DIR`);
  checks provenance hashes; scores every context with `cellscore.py` (cell-eval 0.8.2) on the
  leaderboard and final subsets and by stratum; writes `reward.json`, `report.json`,
  `agent_feedback.json` into the sealed results, which main copies to `/logs/verifier`.

`reward.json`: `reward` (mean final-subset score over contexts), `leaderboard_reward`, and
per-context `_final` / `_leaderboard` values. A context whose inference fails scores 0 without
hiding the others; scorer failures raise and leave no reward.

## Running locally

```
export TASK_DATA_DIR=/path/to/DATA_ROOT/dev TASK_PRIVATE_DIR=/path/to/DATA_ROOT/private
docker compose -f tests/docker-compose.yaml -f local-override.yaml up --abort-on-container-exit
```
where `local-override.yaml` mounts the package at `/workspace/submission` in `main`. Unit tests:
`bash tests/run_unit_tests.sh vcz-verifier:latest`.

`agent_feedback.json` holds the leaderboard-subset scores only. Under TASK_STANDARD.md the
controller scores snapshots of `/workspace/submission` without telling the agent, so nothing is
returned by default; a runner that does return feedback should return only this file.
`report.json` and the final-subset scores stay with the operator.
