# Validation records

Baseline scores and their raw evaluation records: `environment/baseline/*/RESULTS.md` and
`environment/baseline/evaluation_history.jsonl`.

## Layout check after TASK_STANDARD.md (2026-09-27)

`baselines/` became `environment/baseline/` (with `publish.py` for atomic publication and a
README/RESULTS per package), the research-record template moved to `environment/evaluation/`, the
baselines' evaluation rows to `environment/baseline/evaluation_history.jsonl`, and the operator-only
no-effect cache script to `environment/data/`. The release's `dev/` entries are mounted one by one
under `/workspace/data` (`TASK_DATA_DIR`), SE-600M is reachable at `/opt/se600m` through a link, the
verifier reads `TASK_DATA_DIR` and `TASK_PRIVATE_DIR`, and `/workspace/research` is an empty
writable mount. Checked on one host with rebuilt images and the task's own compose files:

- the development container started healthy with every `dev/` entry read-only under
  `/workspace/data` and `/opt/se600m` resolving to `/workspace/data/se600m`;
- the B1 table was built in `/workspace/research/b1` (18,301 targets × 19,111 genes, 1.4 GB) and
  `baseline/publish.py` published it into the image-layer `/workspace/submission` twice, the
  first time after `check_submission.py` passed on `ctx_far`, with no staging leftovers;
- `run_loco.py` retrained the table without jurkat and scored the jurkat fold at 0.153, the
  recorded value;
- the isolated verifier (`tests/docker-compose.yaml`, main + grader) replayed the published
  package (default `mode.json`, B1g) in 24 min: reward 0.10222838274873323, identical to the
  recorded B1g row (near 0.1497, mid 0.1179, far 0.0392);
- 7 unit tests pass in the verifier image (`tests/run_unit_tests.sh`), including two new ones for
  atomic publication and the development mounts.
