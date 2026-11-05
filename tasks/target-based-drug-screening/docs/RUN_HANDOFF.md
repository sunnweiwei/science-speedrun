# Shared Harbor lifecycle

Use the shared runner in the [task README](../README.md#5-running), just as
for the other tasks. The old task-local `controller/` and its reference driver
were removed on 2026-11-05. They duplicated orchestration already owned by
the repository runner and are not needed by the model or verifier.

1. `prepare.sh` downloads the pinned release and stages separate development
   and private data views. Only development data enter the agent workspace.
2. `run.sh` starts the Harbor session. The shared runner chooses the agent,
   credentials and external budget, and resumes research after ordinary
   agent turn endings. The task prompt does not specify a time budget.
3. `snapshot.sh` saves published packages when they change, checking that
   the package stayed unchanged during each copy.
4. After the session, `run.sh` collects the submission and trajectory and
   invokes `run_verify.sh` in a separate offline verifier environment.
   That verifier alone receives private test data.
5. `score_snapshots.sh` can replay selected saved packages to make a curve.
   Each point requires a separate verifier run.

The runner stores results under `$EVAL_ROOT/results/<run-name>/`. Preserve
`submission/`, `agent/`, `snapshots/`, `reward.json`, `report.json` and the
associated logs. Additional provenance requirements are described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For a behavior-only session with scoring deferred, use the shared
`run_trial.sh` and `snapshot.sh`; score the saved package later with
`run_verify.sh`. A completed agent session alone is not evidence of a valid
held-out score. Inspect its trajectory, package and verifier outputs.

The task still owns data materialization, model training/publication and the
independent verifier. Operator-side data preparation files under
`environment/` are not a second agent-session runner and are not copied into
the agent image. Agent authentication belongs only to the external runner.
