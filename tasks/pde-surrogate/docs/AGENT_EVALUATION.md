# Agent evaluation protocol

An agent pilot is out of scope for this contribution (agreed with the
maintainers; the contributor machine has no Docker, which `agent-runner`
requires). This document fixes how one is run and what is recorded, so the
first pilot is comparable to later ones.

## Run

```bash
# agent host data: the release's train/ subtree plus the upstream training pools (RUNNING.md step 1)
agent-runner start --task pde-surrogate --run-dir /large-disk/agent-runs/pde-001 \
  --data-dir /large-disk/pde-surrogate-agent --data-env-var TASK_DATA_DIR \
  --agent <adapter> --model <provider/model> --network no-network --duration-hours 48 --gpus task
# verifier data: the files the held-out cases reference (materialize.sh DEST, or --from-release)
TASK_PRIVATE_DIR=/large-disk/pde-surrogate agent-runner evaluate --task pde-surrogate --run-dir /large-disk/agent-runs/pde-001 --snapshot <id>
```

Each entry of the agent-side directory is mounted read-only under `/workspace/data`, and must not hold a
held-out trajectory (`environment/entrypoint.sh` refuses one). The verifier's directory holds the files
referenced by `split.json`. The agent sees the standard workspace (TASK_STANDARD.md):
`/workspace/{instruction.md, contract.md, data/, baseline/, references/, evaluation/, research/}`.
It writes `/workspace/submission`.

## Record

For every pilot, commit under `results/<agent>-<date>/`: `result.json`
(`reward.json`, `status.json` and the per-group table from `report.json`),
`model_manifest.json`, the submission's `README.md` and `EXPERIMENTS.md`,
agent, model, duration, GPU type, snapshot id, and the runner's
`campaign-budget.json`. Never commit weights.

## Interpretation

A pilot demonstrates the workflow. It is a matched comparison only when agent,
model, budget, hardware and number of attempts are fixed in advance and
identical across runs. Scores are compared with `docs/baseline_score.json`
(task starter, from-scratch references and external targets) and the normalised
score in `reward.json`, never with validation numbers reported by the agent.
