# Task standard

Every task in this repository follows the same shape, so any agent that runs inside a Harbor
container can pick it up, and any operator or developer can run, monitor and review it the same
way. The format is native [Harbor](https://www.harborframework.com/docs/tasks); this document fixes
the conventions on top of it.

## How a task runs

1. **Prepare data (controller).** An external controller runs the task's
   `environment/data/materialize.sh` once per host. It downloads pinned revisions, verifies
   checksums, and writes two directories on a large disk:
   - the development data the agent may see;
   - the private evaluation data, which only the verifier sees.
2. **Start the workspace (Harbor).** Harbor builds `environment/`. It mounts the development data
   read-only and a fresh, empty, writable research directory into the container. It gives the
   agent `instruction.md` as its only prompt.
3. **Develop (agent).** The agent works inside `/workspace` without network access (only its
   own model API is reachable). It keeps a complete package that follows the contract at
   `/workspace/submission`, and replaces it whenever it has a better one. There is no end point
   and no final answer.
4. **Evaluate (controller, invisible to the agent).** The controller snapshots
   `/workspace/submission` whenever it likes and scores the snapshot with `tests/` in a separate,
   offline verifier container. The development budget (for example 8 GPUs for 48 hours) is part
   of this external setup. It lives in `task.toml`, never in the instruction.

## What the agent sees

```text
/workspace/
  instruction.md     the task and its goal (the same text as the prompt)
  contract.md        the submission package: files, entry point, input/output formats, inference limits
  data/              README.md and small task files (splits, manifests) from the image;
                     each materialized dataset mounted read-only next to them
  baseline/<name>/   verified reference implementations: code that runs in this workspace and publishes a
                     contract-conforming package, a README with the commands, and RESULTS.md with the
                     numbers we reproduced in this setup (weights only where the task allows them);
                     reproduction/ holds that run's record: training logs, validation scores and,
                     where small and permitted, validation predictions (never test predictions)
  references/        further material we have not verified in this setup: surveys of released models,
                     other architectures and code, papers
  evaluation/        development evaluation, the same scoring code as the verifier
  submission/        the current submission (empty at start)
  research/          writable scratch for derived data, checkpoints and notes (empty at start, large disk)
```

- Everything except the mounted datasets is copied into the image. It is the agent's own copy,
  free to change, and changes never reach the repository or other runs.
- Mounted datasets are read-only. The agent may read and transform them, but writes its results
  under `research/`.
- Test inputs with labels, private data and verifier code are never in this container.
- There is no network. Harbor's egress control lets only the agent's model API through.
- The number of GPUs is whatever the container exposes. The instruction does not state it.

## Repository layout of a task

```text
<task>/
  README.md               for developers (template below); the agent does not see it
  task.toml               Harbor configuration
  instruction.md          the prompt
  environment/            everything that builds the agent's container
    Dockerfile            runtime plus the COPY of every agent-visible file
    docker-compose.yaml   mounts: ${TASK_DATA_DIR}/<entry> -> /workspace/data/<entry> (read-only),
                          ${TASK_RESEARCH_DIR} -> /workspace/research
    contract.md
    data/                 README.md and small task files (copied); materialize.sh and build
                          scripts (controller-side, not copied)
    baseline/<name>/      README.md, RESULTS.md, code, reproduction/ (the record of our run)
    references/
    evaluation/
  tests/                  the verifier: Dockerfile, docker-compose.yaml (private data from
                          ${TASK_PRIVATE_DIR}), test.sh, grading code, unit tests
  docs/                   development and acceptance records, run notes, licenses of upstream parts
  solution/               optional
```

## instruction.md

Five parts, in this order:

1. **Task and goal**: the model to develop, why it matters, and the metric it is judged by.
2. **Workspace**: one line for each directory above.
3. **Rules**: allowed data, pretrained assets and methods; test data are off limits; no network.
4. **Submission**: keep a complete package that follows `contract.md` at `/workspace/submission`
   from early on. Replace it atomically (stage elsewhere, then swap, as the baseline's publish
   script does) whenever a better model exists.
5. **Continuous research**: there is no end point. Keep improving the model and the submission,
   and never stop to hand in a final answer. State that the baselines are already reproduced
   (`RESULTS.md`, `reproduction/`) and that the agent should check its own pipeline with a short
   run rather than retrain a baseline to confirm its numbers.

Leave out time budgets, GPU counts, the verifier's schedule and runner or operator details.
Inference limits that bind the package (for example one GPU, four hours) belong in `contract.md`.

## task.toml

```toml
schema_version = "1.3"
artifacts = ["/workspace/submission"]

[task]
name = "ai4sci/<task-directory>"
version = "x.y.z"
description = "One sentence: the model the agent develops."
authors = [{ name = "..." }]
keywords = ["model-development", "<domain>"]

[metadata]
data_status = "..."

[agent]
timeout_sec = 172800.0          # external development budget; the agent is not told

[environment]
network_mode = "no-network"     # never network_mode in docker-compose.yaml
cpus = 32
memory_mb = 262144
storage_mb = 409600
gpus = 8
gpu_types = ["H100"]

[verifier]
environment_mode = "separate"
timeout_sec = 21600.0

[verifier.environment]
network_mode = "no-network"
gpus = 1
gpu_types = ["H100"]
```

Compose files do not set `network_mode`, `gpus` or device reservations for the agent's
container. `task.toml` is the single source for network and GPUs.

## Operator environment variables

| Variable | Meaning |
| --- | --- |
| `TASK_DATA_DIR` | materialized development data (read-only for the agent) |
| `TASK_PRIVATE_DIR` | materialized private evaluation data (verifier only) |
| `TASK_RESEARCH_DIR` | an empty directory per run on a large disk, mounted writable at `/workspace/research` |

Task-specific extras (model-gateway settings, scorer assets) are listed in the task README.

## Verifier output

`tests/test.sh` writes `/logs/verifier/reward.json`. It contains `reward`, a number in [0, 1] and
higher is better, next to every raw metric. The exceptions are tasks that deliberately report
several metrics without an aggregate, because no weighted mean of them would be a measured
quantity (hydrogym, protein-design, atomistic-materials). Invalid
packages score 0. Infrastructure failures produce no reward.

## README.md (developers)

Same sections in every task:

1. **Overview**: task, goal, metric, status.
2. **Data**: sources, licenses, pinned revisions, sizes, the `materialize.sh` commands and the
   resulting layout.
3. **Workspace**: what the agent receives, each baseline with its reproduced result.
4. **Evaluation**: verifier, isolation, reward definition, anchors.
5. **Running**: Harbor command, environment variables, resources (the external budget).
6. **Results and validation**: reference numbers and who reproduced them where; details in `docs/`.
7. **Limitations**.
8. **Citation and licenses**.

## Checking a task against this document

Review the instruction's five sections and the README's eight. Check that every
`/workspace/...` path the instruction names is copied or mounted, each
`baseline/<name>/` has a README and RESULTS, and a task with a materializer includes
`MATERIALIZE.md`.

Check the required files, `task.toml` (`schema_version`, `artifacts`, authors,
`network_mode` on both sides), the agent's compose service (no `network_mode` or GPUs,
the `TASK_*` variables, a research mount), and `tests/`. Confirm that the instruction
states no development budget. Record the review and any deviations in the PR.

A separate grader sidecar, a separate verifier image, and a single-container verifier
are all in use; this standard does not prefer one.
