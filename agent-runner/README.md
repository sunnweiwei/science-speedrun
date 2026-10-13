# agent-runner

`agent-runner` is a small execution layer for long-running model-development
tasks written in the native [Harbor task format](https://www.harborframework.com/docs/tasks).
It does not define another task schema and does not copy evaluator logic into the
runner.

A compatible model-development task declares `/workspace/submission` as a Harbor
artifact and uses `verifier.environment_mode = "separate"`. Everything inside the
submission directory is task-defined; for FoldBench, its `contract.md` requires
`infer.sh INPUT_DIR OUTPUT_DIR`.

The runner adds the lifecycle that model R&D needs:

1. start one detached Harbor agent session with no runner-imposed deadline;
2. expose `/workspace/submission` on the host while the agent is still running;
3. publish stable, immutable snapshots without asking the agent to manage a lock;
4. replay one selected snapshot in a new Harbor trial and the task's separate
   verifier environment.

The current implementation pins Harbor `0.22.0` and retains access to its 41
built-in agents, `acp:` registry agents, and any installed Harbor adapter exposed
as `module:Class`. Pass any of these through the same `--agent` option. For
example, a custom adapter can be selected with:

```bash
agent-runner start \
  --task TASK_DIR \
  --run-dir RUN_DIR \
  --agent package.module:CustomAgent \
  --model provider/model \
  --agent-kwarg 'config="/absolute/path/config.yaml"'
```

Custom harness packages must be installed in the same virtual environment as
the runner. The AI for Science embedded mini-swe implementation now lives in
[`agent-harness`](https://github.com/T0-RSI/agent-harness). The value
`--agent mini-swe-agent` continues to select Harbor's built-in adapter and is
different from that repository's custom audited implementation.

## Install

Python 3.12 and Docker Compose are required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Start a FoldBench development session

The large dataset is external to both repositories. A task-owned materializer can
populate it from a pinned Hugging Face dataset revision:

```bash
agent-runner materialize \
  --task ../ai4sci-tasks/foldbench \
  --data-dir /mnt/large-disk/foldbench-data \
  --env FOLDBENCH_HF_REPO=owner/dataset \
  --env FOLDBENCH_HF_REVISION=0123456789abcdef0123456789abcdef01234567
```

The task currently contains HF placeholders, so that command intentionally fails
until the repository and immutable commit are supplied.

`--data-dir` is passed through the environment variable named by
`--data-env-var` (`FOLDBENCH_DATA_DIR` by default). Other Harbor tasks can select
their own variable name; additional non-secret task variables use repeatable
`--env KEY=VALUE` on both `start` and `evaluate`.

Start Codex after securing the operator-owned authentication file:

```bash
chmod 600 /home/me/auth2.json

agent-runner start \
  --task ../ai4sci-tasks/foldbench \
  --run-dir /mnt/large-disk/agent-runs/foldbench-codex-001 \
  --data-dir /mnt/large-disk/foldbench-data \
  --agent codex \
  --model openai/gpt-5.6-sol \
  --auth-file /home/me/auth2.json \
  --agent-bin codex=/home/me/.local/bin/codex \
  --agent-bin codex-code-mode-host=/home/me/.local/lib/codex-code-mode-host \
  --network allowlist \
  --allow-host api.openai.com \
  --allow-host auth.openai.com \
  --allow-host chatgpt.com \
  --duration-hours 24 \
  --gpus task
```

`--network no-network` is the default. A remote CLI needs `allowlist` plus all
provider and first-run installation hosts it actually uses. `--network public` is
an explicit, warned escape hatch. Independent evaluation is offline by default; a trusted
harness-LLM compute profile retains the task's network topology so its credential-isolating gateway
can reach the selected provider. Authentication contents and `--env` values are inherited by the detached
worker but are not stored in `run.json` or mounted into the model workspace.

For `no-network`, the staged Compose files use `network_mode: none`; this does not
depend on Docker's default bridge. `allowlist` uses Harbor's nftables sidecar and
routes its kernel probe through Docker's `none` namespace, so a host without the
legacy `docker0` interface can still use per-trial Compose networking. Optional
`--agent-bin NAME=PATH` mounts an already installed executable read-only and
avoids granting package-manager egress merely to bootstrap an agent CLI. Current
standalone Codex packages need both `codex` and their matching
`codex-code-mode-host` when Harbor enables unified execution. The
allowlist provider also falls back to Docker's standard builder when the optional
`buildx` plugin is unavailable.

`--duration-hours` limits the entire detached session by wall clock. At the
deadline the worker gracefully stops Harbor, publishes a final stable snapshot,
and records `stop_reason: duration_elapsed` in `state.json`.

At worker startup, the runner writes the resolved wall-clock, resource, and harness LLM
limits to `RUN_DIR/campaign-budget.json`, mounts it read-only at
`/workspace/campaign-budget.json`, and appends a final instruction directing the
agent to read it before planning or training. The file reflects CLI overrides
and the selected compute profile; a `null` CPU, memory, or storage value means
that the runner did not set an explicit limit and the runtime default applies.

## Compute profiles

A task may bundle images and resource requests in `compute-profiles.toml`:

```toml
schema_version = "agent-runner-compute-profiles/v2"
default = "gpu"

[profiles.gpu]
gpus = "task"
cpus = 16
memory_mb = 65536
harness_llm_provider = "gemini"
harness_llm_model = "gemini-2.5-pro"
harness_llm_max_model_calls_per_case = 20
harness_llm_development_total_tokens = 20000000
harness_llm_max_output_tokens_per_call = 65536

[profiles.mac]
gpus = 0
development_image = "task-mac-cpu:latest"
verifier_image = "task-mac-cpu:latest"
```

When `--compute-profile` is omitted, the task's `default` profile is selected.
Tasks without this file retain their Harbor `task.toml` defaults. Select another
profile with `--compute-profile mac`. Explicit `--gpus`, `--cpus`,
`--memory-mb`, `--storage-mb`, `--development-image`, and `--verifier-image`
values override the selected profile. Independent evaluation reuses the resolved
profile recorded by `start`.

Harness-building tasks use schema v2 and must set the provider, model, and all three
`harness_llm_*` budget fields as one complete profile. The runner exports the resolved profile to
the gateway and stores it in `run.json`; provider credentials remain separate `--env` inputs and
are never persisted. Development applies the profile's campaign token total. `evaluate` reuses the
complete profile recorded by `start` by default, or accepts `--compute-profile` to replace the model
and limits together; evaluation has no cumulative token ceiling and retains the per-case call and
per-call output limits.

Harbor's local Docker provider does not translate task GPU metadata into Compose
device requests. The runner therefore stages a private task copy and adds the
requested NVIDIA device reservation to both development and verifier Compose
files. The source task remains unchanged.

## Observe, snapshot, stop

```bash
agent-runner list --runs-dir /mnt/large-disk/agent-runs
agent-runner list --runs-dir /mnt/large-disk/agent-runs --active-only
agent-runner status /mnt/large-disk/agent-runs/foldbench-codex-001
agent-runner snapshot /mnt/large-disk/agent-runs/foldbench-codex-001
agent-runner stop /mnt/large-disk/agent-runs/foldbench-codex-001
```

`list` scans nested run directories and combines durable state with the actual
worker PID. A session recorded as `starting` or `running` whose worker is gone is
reported as `stale`; a terminal session whose worker remains alive is reported
as `orphaned`. Use `--active-only` to show only live worker processes. Output is
JSON, like the other inspection commands.

Stopping a `stale` run discovers Docker Compose resources by the exact Harbor
trial-name label, removes only those containers, networks, and volumes, publishes
a final snapshot, and records `stop_reason: stale_worker_cleanup`. This also
handles a worker killed before Harbor could perform its normal teardown.

The live mutable directory is `RUN/live/submission`. A background watcher hashes
it twice, copies it, verifies the source and copy again, and only then atomically
updates `RUN/snapshots/latest.json`. An escaping symlink or a concurrent write is
rejected/retried. Each `RUN/snapshots/ID/submission` is immutable by convention
and content-addressed in its adjacent manifest.

`RUN/runner.log` is an immediately flushed JSON Lines lifecycle log. It records
worker and trial startup, the non-secret resolved runtime summary, agent start,
new snapshots, stop requests, failures, and the final state. Agent command output
remains in `RUN/harbor/TRIAL/agent/`; Harbor's own setup log remains in
`RUN/harbor/TRIAL/trial.log`. Follow runner events with:

```bash
tail -f RUN/runner.log
```

## Independent evaluation

```bash
agent-runner evaluate \
  /mnt/large-disk/agent-runs/foldbench-codex-001 \
  --snapshot latest
```

Evaluation verifies the snapshot checksum, mounts the snapshot read-only in a new
Harbor trial, copies it to the declared artifact path, and lets the task's own
separate verifier run. It verifies the checksum again afterwards. Results live in
`RUN/evaluations/SNAPSHOT_ID/evaluation.json` and the underlying Harbor trial
directory. The evaluator never receives the mutable development workspace.

## Run layout

```text
RUN/
  run.json                 non-secret resolved runner inputs
  state.json               current session state
  runner.log               detached worker log
  runtime-task/            staged Harbor task; source task is untouched
  live/submission/         mutable delivery directory visible to the agent
  snapshots/latest.json    atomic pointer
  snapshots/ID/
    manifest.json
    submission/            immutable evaluator input
  harbor/                  uniquely named development Harbor trial
  evaluations/             independent verifier trials
```

The runner does not choose a single scalar across multi-metric tasks. For
FoldBench, `evaluation.json` preserves the nine task-defined track rewards; an
experiment controller can decide which evaluated snapshot is "best" without
changing the artifact contract.

## Validation levels

`pytest` runs fast unit tests. The opt-in Docker smoke uses the real FoldBench
Harbor task, real development and separate-verifier images, one native fixture,
and a deliberately trivial replay submission. It proves runtime plumbing only;
the copied native is not a trained model or a benchmark result.

```bash
RUN_FOLDBENCH_SMOKE=1 \
FOLDBENCH_TASK_DIR=../ai4sci-tasks/foldbench \
FOLDBENCH_SMOKE_SUITE_DIR=/path/to/foldbench-full/suite \
FOLDBENCH_SMOKE_GROUND_TRUTH_DIR=/path/to/ground_truths \
FOLDBENCH_SMOKE_GPUS=8 \
pytest -m foldbench_smoke -q
```
