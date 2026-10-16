# runner

Runs one agent session on a task with Harbor, then scores what the agent published with the
task's own verifier. It is the tooling behind the agent-behavior tests of the tasks in `../tasks`.

| file | role |
| --- | --- |
| `run.sh TASK [NAME]` | one command: session, collection, independent score |
| `prepare.sh TASK DATA_ROOT` | downloads or builds a task's data, writes `$EVAL_ROOT/envs/TASK.env` |
| `selftest.sh` | install check on a 1-minute task (no GPU): expect reward 1 and a few resumed turns |
| `snapshot.sh` | started by `run.sh`: keeps every version of `/workspace/submission` the agent publishes |
| `score_snapshots.sh NAME [K]` | scores K snapshots spread over the run plus the last: reward over time in `curve.tsv` |
| `run_trial.sh`, `run_verify.sh` | the two halves of `run.sh` |
| `config.sh` | every setting, overridable from the environment |
| `Dockerfile.runner`, `Dockerfile.prepare` | the Harbor runner image and the data-preparation image |
| `harbor_ext/` | Harbor extensions: GPU-aware Docker environment, the continuing Codex, Claude Code and Antigravity agents, the copy agent |

## Setup

1. **Docker.** Install Docker with the NVIDIA container toolkit. The scripts call `sudo -n docker`; override with `DOCKER=docker`.
2. **Runner image.** `docker build -f runner/Dockerfile.runner -t harbor-runner:0.23.0-bx runner`
3. **`EVAL_ROOT`.** A large, **persistent** directory for trials, results and data env files. Local SSDs that are wiped when the VM stops are not persistent.
4. **Agent.** Pick one.
   - **Codex** (`AGENT_KIND=codex`, default):
     - Unpack the Codex 0.160.1 standalone `x86_64-unknown-linux-musl` package. It ships as `~/.codex/packages/standalone/releases/0.160.1-…` with the Codex installer.
     - Point `CODEX_PKG` at it and add that path to `MOUNT_ROOTS`.
     - Log in with a separate Codex home, so that token refreshes cannot log out another session: `CODEX_HOME=~/.codex-eval-auth codex login --device-auth`.
     - Copy its `auth.json` to `$EVAL_ROOT/.authdir/auth.json`, with mode 644 inside a 700 directory. Non-root task images cannot read a 600 file.
   - **Claude Code** (`AGENT_KIND=claude-code`):
     - Set `CLAUDE_BIN` to a host `claude` binary; the default is the one on `PATH`.
     - Authenticate with a subscription login by exporting `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` (default), or with `ANTHROPIC_API_KEY`.
     - The token is passed by name from your environment and is never written to disk.
   - **Antigravity CLI with Gemini** (`AGENT_KIND=antigravity`):
     - Install `agy` once on the host (`curl -fsSL https://antigravity.google/cli/install.sh | bash`), set `AGY_BIN` to it and add its directory to `MOUNT_ROOTS`. It is mounted read-only into the task container, which has no network to download it.
     - Export `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); it is passed by name and never written to disk. The model defaults to `AGY_MODEL=google/gemini-4-argon`.
     - Per-minute quota errors (429) on a shared Gemini project end a turn early; the resume loop waits before continuing.
5. **Data.** `runner/prepare.sh TASK /data/TASK` for each task you will run. Private Hugging Face repositories need `HF_TOKEN`.
   - Add every data root outside `EVAL_ROOT` to `MOUNT_ROOTS`.
   - natural-disease-progression is MIMIC-IV derived: its release is credentialed and its terms forbid redistribution.

Then `runner/selftest.sh`.

## Running

```bash
export EVAL_ROOT=/persistent/speedrun MOUNT_ROOTS="/data /opt/codex-0.160.1" CODEX_PKG=/opt/codex-0.160.1
AGENT_KIND=codex HOURS=24 runner/run.sh genome-sequence-to-function
AGENT_KIND=claude-code HOURS=48 runner/run.sh single-cell-perturbation-prediction
```

- **Resources.** By default a session gets the task's `task.toml` resources: GPUs, CPUs, memory. `GPUS=1 GPU_IDS=3 AGENT_CPUS=24 AGENT_MEMORY_MB=262144` runs a 1-GPU behavior test instead. Sessions on different GPUs can run in parallel. Verifications take a lock and run one at a time with the task's verifier resources.
- **Results** go to `$EVAL_ROOT/results/NAME/`:
  - `reward.json` (and `report.json` where the task writes one);
  - `submission/` (the published package);
  - `agent/` (trajectory and session logs);
  - `notes/` (the agent's markdown notes from `/workspace/research`);
  - `snapshots/` (every published version, with `index.tsv`).

  Copy them somewhere durable after each run.
- **Snapshots and curves.** Every `SNAPSHOT_INTERVAL` seconds (default 60), `snapshot.sh` hashes the package's file list and copies the package when it changed. It keeps the copy only if the hash is unchanged after copying, so a half-written package is never kept. `score_snapshots.sh NAME 6` then scores 6 evenly spaced snapshots plus the last with the task's verifier and writes `curve.tsv`: hours since start, then one column per reward key. Each point costs one verifier run.
- **Continuation.** Codex and Claude Code end their turn whenever the model answers without a tool call ("research remains active …"), even while training runs in the background. The continuing agents resume the same thread with a prompt that mentions no time budget, until Harbor's agent timeout (`HOURS`). Three resumes in a row that end within 120 s stop the loop. The tasks never tell the agent its budget.
- **Two-phase scoring.** Harbor applies resource overrides to the separate verifier too, and a busy agent host should not also grade. So the session's verifier is deferred (the trial then records "verification deferred"), and `run_verify.sh` scores the collected package in a fresh trial with the task's verifier resources.

## Pitfalls

- **Quota.** When a Codex weekly limit is used up, Codex silently spends paid credits. Check `rate_limits` in a rollout's `token_count` events before a large batch. It was about 1.2 % of a Pro week per agent-hour.
- **Disk.**
  - Image builds fill Docker's disk: run `docker builder prune` between batches.
  - Agents may write hundreds of GB under `research/`, for example decompressed copies of training data. Cap a run with a loop-mounted filesystem on a shared disk.
- **Rollout logs.** `codex exec --json` stdout can stop updating after a few hours; the rollout jsonl under `agent/sessions/` is authoritative.
- **Host networking.** If containers on the default Docker bridge have no outbound access (task image builds fail in `git fetch` or `pip install`), set `HARBOR_BUILD_NETWORK=host`: task images then build on the host network.
- **GPUs lost mid-run.** A `systemctl daemon-reload` on the host resets the device rules of running containers that got their GPUs only through a reservation ("Failed to initialize NVML: Unknown Error"). The runner therefore also passes the GPU device nodes explicitly, which survives the reload.
- **Stopping a run.** `docker stop run-NAME`, then remove its `NAME__*` containers and network by name.
