# Evaluating an agent or model on science-speedrun

This guide reproduces the whole evaluation on your own hardware: agent sessions on the 20 tasks, their scores,
and the overall number. Use it to test another agent or model.

A run is one agent session on one task. The agent gets the task's data, a reproduced baseline and a development
evaluation inside an offline container. For 48 hours it keeps improving a model package at `/workspace/submission`.
After the session, an independent verifier scores the last package on held-out data. Every version the agent
published is also kept, so the reward can be traced over time.

Everything here uses the runner in [`runner/`](runner/README.md). Verifiers, held-out labels and sealed seeds are
all in this repository or in public data releases; only natural-disease-progression needs credentialed data.
[CHANGES.md](CHANGES.md) records what changed recently.

## 1. What you need

- **GPUs.** The published results gave every session **one H100**, although 16 tasks declare 8× H100 in their
  `task.toml`. Run the same way to compare with them (section 4). An 8-GPU host then runs eight sessions in
  parallel. Scoring uses each task's verifier resources: eight GPUs for 8 tasks, one GPU for 11, none for
  tokamak-magnetic-control.
  Results on other GPU types are not comparable; if you have no choice, record it (section 6).
- **Time.** A session takes 48 h plus a five-minute submission window. The verifier runs afterwards, and its
  time limit ranges from 1 h (surface-code-decoding) to 72 h (crystal-structure-generation; see `[verifier]` in
  `task.toml`).
- **Docker** with the NVIDIA container toolkit, on the GPU nodes. A cluster works if its nodes give you Docker.
  Nodes with only Apptainer/Singularity (common on Slurm clusters) are not supported yet.
- **Disk.** Prepared data for all 20 tasks takes roughly 2–2.5 TB. The largest are fluid-flow-control (423 GB),
  genome-sequence-to-function (~308 GiB), timeseries-forecasting (275 GB), probabilistic-global-weather (206 GB)
  and molecular-pretraining-transfer (179 GiB). In addition, one session can write hundreds of GB under
  `research/`.
- **Model access** for the agent you test (section 2).
- **Data access.** The releases are public on Hugging Face, except for three tasks:

  | Task | What to request |
  | --- | --- |
  | natural-disease-progression | PhysioNet credentialing for MIMIC-IV, then access to the derived release (`ScienceRSI/natural-disease-progression`); its data must not be redistributed |
  | brain-computer-interface-decoding | access to the FACED/TUEV release (`AI4Sci-BCI/AI4BCI_private_dataset`) for your HF account; export its `HF_TOKEN` |
  | crystal-structure-generation | accept the terms of the gated `facebook/UMA` model; export the account's `HF_TOKEN` |

## 2. Install and check (about 15 minutes)

```bash
git clone https://github.com/sunnweiwei/science-speedrun && cd science-speedrun
docker build -f runner/Dockerfile.runner -t harbor-runner:0.23.0-bx runner
export EVAL_ROOT=/large/persistent/disk/speedrun     # trials, results, data env files
export DOCKER=docker                                  # the default is `sudo -n docker`
```

Then set up the agent:

- **Codex** (`AGENT_KIND=codex`): unpack the Codex **0.160.1** `x86_64-unknown-linux-musl` package from
  `https://github.com/openai/codex/releases/download/rust-v0.160.1/codex-package-x86_64-unknown-linux-musl.tar.gz`
  to a directory. Set `CODEX_PKG` to it and add it to `MOUNT_ROOTS`.
  Log in with a separate Codex home, so token refreshes cannot log out your other sessions:
  `CODEX_HOME=~/.codex-eval codex login --device-auth`. Copy its `auth.json` to `$EVAL_ROOT/.authdir/auth.json`
  (mode 644 inside a 700 directory).
- **Claude Code** (`AGENT_KIND=claude-code`): a `claude` binary on the host (or `CLAUDE_BIN`). Export
  `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) or `ANTHROPIC_API_KEY`.
- **Antigravity / Gemini** (`AGENT_KIND=antigravity`): install `agy` on the host, set `AGY_BIN` and add its
  directory to `MOUNT_ROOTS`, then export `GEMINI_API_KEY`.

Credentials are passed by name and never written into a task container's image or the results.

Check the installation; expect `{"reward": 1}`:

```bash
AGENT_KIND=codex runner/selftest.sh          # or claude-code / antigravity; ~2-3 min, no GPU
```

## 3. Prepare a task's data

```bash
runner/prepare.sh surface-code-decoding /data/surface-code    # downloads, verifies, writes $EVAL_ROOT/envs/<task>.env
export MOUNT_ROOTS="$MOUNT_ROOTS /data"                         # every data root outside EVAL_ROOT
```

Each task's `environment/data/MATERIALIZE.md` describes its release. Downloads are pinned to a commit and checked
file by file; a prepare step that fails must be fixed, not skipped.

## 4. Run a session with the standard settings

The published runs used these settings. Keep them, so results are comparable:

```bash
export HOURS=48 GRACE_SECONDS=300               # 48 h of research, then a 5-minute submission-only window
export GPUS=1 GPU_IDS=3                         # one H100 per session; a different GPU for each parallel session
export AGENT_CPUS=22 AGENT_MEMORY_MB=229376     # 22 CPUs, 224 GiB (see below for four tasks)
export VERIFY_GPU_IDS=                          # scoring may use any GPU; see Resources below

AGENT_KIND=codex runner/run.sh TASK             # your agent and model (section 5)
```

The published runs themselves were Codex with `CODEX_MODEL=openai/gpt-6-astra CODEX_REASONING=xhigh`, and
Claude Code 2.1.319 with `CLAUDE_MODEL=anthropic/claude-fable-5-2 CLAUDE_REASONING=xhigh`.

- **Resources.**
  - brain-computer-interface-decoding, natural-disease-progression, tokamak-magnetic-control and
    earth-observation-pretraining-transfer ran with 64 CPUs and 128 GiB: use `AGENT_CPUS=64 AGENT_MEMORY_MB=131072`
    for them.
  - Without `GPU_IDS`, every one-GPU session gets GPU 0.
  - Scoring runs when the session ends, with the task's verifier resources (up to eight GPUs), one verification
    at a time. It takes GPUs from `VERIFY_GPU_IDS`; empty means any GPU, and by default it uses `GPU_IDS`, which is
    too few for an eight-GPU verifier. Scoring on GPUs that other sessions still use slows both. On a busy host,
    give `VERIFY_GPU_IDS` free GPUs, or rescore later with `runner/run_verify.sh`.
- **Runs per pair.** One session per task and agent. If a session fails for an infrastructure reason (lost
  GPUs, a host crash, a full disk), tell us before rerunning it, and keep the failed run's results folder.
- **Progress.** Watch `$EVAL_ROOT/results/NAME/`: `trial.log`, `snapshot.log` and `snapshots/index.tsv`, which
  lists every package version the agent published. The agent's own notes are in `$EVAL_ROOT/research/NAME/`.
- **Stopping a run early.** `docker stop run-NAME`, then remove its `NAME__*` containers and network.
- **Reward over time.** After the run, `runner/score_snapshots.sh NAME 6` scores 6 evenly spaced snapshots plus
  the last, and writes `curve.tsv`. Each point is one full verifier run.

### Scores

- **Task score.** `reward` in `results/NAME/reward.json` is the task's score on the metric-V4 scale (what the
  published results report). `harness/ssi_index_v3.py` holds the same formulas; `harness/test_ssi_index_v4.py`
  checks them.
- **Overall score.** The equal-weight mean over the 20 tasks of the final reward, each bounded to [0, 1]. A few
  tasks can report a negative `signed_score`; it is drawn on single-task curves but never averaged.
- **Curves.** Read a task's `curve.tsv` as a step function: each value holds until the next scored point, and the
  value is 0 before the first published package. The overall curve is the mean of the 20 step functions at each
  hour.
- **Submission window.** A package changed in the window is scored apart (`reward-grace.json`). It is not part of
  the 48-hour result.

## 5. Testing a different model or agent

**Same agent, another model.** Set the model in `provider/model` form:

| Agent | Variable | Default |
| --- | --- | --- |
| Codex | `CODEX_MODEL`, `CODEX_REASONING` | `openai/gpt-6.1-sol`, `medium` (published runs: `openai/gpt-6-astra`, `xhigh`) |
| Claude Code | `CLAUDE_MODEL`, `CLAUDE_REASONING` | `anthropic/claude-opus-5-5`, Claude Code's default (published runs: `anthropic/claude-fable-5-2`, `xhigh`) |
| Antigravity | `AGY_MODEL` | `google/gemini-4-argon` |

The task network allows only the agent's model endpoint (`--allow-agent-host` in `runner/config.sh`). A model
served from another host needs that host added there.

**Another agent.** Harbor 0.23 ships many agent integrations (OpenHands, Aider, Goose, Gemini CLI, Qwen Code,
Kimi CLI, mini-SWE-agent, …). To use one:

1. Add `runner/harbor_ext/<agent>_continue.py`, modelled on `claude_continue.py`. Subclass Harbor's agent. Wrap
   its command in the resume loop: whenever the agent ends its turn, continue the same session with the
   continue prompt. Pass the script through `trial_options.with_grace`, so `GRACE_SECONDS` works.
2. Add a branch to `agent_args` and `agent_env` in `runner/config.sh`. It names the agent class and the model.
   It mounts the agent's binary read-only, because task containers have no network to install anything. It
   allows only the model endpoint and passes credentials by name.
3. Run `AGENT_KIND=<agent> runner/selftest.sh`. Expect reward 1 and several resumed turns in `trial.log`.

Whatever the agent, a valid run:
- runs non-interactively inside the task container;
- reaches no network except its model endpoint (no web search or browsing);
- is never told its time budget;
- is continued until the time runs out;
- gets no human help during the session.

## 6. What to send back

For each run, pack the results folder (`tar czf NAME.tgz -C $EVAL_ROOT/results NAME`). It holds:
- `reward.json` and `report.json`: the verifier's scores;
- `submission/`: the final package;
- `agent/`: the trajectory and session logs;
- `notes/`: the agent's research notes;
- `snapshots/`: every published version, with `index.tsv`;
- `grace-period.json`, plus `reward-grace.json` when the window changed the package;
- `curve.tsv` if you ran `score_snapshots.sh`.

If `snapshots/` is too large to send, keep it and send everything else; we may ask for specific versions.

Add a short run card:
- the agent and its CLI version;
- the model and reasoning effort;
- `HOURS` and `GRACE_SECONDS`;
- the GPU type and count, and the host;
- the start time (UTC);
- anything unusual: restarts, lost GPUs, quota errors, a task you could not prepare.

## 7. Known issues

- **fruit-fly-neuron-reconstruction on H100.** Its point-sampling library was rebuilt for H100, and `prepare.sh`
  installs it; data prepared before 2026-11-18 needs `prepare.sh` again. The rebuild matches the old library on
  an A6000 but has not yet run on an H100. If the SHAPE stage fails with "no kernel image is available", tell us.
- **Quota.** When a Codex weekly limit runs out, Codex silently spends paid credits. One agent-hour used about
  1.2 % of a Pro week.
- **GPUs lost mid-run.** A `systemctl daemon-reload` on the host can drop a container's GPU access. Avoid it while
  sessions run; the runner passes device nodes explicitly to survive it.
- **Disk.** Image builds fill Docker's disk (`docker builder prune` between batches). Agents may write hundreds of
  GB under `research/`.

More detail: [runner/README.md](runner/README.md) (setup, pitfalls), [TASK_STANDARD.md](TASK_STANDARD.md) (task
format), each task's `README.md` and `instruction.md`.
