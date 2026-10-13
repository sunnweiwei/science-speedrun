# Evaluating an agent on this task

Version 1.3.0 supplies prepared HF data and a native trainable research path
in addition to the earlier interface starter. The scientific metrics and the 17,556-query test
denominator are unchanged. Use this release for a controlled pilot after the
deployment checks below; production Docker/H100 acceptance and a matched-budget
scratch reference must be recorded before reporting a fully commissioned track.

## What every agent receives

| Resource | Location / policy |
| --- | --- |
| Task prompt and ranking contract | `instruction.md`, `/workspace/contract.md` |
| Experimental training / validation | `/workspace/data/train.jsonl`, `/workspace/data/validation.jsonl`, validation candidate pools |
| Optional synthetic training data | `/workspace/data/synthetic_augmentation`, 2,333,485 held-out-clean ICEBERG spectra |
| MAGMa fragment supervision | `/workspace/data/glacier`, train and validation only; manifest includes exclusions |
| Simple runnable baseline | `/workspace/baseline/starter`, Fourier-DeepSets, random initialization |
| Native research starter | `/workspace/baseline/starter/glacier`, full GLACIER architecture and all-condition fallback |
| Native source and libraries | `/opt/ms-pred`, `/opt/glacier/bin/python`; identical verifier runtime |
| Fast scientific feedback | `/workspace/evaluation`: subset/full validation, condition strata, paired molecule bootstrap |
| Operator reference description | `/workspace/references`; documentation and adapters, without reference weights/predictions |
| Writable research / final package | `/workspace/research`, `/workspace/submission` |

No reference checkpoint, test labels, test predictions, private label directory,
upstream repository checkout containing released data/weights, or organizer run
directory is mounted into development. The GLACIER source archive contains only
the pinned source, package metadata and license. Agents may change architecture,
loss, training stages, condition coverage, augmentation and ranking, and may
copy/edit the provided source. They may not import unauthorized pretrained assets.

The 49.391% official-weight hybrid is an external frontier comparison. The
task-owned scratch GLACIER + DeepSets baseline now has a complete full-test
result: **40.322397% Hit@1**, with 17,556 valid rankings and MCES@1 13.169600.
Its full validation Hit@1 is 35.987441%. The current repaired DeepSets alone
scores 4.004329%; the earlier 5.126% belongs to a different historical run.
These commissioning results use heterogeneous hardware and do not establish
a matched-H100 reference. See the [result and SOTA audit](environment/references/SOTA_AUDIT.md)
and [aggregate evidence](environment/references/task_trained_baseline.json).

For inference, an independent [PR 27 reviewer replay](https://github.com/T0-RSI/ai4sci-tasks/pull/27#issuecomment-5750791449)
reported 3.4–3.9 hours for all 17,556 queries on one H100 at native batch 64,
versus about five hours at batch 8. New native publications use inference batch
64 and two workers; the training batch default remains 8. Use the publisher's
positive `--inference-batch-size` override for less GPU memory and remeasure
runtime. Current native Docker/HF acceptance and matched-budget training remain
separate from this reviewer timing.

## Prepare once, before the agent clock

Build both images from the task-root build context. Download pinned dependencies
and materialize public resources before research; during research, permit only
the campaign's agent model API endpoints through the runner's allowlist.
Do not count shared image, data or agent-tool preparation as agent research time.
The default prepared-data path needs an operator Python environment with
`huggingface_hub==1.32.0`, authenticated read access to the private grader
dataset, and sufficient disk space as documented in
[`environment/data/README.md`](environment/data/README.md). It verifies pinned
payloads without running source preprocessing or installing a host native
runtime. Both Docker images build the pinned native runtime for model execution.

```bash
export TASK_DATA_DIR=/absolute/public-root
export TASK_PRIVATE_DIR=/absolute/private-root
export MASSSPECGYM_DEVELOPMENT_IMAGE=massspecgym-development:campaign-pinned
export MASSSPECGYM_VERIFIER_IMAGE=massspecgym-verifier:campaign-pinned
bash mass-spectrometry-molecule-retrieval/environment/data/materialize.sh \
  "$TASK_DATA_DIR" "$TASK_PRIVATE_DIR"
python3 mass-spectrometry-molecule-retrieval/environment/data/check_resources.py \
  --data-root "$TASK_DATA_DIR/development" --hashes
docker build -f mass-spectrometry-molecule-retrieval/environment/Dockerfile \
  -t "$MASSSPECGYM_DEVELOPMENT_IMAGE" mass-spectrometry-molecule-retrieval
docker build -f mass-spectrometry-molecule-retrieval/tests/Dockerfile \
  -t "$MASSSPECGYM_VERIFIER_IMAGE" mass-spectrometry-molecule-retrieval
```

An explicit `MASSSPECGYM_MATERIALIZE_MODE=source` rebuild additionally needs
`uv==0.9.14`, the `hf` CLI, the documented operator RDKit/NumPy environment and
the pinned native runtime. Install the latter using
`environment/runtime/install.sh`; set `MASSSPECGYM_RUNTIME_ROOT` to a writable
absolute directory if `/opt` is unavailable, then set
`MASSSPECGYM_GLACIER_PYTHON=ROOT/glacier/bin/python` and
`MASSSPECGYM_GLACIER_SOURCE=ROOT/ms-pred`. The `hf` CLI and the core materializer's
dependencies are also required, as documented in `environment/data/README.md`.
For source rebuilding, do not set the two preparation-skip flags for a scored
run. A subset MAGMa manifest fails the full-task resource check. Preparation resumes completed
SQLite rows and rejects changed inputs/settings.

Before a campaign, archive the task commit and working-tree diff, image IDs,
data manifests, native source revision, prompt, harness version, model revision,
agent settings and resource configuration. Do not print credentials into this
archive. A dirty working tree must be explicitly included in the release snapshot.

## Freeze the run protocol

The default task allows 48 hours, up to 8 H100s, 32 CPU cores, 256 GiB RAM and
200 GiB writable storage. These are ceilings, not an obligation to use all GPUs.
The verifier's model-execution container has 1 H100, 16 CPUs, 64 GiB RAM and
50 GiB storage. The trusted CPU-only grader has a separate ceiling of 16 CPUs
and 16 GiB RAM, and no GPU. The complete replay has 6 hours: at most 5 hours
for inference, with setup, sealing and grading sharing the remaining time.
The prompt explicitly
states the budget. Any experiment hiding the budget needs a separately recorded
prompt condition applied equally to all compared agents.

Use the existing generic `agent-runner`, as the clinical tasks do. The inspected
revision `e75f5f212ec0e678bee1a6331c859d0b45b204a4` stages NVIDIA device requests
and disables Harbor's separate GPU allocator. Native Harbor at inspected
revision `6af8d6e31eced13b93849cdf80feeadf24603d15` alone advertises `gpus=False`;
launch through the runner's integration. Do not add a task-specific reasoning
agent, model gateway or research scheduler.

Development Compose deliberately leaves networking unspecified. `task.toml`
defaults to `no-network`; `agent-runner start --network allowlist` overrides the
staged development and agent policies, and Harbor attaches its egress sidecar.
Do not add `network_mode` or `networks` to development Compose overrides: Harbor
excludes explicitly networked services from sidecar routing. Do not launch the
development service with bare `docker compose up`, which does not read the TOML
policy or install the allowlist. Both final verifier services retain explicit
`network_mode: none`, and the verifier TOML policy remains `no-network`.

Prepare the agent executable and its dependencies before starting the clock.
For the existing Codex adapter, mount the operator's pinned Linux `codex` binary
and matching `codex-code-mode-host` binary using the runner's `--agent-bin`
interface. Verify that they run in the prepared development image; the adapter
skips its network installer when a satisfactory Codex executable is present.
Do not bake a particular agent or authentication into the shared task image.

The following uses the same Codex transport pattern as the clinical runner
script. Set `MASSSPECGYM_AGENT_MODEL`, `MASSSPECGYM_AUTH_FILE`,
`MASSSPECGYM_CODEX_BIN`, `MASSSPECGYM_CODE_MODE_HOST_BIN` and
`MASSSPECGYM_RUN_DIR` to the campaign's selected model, private host auth file,
prepared executable paths and fresh absolute run directory. Use the image tags
built above and archive their immutable image IDs. This is a source-checked
launch recipe; live Docker/API transport acceptance remains to be measured.

```bash
agent-runner start \
  --task /absolute/ai4sci-tasks/massspecgym-retrieval \
  --run-dir "$MASSSPECGYM_RUN_DIR" \
  --data-dir "$TASK_DATA_DIR" --data-env-var TASK_DATA_DIR \
  --development-image "$MASSSPECGYM_DEVELOPMENT_IMAGE" \
  --verifier-image "$MASSSPECGYM_VERIFIER_IMAGE" \
  --agent codex --model "$MASSSPECGYM_AGENT_MODEL" \
  --auth-file "$MASSSPECGYM_AUTH_FILE" \
  --agent-bin "codex=$MASSSPECGYM_CODEX_BIN" \
  --agent-bin "codex-code-mode-host=$MASSSPECGYM_CODE_MODE_HOST_BIN" \
  --network allowlist \
  --allow-host api.openai.com --allow-host auth.openai.com --allow-host chatgpt.com \
  --duration-hours 48 --gpus task
```

Authentication endpoints depend on the selected adapter/auth mode; freeze the
smallest working host list and the campaign's API/token cost cap before runs.
Use the same policy for compared agents. Do not add package registries, model
hubs, public dataset hosts or arbitrary browsing access during research. The
API exception supports agent reasoning/coding, not an online prediction model,
external training labels or teacher predictions. A host allowlist is a network
boundary, not proof that every API request follows this research policy; retain
the agent trajectory for audit. On other hardware, record a pilot override
rather than claiming the default H100 score.

Keep main CPU/RAM/GPU requests in `task.toml` and the selected runner profile.
Hard-coding those requests in the base Compose files overrides runner resource
settings and leaves GPU requests active in a CPU profile. The manual verifier
override `tests/docker-compose.local.yaml` explicitly requests one GPU and the
declared inference CPU/RAM limits. The trusted grader retains its own fixed
CPU/RAM ceiling. Independent runner evaluation must pass **`--gpus 1`**:
otherwise the runner inherits the development allocation and requests 8 GPUs
for replay. For a started run, the corresponding command is:

```bash
agent-runner evaluate "$MASSSPECGYM_RUN_DIR" --snapshot latest --gpus 1 \
  --env "TASK_PRIVATE_DIR=$TASK_PRIVATE_DIR"
```

The installed mini-swe-agent adapter in `agent-harness` revision `bde0118`
also needs container-side model API access, and runs `uv tool install` during
setup. Preinstalling an executable alone does not skip that adapter's install
step. A campaign choosing it must prepare and verify its complete offline
installation/cache path before timed research; the Codex recipe above does not
certify mini-swe-agent setup. Do not open download hosts throughout research to
work around setup. The recorded CPU usability
pilot uses an outer Codex CLI model connection and an offline inner shell
sandbox; it is separate evidence from Harbor deployment acceptance.

Run `tests/check_runner_network.py` with the actual runner/Harbor environment
before a campaign (see `tests/README.md`). It stages allowlist and no-network
profiles and checks Harbor's actual service selection, including a regression
control for the old Compose conflict. This verifies configuration compatibility;
the live container check must still confirm that authorized API calls succeed,
non-allowlisted destinations fail, and both verifier services remain offline.

A host with NVIDIA Container Toolkit is required for manual Docker Compose
replay. Compose does not select H100 by the declared name or enforce writable
storage quotas; deployment must check GPU models, enforce those quotas, and
archive its resolved configuration. Shared read-only data and image storage are
operator preparation resources, outside the writable agent quota. The optional
`compute-profiles.toml` is consumed by agent-runner, not by native Harbor.

Run each attempt from a clean environment. Give all agents the same assets,
prompt, compute, API policy and feedback. The harness should archive the full
conversation and terminal/command trace; preserve failed attempts as well as
successful ones. Environment setup failures can be repaired and rerun only
under a recorded rule applied consistently. Training failures or missing
submissions caused by the agent are task outcomes, not silently retried trials.

## Completion and grading

The agent establishes a working package early, keeps it runnable while trying
changes, selects on validation, and leaves `EXPERIMENTS.md` plus selected
training source/configuration/logs in the package. A resumed run consumes its
remaining cumulative budget. The harness freezes and relocates the complete
directory, including `.versions`; the live trainer and development data are
not available in replay.

The untrusted replay sees public input only. It runs as UID 10001 with empty
supplementary groups and `no_new_privs`. The separate trusted grader receives
sealed rankings and private labels but never loads agent code or model weights.
The raw fixed-denominator Hit@1 is the primary score. Report submission validity,
Hit@5/20, MCES@1 and actual resource consumption separately; do not hide a failed
attempt by averaging only successful runs or by inventing a weighted score.

## Acceptance and reporting

Before a full campaign, require: a fresh image build; complete checked data;
real train/resume; relocated input-only replay including both routing branches;
rejection of malformed output and changed checkpoint hashes; a completed full
single-H100 replay within the declared limits; and one cold-start agent pilot.
The pilot tests whether the prompt and supplied resources let an agent conduct
and submit an experiment without organizer help. It does not establish a
reliable ordering between agents. Compare agents over multiple independent,
identically budgeted attempts and report between-attempt variability.

Use a result table with distinct comparison types:

| Run | Initialization / method | Hardware and measured cost | Validation Hit@1 | Test Hit@1 | Valid relocated submission |
| --- | --- | --- | --- | --- | --- |
| External frontier | Official GLACIER + independently trained clean MIST fallback | See frozen reference ledger | — | 49.391% | Operator replay; acceptance scope recorded separately |
| Matched scratch reference | Exact released recipe, seed and budget | Must be measured | Pending | Pending | Pending |
| Agent attempt | Model/harness revision, seed, selected experiment | Measured wall/GPU/API cost | Measured | Trusted grader | Yes/no and failure reason |

Archive the checkpoint and manifest hashes, public input manifest hash, sealed
ranking hash, grader report, experiment history and agent conversation location
with this table. Test results are final assessment, not a feedback loop for
further method selection. `BENCHMARK_READINESS.md` and
`release_validation.json` record the evidence and remaining deployment gates.

The 2026-09-08 commissioning run now includes a completed cold CPU agent
workflow and full input-only replay through the real trusted scorer:

| Attempt | Training / selection | Full test validity | Test Hit@1 | MCES@1 | CPU replay |
| --- | --- | --- | --- | --- | --- |
| Full-data DeepSets, repaired starter | 50 epochs, 1 H200, 504.8 s; full validation Hit@1 2.646% | 17,556 / 17,556 | 4.004% | 22.560 | 283.3 s |
| Original cold agent | 2,048 training / 128 validation queries | Failed on a legacy candidate SMILES | No quality measurement; failure reward 0 | Failure convention 100 | Failed after 7.9 s |
| Fresh cold agent, repaired starter | 4,096 training / 256 validation queries; 709.1 s agent wall time | 17,556 / 17,556 | 1.572% | 25.821 | 264.3 s |

The two agent attempts use Codex CLI 0.153.4, gpt-6-astra/xhigh, a 30-minute
eight-CPU/no-GPU pilot override, and no prior conversation or organizer weights.
Both original packages and every failure are retained. The second package was
independently replayed without changing its code or weights. Its learned
parameters were independently checked against random initialization. Different
starter revisions and training budgets make this a usability/acceptance table,
not an agent ranking or an H100 campaign result. The pinned native runtime was
present in the CPU pilot, but the image's separate base-Python MCES dependencies
were not; MCES was therefore measured only by the trusted organizer grader.
Completed follow-up on 2026-09-13: the native H200 scratch reference finished training at
step 24,186 with 20,988.4 cumulative charged seconds. Full validation replay
completed in 16,871.5 seconds (4 h 41 min), with all 19,429 rankings valid and
Hit@1/5/20 = 35.987% / 49.967% / 63.467%. The combined 11-hour Slurm allocation
expired during subsequent test inference. The queued H200 recovery was canceled
at the user's request. Recovery on four CS A5000s completed prediction; a
subsequent 16-CPU pass completed ranking and trusted grading in 488.1 seconds.
All 17,556 rankings are valid: Hit@1/5/20 = **40.322397% / 56.846662% /
70.773525%**, MCES@1 = **13.169600**. The checkpoint and scorer were unchanged;
the preserved 2,537 partial rankings match exactly. This completes score
reproduction, with no job still queued. Production Docker privilege separation,
single-H100 timing, storage quotas and live API allowlist transport remain
deployment gates; this heterogeneous recovery does not certify those limits.

These full-campaign gates are our commissioning criteria for agent comparison,
not extra repository contribution rules. `CONTRIBUTING.md` allows a documented
draft with a small real training/replay smoke and explicitly does not require
expensive full training. A complete experiment record is a campaign reporting
requirement; the model verifier does not grade prose or reject an otherwise
valid package because that record is absent.
