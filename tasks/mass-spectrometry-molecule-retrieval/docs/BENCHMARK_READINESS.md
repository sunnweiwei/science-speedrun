# Benchmark readiness and agent research design

Audit date: 2026-09-08. Inspected task revision: `539eb43`; fetched main:
`e0606c4`. This is an organizer-facing assessment, not an agent instruction or
a change to the frozen scoring/data/initialization policy.

## Current deployment follow-up (2026-09-25)

An independent [PR 27 reviewer replay](https://github.com/T0-RSI/ai4sci-tasks/pull/27#issuecomment-5750791449)
reported **3.4–3.9 hours for the full 17,556-query replay on one H100** at
native inference batch 64, versus about five hours at batch 8. New publications
now default to inference batch 64, retain two CPU workers, and accept a positive
`--inference-batch-size` override when GPU memory requires a smaller batch.
Training still defaults to batch 8. This setting does not change model weights,
routing, training data, reference metrics or score anchors.

This reviewer timing supersedes the earlier blanket statement that single-H100
timing is unavailable. It does not establish acceptance of the current native
Docker images with freshly downloaded HF inputs, resource enforcement, live
agent API transport or a matched-budget training campaign. Those deployment
checks remain separate. Dated audit and execution notes below, including their
then-outstanding checks, are preserved as historical evidence.

## Results and SOTA publication pass (2026-09-13)

The user authorized committing the completed task updates to existing PR 9 and
requested explicit training/test data, reproduced results, reference methods,
and an independently checked SOTA statement. Preserve the frozen score/data/
initialization contract; use CS only and launch no new training or test inference.

Competing claims: (H1) GLACIER is the highest supported forward-simulator result
on the released simulation domain; falsify with a newer higher comparable
result. (H2) a larger headline such as SpecBridge can be used as the full-v1.5
target; falsify with withdrawal, changed split/candidate protocol, or absent
corrected evidence. Check current primary-source status, tables, source
configuration and the official results CSV before writing. SpecBridge's latest
arXiv record confirms withdrawal on 2026-03-03 for preprocessing/evaluation
problems; the older 84.695% number is excluded from current SOTA claims.

Budget: up to 40 minutes of literature/source inspection, documentation, local
CPU checks and publication, reserving 15 minutes for checks and PR verification;
no extra GPU allocation or training/test-model run. Preserve source URLs and
result hashes in the existing reference/evidence layout. Completion requires
consistent data/method/result tables, explicit unresolved deployment gates,
passing relevant regressions, a clean reviewed commit, successful push to the
existing PR branch and read-back of the updated PR description. Do not create
another PR. This pass cannot establish a community-wide full-v1.5 first place
from a subset result or an incomplete leaderboard.

Publication package validation: all **33** regressions passed in 3.411 s; all
six actual runner/Harbor networking profiles and the legacy negative control
passed. Fourteen shell files and fifteen documented Bash blocks pass syntax
checks; thirty local Markdown links resolve. Aggregate test/branch metrics,
checkpoint identities and the original trusted report agree. The frozen
reference JSON, inference contract and trusted grading code are byte-unchanged.
The first ad hoc evidence check used the wrong filename `tests/grade.py`; it
was corrected to the existing `tests/grader.py` and passed. No model execution
was involved in these checks. Primary-source findings and exact task scores
are published in `environment/references/SOTA_AUDIT.md` and
`environment/references/task_trained_baseline.json`. The production Docker/API,
single-H100 timing and campaign resource-enforcement gates remain explicit.

## Active deployment acceptance (2026-09-08)

### Latest result: full scratch-model test scoring complete (2026-09-13)

The frozen scratch GLACIER + trained DeepSets fallback has now completed all
17,556 test queries. Trusted grading accepts every ranking, with no missing or
invalid rows: **Hit@1 40.322397%, Hit@5 56.846662%, Hit@20 70.773525%,
MCES@1 13.169600**. The hit rates agree exactly with an independent primary-only
invocation of the same trusted grader on the sealed output. This is the full
test result; the earlier 35.987441% Hit@1 is the validation result. The external
official-weight GLACIER + strict-clean MIST reference remains 49.390522% Hit@1,
9.068125 percentage points higher, with different initialization and fallback.

The four-GPU prediction stage of CS job `12582664` completed and preserved all
four closed HDF5 shards. Its serial CPU ranking then exceeded the operator's
combined 13,800-second deadline after 2,537 complete native query rankings.
That failed attempt and the superseded single-GPU job remain archived; both
GPU allocations have ended. The earlier three-hour estimate underbudgeted CPU
ranking and was incorrect.

Recovery completed on existing CS CPU allocation `12473780`, using 16 CPU
workers and no GPU, in 488.090 seconds including trusted grading. Saved
prediction files were hash-checked and copied to local scratch. The unchanged
frozen scorer processed 16 disjoint query groups in input-only namespaces.
All 9,954 native rankings have zero missing supported predictions, all 2,537
previously completed ranking rows match exactly, and the 7,602 preserved
fallback rankings complete the full test set. Training, GPU prediction and
model selection were not repeated. The checkpoint and model manifest are
unchanged. Full evidence, hashes and `grading/report/report.json` are in
`_massspecgym_repro/task_release/acceptance-40edda8/cs-cpu-score-20260913`.

This completes score reproduction on the available CS hardware. Production
Docker/API transport and the declared single-H100 inference deadline remain
unverified; this heterogeneous execution does not establish a matched-H100
reference or an agent comparison. The dated execution notes below preserve
historical plans and intermediate states; this result supersedes their running
or queued status. At that scoring stage no PR update was made; publication is
handled by the separately authorized pass above.

### Historical execution: CS four-GPU acceleration (2026-09-12)

The user requested idle A5000 GPUs to obtain the frozen model's test result
sooner. CS job `12582664` now reserves four A5000s on `compsci-cluster-fitz-15`,
32 CPUs and 192 GiB for at most four hours; four-GPU CUDA preflight passed.
The original single-GPU job `12582010` remains available until a public-input
single/four-GPU numerical consistency check passes and all four full-inference
shards produce batches. The collector then cancels only that superseded job
and its own old collector, preserving the original attempt and logs.

A brief, 5.77-second pause captured the original temporary output and always
resumed the predictor. Its still-open HDF5 snapshot is unreadable and is
excluded from scoring. Closed candidate preparation, public input partitions
and completed DeepSets fallback rankings are reused with recorded hashes.
The first four-GPU attempt `12582663` failed its predefined numerical control:
upstream molecule-based repartitioning changed batch composition and one checked
binned spectrum differed by relative L1 0.00392, above the 0.0001 tolerance.
It ended before full inference and before any handoff; the failure is preserved.
The replacement driver distributes complete original batches across GPUs,
retaining their composition and ordering, then invokes the frozen upstream
shard executor. It keeps the same numerical tolerance, checkpoint, batch size
and candidate scorer/merger. Its numerical control passed for 63 public
candidate spectra with maximum relative L1 difference exactly zero; full
candidate preparation is now running. Completed HDF5 files move from node-local scratch
to persistent output after closing.
It must verify no supported candidate predictions are missing and grade all
17,556 queries. This is multi-GPU score reproduction, not the original
single-GPU package's timing acceptance. No training or test-driven selection
is performed. Evidence and current per-run status are in
`_massspecgym_repro/task_release/acceptance-40edda8/cs-parallel-batches-20260912`;
the first attempt and snapshot remain in `cs-parallel-20260912` alongside it.

### Latest follow-up: runner network compatibility (2026-09-12)

Following the user's request to use the other tasks' solution, development
Compose now leaves network ownership to agent-runner/Harbor, as the clinical
tasks do. The task TOML retains its fail-closed default. Runner `--network
allowlist` can attach the development container to Harbor's API-only egress
sidecar; final model execution and trusted grading retain `network_mode: none`.
The prompt and operator launch recipe now distinguish research-agent API calls
from locally supplied scientific data/runtime assets and fully offline replay.
The recipe uses prebuilt images and mounted Codex binaries. Other adapters need
their own verified dependency preparation, without download hosts during research.

`tests/check_runner_network.py` passes against runner `e75f5f2` and Harbor
`6af8d6e`: six staged profiles (allowlist/no-network, 0/1/8 GPUs), actual Harbor
service selection, both offline verifier services, and a negative control that
reproduces the old Compose conflict. All 33 existing regression tests pass.
This resolves the configuration conflict recorded below. It does not certify
live Docker/API transport; this CS host has neither Docker nor Podman available.
No new training or agent run was launched for this change. These configuration
repairs are included in the September 13 publication pass above.

The earlier DCC recovery plan below is historical and superseded by the user's
CS-only instruction: queued jobs `55271690` and `55274238` were canceled. The
unchanged frozen model is currently replaying on CS job `12582010`, one A5000,
16 CPUs and 64 GiB, with a 12-hour allocation and an 11.5-hour inference ceiling.
This measures the test score on CS hardware, not the formal five-hour H100
deadline. The existing collector and experiment continue unchanged. Current
operator evidence: `_massspecgym_repro/task_release/acceptance-40edda8/cs-replay-20260912`.

### Follow-up status (2026-09-12)

The combined job `54735107` ended with Slurm `TIMEOUT` on 2026-09-09 at
07:01:55 America/New_York after 39,614 allocated GPU seconds. Native training
finished at step 24,186 (20,988.441 cumulative charged seconds; selected
validation spectrum loss 0.43755746), and the frozen package completed the
full validation replay in 16,871.538 seconds. The test replay then lost its
allocation before producing rankings. This is an operator scheduling failure:
the 11-hour combined reservation did not leave a complete test-inference window.
It is not evidence that a standalone test replay exceeds its five-hour limit.
The previous monitor recorded the failure and exited; no run continued afterward.

Recovery preserves that failed attempt, the frozen weights and validation
predictions. Collect and independently score completed validation now. Run only
the frozen package's full test replay in a separate one-H200 / 16-CPU / 64-GiB
allocation capped at 5.5 hours, retaining the unchanged five-hour model deadline;
then collect, seal and grade on the CS CPU host. No additional training or model
selection is authorized by this recovery plan. The two prior GPU allocations
used 42,586 seconds (11.829 GPU-hours); this bounded replay adds at most 5.5
GPU-hours. Preserve model/input/ranking hashes and report any further failure.
H100/Docker acceptance remains unverified.

The recovered validation artifact passes submission validation and all recorded
input/model/ranking hash checks. Trusted validation scoring accepts all 19,429
rankings: Hit@1/5/20 = 0.3598744 / 0.4996654 / 0.6346698. Test recovery job
`55271690` is submitted and currently queued with `AssocGrpGRES` (account GPU
quota); its queueing time is unknown. A separate collector retains validation
even if test replay fails, waits at most 24 hours for allocation, and then
collects and grades only the unchanged frozen test submission. It cancels only
this recovery job if its monitoring deadline is reached. Raw recovery evidence:
`_massspecgym_repro/task_release/acceptance-40edda8/native-recovery-20260912`.

User request: "那你快验证啊". Move from audit to actual execution. The release
under test is `40edda8`, already pushed to PR 9. Do not treat another checklist
or synthetic test run as completion of the outstanding deployment gates.

Hypotheses: (H1) the existing generic agent-runner can run this task after
explicit deployment configuration; falsify with actual startup, resource or
artifact-transfer failures. (H2) the native scratch recipe and relocated
inference fit a useful research budget; falsify with measured full-data timing,
OOMs, incomplete rankings or a failed cold agent submission.

Execution plan: verify accessible CS/DCC resources and the existing runner;
run an isolated cold agent workflow pilot; run the released full-data scratch
recipe and independent replay on an allocated GPU. Initial reservation is one
GPU for up to 12 hours (six-hour recipe, five-hour replay, one-hour contingency),
plus a bounded 30-minute agent pilot on the existing authenticated model. This
is commissioning, not an identically budgeted agent-comparison campaign. Archive
raw usage and actual hardware; no H100 claim may be inferred from another GPU.
Keep hidden labels out of model execution and do not tune on test outcomes.

Initial infrastructure evidence: CS has no H100. DCC lists an idle H100 node,
but Slurm rejects this user's account/partition combination. The user has
H200 allocation rights. DCC Podman is available but has no subordinate UID/GID
range for this account; do not equate single-UID emulation with the production
UID-10001 boundary. The generic runner at `e75f5f2` already handles Docker GPU
reservations, so its actual integration must be checked before proposing a
new backend. Fresh runtime/input-only namespace checks remain useful evidence
when a production container environment is unavailable.

Historical execution ledger (2026-09-08): DCC job `54732494` ran one H200,
16 CPUs and 64 GiB under `/work/xy200/massspecgym-acceptance-40edda8`.
All public data (38.0 GB) and the pinned runtime (7.2 GB) were transferred;
private test labels were not. Full-data DeepSets completed 50 epochs in
504.792 seconds. Native GLACIER completed 128 optimizer steps and full
9,720-row fragment validation in 616.996 wrapper seconds, peak CUDA allocation
1,510,504,960 bytes; validation loss 0.896647. The cumulative six-hour recipe
is resuming, with retrieval assessment still pending.

The clean DCC Codex pilot (`54733191`, CLI 0.153.4, gpt-6-astra/xhigh,
30-minute CPU cap) exited normally in 760.513 seconds. It trained on 2,048 rows,
selected epoch 3, reported 128-query validation Hit@1 0.03125, and passed its
own relocated subset replay. Its full-input organizer replay **failed** in
7.928 seconds: native RDKit 2025 rejects a legacy candidate valence accepted
by the task's older RDKit. The original package and zero-reward failure are
retained; the subset success does not count as full acceptance. The full-data
DeepSets package failed on the same candidate, isolating the starter defect.
Repair: keep unparseable inference candidates using zero fingerprint features
and deterministic ID ties, while retaining strict training validation. Also
repair precursor-only encoding when the model's m/z window removes every
positive peak. The repaired reference completed all 17,556 test queries in
283.284 CPU seconds: Hit@1/5/20 = 0.0400433 / 0.1235475 / 0.2740374,
MCES@1 = 22.55958. Full 19,429-query validation completed in 322.710 seconds,
Hit@1 = 0.0264553. These are complete input-only namespace replays followed by
the actual sealing/trusted-scoring code; they are not Docker/H100 acceptance.
A fresh second cold pilot (`54734283`) received the repaired starter and the
same prompt/model/30-minute cap, with no previous trajectory or checkpoints.
It finished in 709.126 seconds, trained 4,096 rows for three epochs, and selected
on 256 validation queries (Hit@1 = 0.01171875). Its independently validated
original package completed all 17,556 test queries in 264.326 CPU seconds:
Hit@1/5/20 = 0.0157211 / 0.0577011 / 0.1625655, MCES@1 = 25.82126.
All 20 learned parameter tensors differ from seed-0 initialization, all weights
are finite, and its model source matches the released starter. Both pilot
outcomes are retained; their differing starter revisions and tiny budgets do
not establish an agent-system ranking.

Native input support probes also exposed a failure when every candidate exceeds
the model's 100-heavy-atom limit. The native starter now partitions using public
observation and structure support, routing such cases through the trained
fallback. The official-weight reference's default routing remains unchanged.
A relocated four-query probe covers actual native inference, precursor-only
fallback, an entirely unsupported pool and a legacy valence: four exact
permutations, no missing candidates, and only the contracted ranking file.

The first full-data timing run exposed a cumulative-budget gap: the resumable
fit checkpoint predates final validation/export. Resume now charges the maximum
of checkpoint usage, the completed result and an optional interruption ledger.
At optimizer step 500, job `54732494` was stopped with its state preserved and
resumed with the repaired starter as job `54735107`. Its 1,626.558-second prior charge is
conservative and explicitly includes in-flight work and the initial profile;
it is not a measured elapsed-time claim. The new reservation is 11 hours, keeping
total planned GPU allocation below the initial 12-hour envelope. Full native
training, full validation retrieval and full test replay were in progress at
that point; the follow-up above records the eventual timeout and recovery.

The earlier CS pilot failed before commands because nested namespaces were
blocked by AppArmor. The first DCC transport startup failed before a model
response because its system CA symlink target was not mounted. Both failures
remain archived, distinct from the completed agent's failed full replay.
Current pilot transport shares the outer CLI network for authenticated model
calls while Codex shell commands run in its offline sandbox. This does not
commission Docker/Harbor or the installed mini-swe-agent adapter.

Runner staging at `e75f5f2` exposed two configuration issues: hard-coded main
Compose resources survive CPU/profile overrides, and evaluation inherits the
development GPU count unless `--gpus 1` is explicit. Remove main resource
hard-coding from base Compose files and keep it in the manual replay override.
Staging 0/1/8 GPU profiles then passes. The installed `agent-harness` adapter at
`bde0118` needs installation/model egress; the task's then-explicit development
`network_mode: none` blocked it even with runner allowlist policy. The latest
follow-up above fixes that Compose conflict and supplies a prepared Codex launch
recipe. Mini-swe-agent installation and live API transport remain separate
deployment checks; do not open download hosts during research to bypass setup.

Local raw evidence and repaired artifact copies:
`_massspecgym_repro/task_release/acceptance-40edda8`. Agent authentication files
are outside evidence and removed after each completed attempt.

## Cross-task requirements check (2026-09-08)

Fetched `origin` again; main remains `e0606c4`. The normative sources are the
root [contribution guide](../CONTRIBUTING.md) and [README](../README.md).
Sibling implementations are examples, not additional global requirements.
The meeting image is context only: hiding the budget, choosing a particular
architecture or changing all tasks into pretraining/finetuning are not adopted
as mandatory rules.

**Verdict: the task fits the repository convention after the repairs below;
production agent-evaluation acceptance is still incomplete.**

| Item | Common requirement and sibling comparison | MassSpecGym finding |
| --- | --- | --- |
| Budget | Explicit, realistic resources; no global hardware/time cap. Clinical tasks use 24 h / 1 RTX 5080; CrystalGen 24 h / 8 H100; weather 24 h / 4 A5000. FoldBench, PEER, virtual-cell and HydroGym omit an agent timeout. | 48 h / 8 H100 is permitted, but its 384 GPU-hour allocation ceiling is not a measured reference cost. Retain the declared ceiling pending calibration; do not claim parity by matching hours alone. |
| Prompt | Objective, assets, allowed changes, data/resource rules and portable deliverable must be explicit. | Clarified validation inspection versus fitting, 8-GPU development versus 1-GPU inference, the 5 h inference / 6 h total deadline, supplied-runtime exceptions and required README. |
| Data | Accessible, pinned, attributed, integrity-checked resources; explicit injection and test visibility. | Core HF revision and all prepared resource hashes are recorded. Version 1.3.0 distributes prepared core, sanitized augmentation and MAGMa views through a pinned task-owned HF release, preserving the original HF/Zenodo provenance. Train/validation and private test labels are separated; upstream answers remain public. |
| Starting conditions | Reconstructable training starter and explicit pretrained policy; domain-specific choices are allowed. | Native scratch GLACIER and scratch DeepSets are supplied. No weights are authorized. 49.391% is an external official-weight target; the independently trained recipe obtained 40.322% on the full test using historical H200 training and CS A5000/CPU replay. Matched-H100 timing remains unverified. |
| Runtime | Development assets must actually be present; independent replay must contain declared dependencies and relocated artifacts. | Source/runtime and data probes passed. Main CPU/RAM/GPU limits come from the runner profile; manual replay has an explicit override. Runner/Harbor API allowlist staging passes; Docker/H100 builds, writable-storage quota enforcement and live API transport remain deployment gates. |
| Acceptance | Rerun the package on unseen conforming inputs; numeric reward in `/logs/verifier`; missing predictions stay in the denominator; scorer faults differ from model failures. | Native schema/artifact path/separate verifier agree with siblings. Fixed reward directory permissions, stale-reward removal, failure statuses, atomic result publication, malformed-ID handling and reference comparison on changed inputs. |
| Research record | The artifact is the graded object; a report is not a substitute for it. | `EXPERIMENTS.md` and selected research files remain campaign audit requirements. Their presence/prose is not part of Hit@1 or automated model validity. |

Sibling task configurations were read at: main `e0606c4`, PEER `2460f7c`,
CrystalGen `039145e`, weather `b59bf2d`, virtual-cell `daa4b9a` and HydroGym
`b525043`. No sibling branches were merged. Their checkpoint, data and runtime
claims were not freshly reproduced in this compatibility check.

Concrete implementation corrections:

- The trusted log directory was mode 0777. An inference process could write
  `reward.json` or `reward.txt`, and the inspected Harbor verifier reads a
  reward file without requiring a successful test exit. It is now owned by
  the trusted verifier, mode 0755; the sealed volume is 0700 and stale rewards
  are removed. Production cross-UID Docker verification is still required.
- Invalid packages and model timeout/process failures now report explicit
  failure statuses and zero hit rates. Input/setup/trusted-grader errors
  report infrastructure failure without a reward; a grader error is signaled
  immediately rather than waiting an hour. READY and JSON publication are atomic.
- An invalid candidate permutation invalidates the whole ranking file, as the
  existing scorer already did. The contract now says so. Missing queries alone
  remain individual misses. Non-string query IDs no longer crash the scorer.
- Future conforming or subset inputs receive their actual metrics, but no
  `beats_reference` claim against a different frozen test set.
- Manual replay keeps main alive and invokes `test.sh` explicitly. The old
  `--abort-on-container-exit` command could stop main when the grader exited
  before reward collection. Use fresh Compose volumes for each attempt.

Harbor source validation used local revision
`6af8d6e31eced13b93849cdf80feeadf24603d15`. `TaskConfig` accepts all local
task configurations. Its default Docker backend reports `gpus=False` and only
CPU/memory resource limits. Thus the earlier unqualified `harbor run` example
was not a working GPU launch command. The protocol now requires an explicitly
validated generic backend supporting GPUs, Compose/sidecar isolation, artifact
transfer and controlled agent API access. The current recipe uses agent-runner's
GPU integration and Harbor's API allowlist. Adding Compose GPU reservations does not
override Harbor's backend capability check. GPU model and disk quota enforcement
also require deployment evidence. No task-specific runner was added.

The single-H100 replay remains outstanding. The native scratch recipe has a
complete full-test score on heterogeneous hardware, and the cold CPU pilot
above has passed the complete submission/scoring workflow.
These are our criteria for a commissioned agent comparison, not repository PR
prerequisites. The contribution guide explicitly permits drafts with gaps and
does not require expensive full training. Freeze agent/model revision, API/token
cost policy, attempt count, prompt, stopping rule and hardware in the campaign
before comparing agents; those runner settings are still unconfigured.

Validation for this follow-up: 29 unit/regression tests passed in 2.716 s;
native Harbor schema parsing and Compose CPU/RAM/GPU declarations agree;
14 shell scripts pass syntax checks. A synthetic local sealing/scoring CLI
probe produced Hit@1 = 0, Hit@5/20 = 1 and MCES@1 = 2 for one deliberately
wrong first-ranked candidate. Corrupting the private-label hash produced a
grader error and no reward. This checks the real MCES and error-reporting paths,
not model inference or production UID isolation. Evidence is linked from
`release_validation.json`.

## Implementation state (2026-09-08)

The user now requests implementation so the task can evaluate agents. The
read-only scope of the original audit below is superseded for this phase.
Keep the current from-scratch initialization rule, complete test denominator,
and frozen 49.391% official-weight comparison. Implement a native GLACIER
research starter, prepared train/validation-only fragment supervision, offline
runtime, resumable training, continuous publication, validation feedback,
independent replay, and launch/acceptance documentation. Do not change the
research architecture merely to obtain an easier smoke result.

Execution envelope: begin with small CPU data probes and <=1 GPU-hour smoke;
profile before a bounded reference run (initial cap 8 GPU-hours). Shared
train-only preprocessing may run in parallel with implementation. Reserve
compute and wall time for full validation/replay, a second method if needed,
and failures. This host has A5000/P100-class cluster resources, not H100s;
record actual hardware and do not infer H100 acceptance. No token/API-spend
budget for a scored agent campaign has been configured.

Implementation checklist:

- [x] Native runtime installer succeeds in a fresh directory without weights;
  both Dockerfiles use it. Actual production Docker builds remain unverified.
- [x] MAGMa preparation consumes supplied train/validation only; source/data
  binding and resumable preparation are implemented. Full preparation and
  development-view installation passed complete file-hash validation.
- [x] Native GLACIER fit/resume and portable all-condition submission.
- [x] Periodic resumable checkpoints and atomic complete publication on NFS.
- [x] Validation strata, molecule-cluster uncertainty and experiment record policy.
- [x] Real-data train/relocate/replay and failure probes; full-H100 gate remains.
- [x] Native Harbor task schema and frozen resource/initialization policy.
- [x] Generic runner GPU staging and Harbor API allowlist service selection.
- [ ] Live GPU/Compose execution and controlled agent API transport.

Implementation evidence: the native 15,108,154-parameter model trained 128
optimizer steps on 4,086 annotated training spectra with 511 validation spectra
in 241.1 seconds of wrapper elapsed time on one P100. This is a runtime probe,
not a full-data reference. Its selected validation spectrum loss was 0.893976;
that loss is not retrieval Hit@1. A separate CPU experiment produced identical
parameters, batch order and RNG for continuous versus resumed four-step fitting.
The repairs include the native MAGMa schedule counter, independent sampler and
worker RNG generators, and checkpointing before Lightning advances end-of-fit
bookkeeping. Both CPU and P100 input-only namespace replays covered two native
and six fallback real queries. A random 32-query validation replay also passed,
covering 18 native and 14 fallback queries over both instruments and missing
collision energies. Both single-process and two-worker CPU recovery checks
produced identical parameters and batch/RNG traces. The final package was
re-exported with cross-UID read permissions and replayed again after training
its fallback in the fresh pinned runtime.

The full fragment resource selects all 99,341 eligible training and 9,734
eligible validation spectra. It annotates 99,119 and 9,720 respectively; the
other 236 have no MAGMa fragment and are recorded explicitly. No query is
removed from the benchmark denominator. All declared core, synthetic and
fragment resource hashes passed, including the 2,333,485-row synthetic corpus.
The initial twenty-three unit/regression tests, all shell syntax checks and `git diff
--check` pass. Actual evidence and operator artifact paths are recorded in
`release_validation.json`.

Isolation canaries confirmed absent organizer-private/training paths, read-only
input/package mounts, no external network route, and zero effective
capabilities. These use rootless mapped UID 0, not the production UID-10001
Docker path. No real agent/API campaign or production H100 experiment has run.
The engineering deliverable is a concrete task package and pilot protocol;
benchmark ranking validity is not inferred from these commissioning probes.

Latest decision: retain the native method and full test denominator. The
runtime probe gives no reason to simplify the architecture for feasibility.
Resource materialization, integration checks and evidence archiving are complete.
Leave matched-budget full reference, Docker/H100 acceptance and a named
agent pilot as explicit campaign gates in `AGENT_EVALUATION.md`.

Working alternatives: native GLACIER plus a small trainable all-condition
fallback; a single extended native forward model if condition coverage permits.
Start with the former, since its public routing rule is already frozen. A cold
agent usability probe can follow once the actual release passes the offline
training and replay gates. Benchmark agent ranking requires a later campaign;
do not equate a plumbing pilot with such evidence.

## Original audit contract (before implementation)

Question: can this task measure autonomous scientific model development from
the resources actually supplied to an agent, under a reproducible budget?
Deliverables: evidence-backed readiness judgment, comparison with sibling
tasks, a concrete agent resource design, and an ordered acceptance plan.

The user asked to pull updates, inspect other tasks and the newly completed
SOTA work, identify corrections, and carefully design what an autonomous agent
receives. The meeting image is reference material, not an instruction to execute
its proposed changes. Preserve the full 17,556-query denominator and existing
data policy while auditing. Do not conflate official-weight replay with training
from scratch or a paper result with a task result.

Live hypotheses:

- H1: the existing release already supports research from a strong starting
  point. Falsifier: its advertised methods require unavailable offline assets,
  dependencies, supervision or nonportable organizer paths.
- H2: this is a suitable scientific task but needs a complete research starter
  and deployment acceptance. Falsifier: after those repairs, representative
  train/evaluate iterations cannot fit the declared budget, or independent
  agent pilots mostly measure environment repair rather than model choices.

Evidence ladder: inspect committed code and sibling evidence; verify local
ranking/report digests and inexpensive metrics; run existing unit tests; then
specify clean-container training/replay and cold-start agent pilots. This audit
uses CPU/read-only checks and bounded documentation edits, with no new GPU
training or paid agent campaign. Token/API and GPU budgets for future pilots
must be frozen with the runner configuration; none are inferred from the meeting
notes. Reserve pilot time for checkpoint export, replay and one recovery attempt.

Pivot if the strong recipe cannot make multiple useful iterations within budget:
shorten or simplify a measured recipe and retrain its reference under the same
rules; alternatively define a separately versioned weight-initialized task.
Do not quietly shrink the test set or allow only the reference extra resources.

## Historical initial decision and evidence (2026-09-08)

The dated follow-ups above supersede this initial assessment where implementation
or measurement subsequently completed.

**Suitable benchmark candidate; research-ready release acceptance is incomplete.**
The scientific objective, full-data baseline, useful performance headroom and
strict ranking contract are established. The advertised strong research path is
not yet supplied inside the offline agent environment. H1 is falsified by the
resource/image inspection; H2 remains the working interpretation. The next
milestone is a complete, measured agent starter and a cold-start pilot, not
another operator-side test-score optimization.

The audit independently loaded the frozen public input with `tests/common.py`,
validated every candidate permutation with `tests/grader.py`, and recomputed
Hit@1/5/20 from the existing frozen rankings. All three match the reference lock.
Hybrid, MIST, independent-rescore and stored MCES-report digests also match.
MCES itself and GPU forward inference were not rerun. Detailed aggregate checks
are in [readiness_checks.json](readiness_checks.json).

| Existing result | Queries | Hit@1 | What it establishes |
| --- | ---: | ---: | --- |
| Fourier-DeepSets | 17,556 | 5.126% | Fast trainable interface starter |
| Strict-clean MIST pipeline | 17,556 | 39.872% | Strong full-task comparison; MIST encoder trained independently |
| GLACIER no-CF, supported conditions | 9,954 | 67.490% | Official-weight replay on the released condition domain |
| MIST fallback, remaining conditions | 7,602 | 25.691% | A substantial remaining coverage challenge |
| GLACIER/MIST hybrid | 17,556 | 49.391% | Current frozen full-task reference |
| Constant-spectrum hybrid | 17,556 | 3.139% | Reference depends on observed spectral information |

GLACIER training was **not** performed. The 33.1 aggregate P100 GPU-hours
measure inference construction/recovery, not training or a single-H100 replay.
The independent byte-identical result is a rescore of cached simulations, not
a second complete GPU inference run. The paper's no-CF 69.69% and CF 69.95%
remain distinct published results; neither is the full-task target. See the
[official paper](https://arxiv.org/html/2606.29161v1) and
[reference reproduction ledger](environment/references/GLACIER_REPRODUCTION.md).

The fallback domain comprises 43.30% of the test spectra. A hypothetical ten
percentage-point gain there would add 4.33 points overall if the other domain
were unchanged. This is a sensitivity calculation, not an achievable-gain claim.
Use validation to choose methods and diagnose conditions; these organizer-side
test strata must not become an iterative test-selection channel.

## Comparison with other task work

The local branch already contains fetched `origin/main`. Sibling branches were
read with `git show` at the revisions below; they were not merged into this task.
The related Codex task `rsibench` was inspected for context. GitHub PR search
was unavailable to the connected account, so PR review/approval state is not
asserted. Sibling scores below are checked-in evidence, not fresh reproductions.

| Task / revision | Evidence inspected | Lesson and remaining limitation |
| --- | --- | --- |
| clinical-prediction, `e0606c4` main | `results/ClinicalSetTreeSVD-Foundation/README.md`, `agent_score.json`: a real agent result, isolated index-0 replay, combined score 0.7355 | Archive agent identity, snapshot, cost and scores. Its archive omits exploratory training scripts and the checkpoint payload, so preserve a fuller reproduction bundle here. |
| clinical-trial-modeling, `e0606c4` main | Shared runtime plus a trainable eleven-task multimodal reference under `environment/baseline/starter/reference/` | Put usable research code beside the starter; a method description alone is insufficient. |
| PEER, `2460f7c` | Random-init pretrain/finetune references, 0.6147 and 0.6368 reward, each about 11 h on one A100; resume/publication and cold-context integration evidence | Closest model for our research starter. Real Docker privilege separation and authenticated Harbor agent conversation still have gaps; the full 37.3M-sequence corpus was not used for those references. |
| CrystalGen, `039145e` | Trainable conditional OMatG port, prepared scorer assets, eight-H100 smoke | Supply the actual modern architecture and matched evaluation. Its 200-example smoke had zero SUN/MetaSUN, and full 12,500-structure replay remains pending. |
| Weather, `b59bf2d` | `WN2_SHORT_TRAIN.md`: full WN2 architecture with shortened random-init curriculum; profiling in progress | Scale training by measured stage cost. Legacy GenCast delivery success does not certify the new WN2 task. |
| Virtual Cell, `daa4b9a` | STATE-oriented design, target-disjoint split and resource materializer | Real full-data Harbor/Docker replay and Cell-Eval parity remain pending. The meeting's pretraining proposal is not established by this branch snapshot. |
| HydroGym, `b525043` | Frozen 39-case, 453.756 GB release; short native control/cost probes | Data availability and unit tests do not establish research feasibility: whole reference-round estimate is about 1,118 GPU-hours; full physical/evaluator and joint-model acceptance are pending. |

Other tracks are at different stages. Their incomplete checks are reasons to
report our own stage precisely, not reasons to lower the acceptance bar.

## Corrections required before a research-ready release

| Priority | Finding and direct evidence | Concrete correction / acceptance |
| --- | --- | --- |
| P0 | Offline image copies wrappers/docs and DeepSets, with torch 2.3, NumPy 1.24 and a small dependency set. GLACIER preparation requires network downloads and an external `ms-pred` environment; its pinned source requires a different compiled dependency stack. `environment/Dockerfile`, Compose and `prepare_glacier_reference_assets.sh` establish the mismatch. | Build and pin the real train/inference runtimes and install source before the timed run. Test actual imports, one backward update and relocation with networking disabled. Do not simply install the MIST requirements over the GLACIER environment. |
| P0 | `instruction.md` advertises GLACIER contrastive finetuning while the initial release supplies no pretrained assets; neither GLACIER nor MIST source/weights are mounted. The exact clean fallback checkpoint is an operator-local path. | Freeze one initialization policy and make the reference/agent resource comparison explicit. Implement one of the two versioned protocols below. |
| P0 | There is no task-owned GLACIER training adapter from `/data`, prepared MAGMa training supervision, or bounded train-negative path. The 2.33M simulated-spectrum store is not GLACIER's atom-level breakage supervision. | Ship a tested train-data adapter, train-only derived supervision and local negative construction. Keep all answer-bearing raw releases outside the agent image. |
| P0 | Full reference rankings were assembled by operator scripts. No full hybrid artifact has demonstrated the exact fresh offline verifier path under the declared hardware/storage limit. | Export `infer.sh` plus every owned asset, move it, run the real main/grader split, and time full inference plus sealing and scoring. Validate 17,556 complete rankings without shared organizer caches. |
| P1 | 48 h × 8 H100 is a configured ceiling, not a measured research budget. `main_verifier.py` defaults to **18,000 s inference**, followed by up to one hour for grading, inside the six-hour verifier envelope. | Measure total recipe/validation/export/replay costs. The model must fit the five-hour inference default, not consume six hours of inference. Leave headroom for scoring and failures. |
| P1 | DeepSets writes its best state only after all epochs; there is no resumable optimizer/RNG state or periodic complete publication. Only `/workspace/submission` is declared as a Harbor artifact. | Add separate periodic resumable research checkpoints and validated atomic publication of a selected inference package. Archive training code/configs, experiment ledger and agent trajectory through the runner/artifact configuration. |
| P1 | Only the GLACIER-domain validation reference is recorded here; a comparable full hybrid validation result is needed for the agent's primary objective. | Publish full-validation and condition-stratified reference reports under the same complete ranking contract. Keep final test selection operator-only. |
| P1 | There is no archived autonomous-agent research run for this task. Unit tests and hand-driven reference construction do not measure agent usability. | Run a cold-start smoke after P0 closure, then independent research pilots with frozen prompts/resources. Record failed attempts as well as successful ones. |

Do not classify operator source-download wrappers as an offline research starter.
The pinned source can remain immutable as reference provenance while a writable
copy is supplied for agent edits. The reference replay's exact checkpoint-hash
guard must not be reused as a restriction on candidate model weights.

## What to supply to the agent

Keep three roles explicit: DeepSets for a quick contract check; a modern,
trainable research starter for substantive experiments; frozen reference results
for comparison. The research starter should preserve GLACIER's useful scientific
structure: graph encoder, fragment queries, spectrum objective, and a documented
route for conditions outside its current domain. Model size and recipe length
should be selected by measured cost, not by making the architecture unrelated
to the reference.

Recommended main protocol: **train from scratch on supplied resources**. This
preserves the initial release's no-pretrained-assets policy and tests model
development. Supply the modern architecture and complete training recipe, then
measure a task-owned reference under the same data/initialization/compute rules,
including the formula/fallback components. Until that exists, 49.391% is a
valuable official-weight frontier comparison but is insufficient as evidence of
a resource-matched improvement threshold. Keep its frozen numeric result; do
not silently replace it with a weaker target.

Alternative: **improve an organizer-supplied trained reference**. Publish pinned
GLACIER, MIST and formula-stage weights equally to every agent, permit their
finetuning/distillation explicitly, and verify the starting package reproduces
49.391%. This can shorten iteration, but measures a different initialization
regime. It requires a new task/resource version and separate results. Merely
putting the weights on disk while retaining “none supplied” is not a valid fix.

| Resource | Agent view and use | Organizer responsibility |
| --- | --- | --- |
| Experimental train | Raw spectra, molecular structures, conditions; fitting | Verify split identity, preprocessing and checksums |
| Validation | Labeled full fold plus a fixed molecule-stratified quick subset | Same Hit@k definitions; report domain strata; forbid fitting to validation |
| Synthetic train | Existing sanitized SQLite store, reader and examples | Record generator/provenance and held-out filtering; keep optional |
| GLACIER supervision | Train-only MAGMa fragments/breakage targets and builder | Freeze construction parameters, measure preprocessing cost, audit identities |
| Hard negatives | On-demand train-pool or permitted graph-derived decoys | Distinguish train target fitting from validation/test candidate inputs; freeze any external molecule corpus separately |
| Research code | Writable pinned-source copy, task adapter, tiny/short/full configs | Working `train`, `resume`, `evaluate`, `publish` paths; original licenses |
| Runtime | Tested offline development and verifier environments | Pin compatible compiled wheels and CUDA; verify the submitted package's actual interpreter |
| Starting weights | None in the recommended scratch protocol | If choosing the alternative, version and declare all weights explicitly |
| Feedback | Full/quick validation scores, runtime, memory and fixed reference reports | No private labels, answer-keyed source tables or test-ranking cache in the agent view |
| Research record | Hypothesis, changed factor, validation delta, cost, checkpoint hash | Preserve it and the training recipe separately from the inference artifact |

Do not require a particular research discovery in the prompt. Candidate choices
include condition coverage, spectrum/molecule representation, objectives,
train-only contrastive learning, and inference efficiency. A few discriminating
experiments plus validation-based selection should be possible before the final
training/export reserve is consumed.

## Budget, prompt and completion design

First measure `T_prepare`, `T_smoke`, `T_train`, `T_val_quick`, `T_val_full`,
`T_export` and `T_replay` on the intended hardware, including large molecules,
candidate-pool sizes and unsupported/missing conditions. CPU preprocessing,
host RAM and disk throughput matter as well as GPU execution. Reusable,
train-only preparation may be organizer work before the clock starts if it is
supplied identically to every run and its cost is reported separately.

Freeze a budget that can accommodate several short trials, at least one
substantial alternative and a final selected run. A proposed planning envelope
is 10% setup/first valid package, 50% experiments, 20% final selection/training
and 20% export/replay/recovery. These fractions are design targets, not measured
feasibility. Record active and allocated GPU-hours, elapsed time, agent-model
tokens/cost, and interruptions; retries and resumes share the same budget.

Tell the agent the resource/deadline envelope and require an early valid package
plus periodic publication. Hiding the deadline is an experimental ablation,
not the default benchmark condition. If studied, freeze prompt variants before
the campaign; do not change only unsuccessful runs to demand more submissions.
Clock enforcement, model-provider access, session capture and generic snapshot
collection belong to the runner. Scientific data preparation, model code,
delivery contract and evaluation belong to this task.

Acceptance order:

1. **Offline starter gate:** a fresh image and only declared mounts support a
   tiny real-data fit, resume, validation score and portable publication.
2. **Reference gate:** the chosen initialization protocol has a measured reference
   and full validation result; every baseline asset is accounted for.
3. **Verifier gate:** real Docker/Harbor execution, genuine privilege separation,
   no label/network access during inference, full relocated replay within the
   five-hour inference/six-hour total budget, and deliberate missing/unknown-ID
   failure checks. Apptainer host execution alone does not establish this gate.
4. **Cold-start usability gate:** an agent given only the release can train,
   modify, evaluate and submit without undocumented operator interventions.
5. **Research discrimination gate:** predeclare several independent pilots
   (for example two agents × three seeds), retain every outcome, and report
   completion rate, best-validation-selected final score, uncertainty and cost.
   This small pilot diagnoses usability and floor/ceiling problems; it does not
   establish a statistically powered ranking of agent systems.

Publish a completion table with method, initialization, data/recipe version,
validation score, full-test Hit@1/5/20, MCES, domain breakdown, runtime, memory,
checkpoint/package digest and run/session link. Include the matched reference
and the agent in the same table. Preserve full training scripts and configs,
not only the final inference code. Do not require an agent to beat SOTA before
its submission counts as completed; valid submission and scientific improvement
are separate outcomes.

## Statistical and identity checks

Keep the official spectrum-weighted Hit@1 objective. For scientific claims,
also pair models at the molecule level: resample molecule groups with
replacement, retain all spectra within each sampled group, and divide the
paired hit-count difference by the sampled spectrum count. This preserves the
micro-average estimand while accounting for within-molecule dependence. It
still does not represent training-seed uncertainty or all chemical-family
dependence.

The existing query-paired 99.9% interval for hybrid minus clean MIST is
[+8.345, +10.686] points. An additional fixed-model, 100,000-resample,
seed-0 molecule-cluster diagnostic gives **[+5.751, +13.438] points**. The
gain remains positive; the wider interval illustrates why the query-level
test alone is not sufficient for a new-molecule claim. This diagnostic does not
alter the frozen grader or retrospectively select a different model.

Identity convention needs an explicit lock: upstream supplied keys count 2,998
test structures; RDKit 2023.9.6 recomputation from the actual SMILES counts
2,997 groups. The diagnostic above uses the latter, in sorted group order.
All experimental folds remain disjoint after recomputation (zero train/val,
train/test and val/test overlap). The recomputed train/val counts are
22,742/3,184 versus supplied 22,746/3,185. These are identity-convention
differences, not evidence of experimental leakage. A follow-up audit now
recomputes all 435,076 distinct synthetic SMILES with RDKit 2023.9.6, covering
2,333,485 spectra and 435,064 connectivity groups. All synthetic keys agree
with their supplied connectivity keys; zero synthetic molecules/spectra overlap
either recomputed validation or test groups. Experimental train/validation/test
also remain pairwise disjoint. The 1,045 synthetic structures overlapping
experimental training are allowed augmentation, not held-out leakage. Raw input
hashes, counts and the 38.451-second audit are in `release_validation.json`.

## Original audit scope (before implementation and deployment probes)

Fifteen existing tests pass in the local PyTorch environment. A first attempt in
the MCES-only environment failed because it lacked torch, then passed in
`_massspecgym_repro/venv_rtx`. No Docker executable is available on this host;
fresh Docker acceptance, GPU replay timing, new training and autonomous-agent
pilots were not run. The checks do not certify the shipped image.

Decision at that audit stage: retain this task as a benchmark candidate and retain the
verified 49.391% result. Close resource/initialization, trainable-starter and
portable-verifier gaps before declaring a research-ready release. Benchmark
agent comparisons remain unvalidated until pilots provide evidence.
