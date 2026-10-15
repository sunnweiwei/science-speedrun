# protein-sequence-modeling: validation record

Moved from the README when the task adopted TASK_STANDARD.md. Paths below predate the move
(`/data` is now `/workspace/data`, `environment/starter` is now `environment/baseline/starter`).

## Readiness

Verified on real data:

- All eleven archives download and their **md5 checksums match** the pinned
  values; `materialize.sh` converts all 33 splits (292,017 rows) and every
  measured count matches the pinned expectations.
- Label isolation holds: the verifier view carries inputs only, labels go to
  the separate private view.
- Class cardinalities checked against the data (`fold` = 1,195 classes, all
  present; `subcellular_localization` = 10).
- Grader unit tests pass (`tests/run_unit_tests.sh`, 8 tests), and a synthetic
  end-to-end run — starter inference, then `evaluation/score.py` — produces
  sensible trivial-predictor scores and zeroes exactly the affected task when
  predictions are deliberately corrupted.
- The corpus builder ran end to end against the real UniRef50 release, at three
  scales. Decontamination was checked by positive control — two verbatim eval
  sequences planted into a corpus were both removed — and the removal rate is
  stable across a 25× size change (0.128% → 0.122%), so it reflects the corpus
  rather than sampling noise.
- Two full reference baselines trained and scored under the task's own budget;
  see below.
- The agent's whole path was exercised inside a built container: every file
  `instruction.md` points at exists, network egress is blocked, the starter
  trains, checkpoints, resumes, publishes atomically, runs standalone
  inference, and self-scores with zero contract violations.

### Smoke run

The [CONTRIBUTING smoke run](../CONTRIBUTING.md#the-smoke-run) completed, all
six steps. Config: one A100-80GB, **4m32s wall**, a 9,409,569-parameter
6-layer d=320 encoder trained from random initialization — 3,000 MLM steps on
the UniRef50 subset then 4,000 multi-task steps on `fit` (batch 32, seq ≤512,
bf16, seed 0). The package was relocated to a fresh directory, run through its
own `infer.sh`, sealed, and graded by `tests/grader.py`.

| | trained | trivial | delta |
| --- | --- | --- | --- |
| binary_localization | 0.7404 | 0.5752 | +0.1652 |
| subcellular_localization | 0.4443 | 0.2914 | +0.1529 |
| yeast_ppi | 0.5431 | 0.4695 | +0.0736 |
| fold | 0.1003 | 0.0279 | +0.0724 |
| solubility | 0.5638 | 0.4997 | +0.0640 |
| secondary_structure | 0.4889 | 0.4306 | +0.0582 |
| fluorescence | 0.0563 | 0.0000 | +0.0563 |
| human_ppi | 0.5316 | 0.4979 | +0.0338 |
| stability | 0.0039 | 0.0000 | +0.0039 |
| beta_lactamase | 0.0000 | 0.0000 | +0.0000 |
| ppi_affinity | 0.3365 | 0.3688 | −0.0322 |
| **primary reward** | **0.3463** | **0.2874** | **+0.0589** |

**The metric measures quality, not compliance**: the trained model beats the
fixed-statistic trivial predictor on 9 of 11 tasks and on the primary reward.
Note how much the trivial predictor scores on its own (0.2874) — constant
majority-class output collects most of the accuracy-based tasks, which is why
any reported number needs this control beside it.

Deliberate breakage was confirmed to fail loudly and locally: dropping one
prediction (`solubility`), renaming one `example_id` to an unknown value
(`fold`), and emitting a negative score (`binary_localization`) each zeroed
exactly that task with an `invalid_reason`, and left the other eight rewards
bit-identical.

These are plumbing evidence, not benchmark results — the model is far too
briefly trained to mean anything.

### Reference baseline

Two full references have been trained and scored under the task's own budget —
one A100, 8 h of masked-language pretraining then 3 h of joint supervised
training, with model selection on the labeled `tune` split. Full provenance is
in [environment/references/measured_runs.md](environment/references/measured_runs.md).

| | architecture | backbone | held-out reward |
| --- | --- | --- | --- |
| run 1 | generic pre-LN transformer | 37.3M | 0.6147 |
| **run 2** | **ESM-2 (`esm2_t6_8M`)** | **7.4M** | **0.6368** |
| run 2 reproduced (H100, UniRef50 2026_03, seeds 0 / 1) | ESM-2 (`esm2_t6_8M`) | 7.4M | 0.6695 / 0.6774 |
| control | fixed-statistic predictor | — | 0.2874 |

**The 7.4M ESM-2 beats the 37.3M generic transformer**, on 7 of 11 tasks and on
the aggregate, which is the clearest signal here that architecture matters more
than width at this scale. Run 2 is the number to beat, and it is a lower bound:
it carried a learning-rate scheduling defect (since fixed) that left the rate at
exactly zero for a third of pretraining. The two independent reproductions of
the same time-budgeted recipe on an H100 (23.0 B pretraining tokens each,
packages replayed through the isolated verifier) land at 0.6695 and 0.6774.

**All eleven tasks are discriminative** — every one beats the trivial control,
settling a question the smoke run left open about `beta_lactamase` and
`ppi_affinity`.

`fold` remains hardest by a distance: 0.2019 over 1,195 classes, against 0.65+
for most other tasks. It has the most headroom and is the task most likely to
separate serious attempts.

Both runs used a 5M-sequence sample. The full 37.3M-sequence corpus is built
and shipped but no reference has been trained on it yet.

### Container verification

No Docker daemon was available, so `tests/Dockerfile` was mirrored into an
Apptainer definition and built (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
base, ~10 min, 4.0 GB image). What that established:

- The Dockerfile's contents are valid: `useradd --create-home --uid 10001
  agent` succeeds, all ten `tests/` files land in `/tests`, `test.sh` is 0755,
  and the base image supplies Python 3.11.13 with torch 2.8.0+cu128 and the
  intended offline environment (`PIP_NO_INDEX`, `HF_HUB_OFFLINE`).
- With the compose mount layout — `/benchmark` read-only, `/sealed` and
  `/logs/verifier` writable — and networking disabled, **network egress is
  blocked** and **the inference stage sees no labels**: verifier rows carry
  exactly `example_id`, `features`, `task`.
- `main_verifier.py`'s input-manifest and submission-package validation both
  pass against the real data and the trained smoke package.

**What it could not establish, and why:** running the submission as the
unprivileged `agent` user needs real root in the container. This host has no
`/etc/subuid` allocation for the user and Apptainer is installed
unprivileged-only, so no genuine uid mapping is available. The verifier
correctly **refuses to proceed** rather than degrading — it now raises a
precise error saying it must run as root under a normal Docker daemon.

Attempting this surfaced a real defect, since fixed. Under Apptainer's fallback
`--fakeroot` (an `LD_PRELOAD` shim, not a uid mapping), `os.setuid()` returns
success and `geteuid()` reports the new uid **while the process keeps full
privileges** — verified directly: it could still read `/etc/shadow` afterwards.
The old `demote()` took the drop on faith, so in such an environment it would
have run untrusted submission code as root and looked healthy. It now verifies
the drop took effect and that root cannot be regained; the new check was
confirmed to reject exactly that deceptive case.

Closing this gap needs either a normal Docker daemon or an `/etc/subuid`
allocation for the operator.

Remaining gaps, honestly:

- **The privilege separation remains unverified.** See the container section
  below for exactly how far the verifier was exercised and where it stops.
- **Harbor's claude-code harness is verified except the authenticated
  conversation.** An install-only run detected the CLI in the environment
  (agent version recorded), and a deliberate invalid-key run showed the full
  launch path working: the instruction delivered verbatim as the first
  message, the real CLI started in `/workspace`, the API reached from inside
  the environment (401 + retries), and the native session collected and
  converted to an ATIF trajectory. Only a run with a valid `ANTHROPIC_API_KEY`
  (or `claude setup-token` OAuth token) remains. Everything around it has now been exercised for real: a cold-context
  LLM agent, given only `instruction.md` and this environment, followed the
  documented workflow end to end — read the reading list, probed hardware,
  published a replay-valid candidate early, kept research state out of the
  submission, and handed off — surfacing two starter defects that are now
  fixed (the `metrics` import broke in trainers copied out of
  `/workspace/baseline/starter`, and checkpoints were step-cadence only, losing a phase
  to a kill on slow hardware). Its candidate scored 0.2125 held-out — below
  the trivial control, as expected from minutes of CPU compute — so the test
  validates documentation and tooling, not achievable score. Harbor's own
  orchestration has likewise been exercised — `harbor run` with the `nop` and `oracle` agents on
  the Singularity backend parsed this `task.toml`, started the environment
  image, executed the oracle's `solve.sh` (which published the trivial
  candidate through `starter/publish.py`), and collected `/workspace/submission`
  as an artifact that then scored exactly the trivial control's 0.2874 under
  the grader. What that does not test is whether `instruction.md` and the docs
  are sufficient for an agent working cold; that needs a `claude-code` (or
  similar) run with an API key. Verification was disabled in these runs: the
  in-Harbor verifier phase needs real Docker (root, genuine setuid, compose
  data mounts), which the Singularity backend cannot provide.
- The published reference is run 2 below, not a literature number. See
  "Comparability with the PEER paper" for why the paper's results cannot fill
  that role.
- **No reference has been trained on the full corpus yet.** Both runs used a
  5M-sequence sample; the full 37.3M-sequence corpus is built and shipped but
  unexercised, so its practical effect on scores is unmeasured.
- The pretraining subset requires an ~8.2 GiB UniRef50 download; the current
  release is not a versioned URL, so builds are pinned by the recorded
  SHA-256 rather than a release string.

