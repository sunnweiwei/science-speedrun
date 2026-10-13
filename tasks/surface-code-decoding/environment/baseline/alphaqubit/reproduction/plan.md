# Sycamore baseline reproduction contract

## User authorization and scope

2026-09-19: “用cs的cluster来做baseline复现吧。你尽可能制定个最多h200 x 8 x 2d 能完成的（当然少了肯定也行）。你先复现baseline吧。这里没有h200，但可以用其他卡。开始吧，时间 不重要。”

Latest steering (2026-09-19 local): “是啊。复现个牛的直接。只要代价别太离谱，就搞！”
The primary objective is now an independent, faithful reproduction of
**AlphaQubit (Nature 2024)** on Sycamore d3/d5, using its published architecture
and training protocol. QRENND is a secondary comparison, not the target method.
Implement from the paper and supplement, audit against primary sources, profile
actual GPU cost, train, and evaluate portable weights. Distinguish a single-seed,
single-fold reproduction from the paper's multi-seed ensemble and full d3–11
Pauli+ study. Do not claim official code or checkpoint provenance.
PyMatching standard/correlated and the released TN/BP predictions remain controls.

## Resources and isolation

- Isolated worktree: `ai4sci-tasks-qec`, branch `codex/surface-code-baseline`.
- Persistent assets: `/home/users/xy200/yinxunjian/mycode/ai4sci-artifacts/surface-code-decoding/20260919`.
- CS allocation only; do not modify or cancel unrelated user jobs.
- Conservative actual GPU accounting ceiling: **384 allocated GPU-hours**,
  **at most eight concurrent GPUs**. No invented H200-equivalence conversion.
- Resource allocation is governed by the shared ledger and the AlphaQubit
  pivot allocation below; the initial QRENND partition is retired.
- Training jobs run through Slurm, write resumable checkpoints and machine-readable
  progress, and have hard per-job time limits. Queue time is not GPU compute.
- No wall-clock deadline. Reserve time and budget for independent replay,
  statistically adequate scoring, review, artifact provenance and reporting.
- AlphaQubit pivot: prior campaign consumed approximately 145.9 GPU-hours when
  stopped; exact final usage is in the shared accounting ledger. Preserve all
  existing checkpoints, cancel unstarted QRENND work, and allocate the remaining
  approximately 238 hours to AlphaQubit. Initial allocation: <=8 hours for
  implementation/throughput/overfit GPU probes; hold >=32 hours for validation,
  adaptation, replay and contingency. Set full training caps only after measured
  throughput and learning evidence. The 384-hour total includes earlier QRENND
  work and is a conservative physical-card budget, not an H200 conversion.

## Frozen data and evaluation

Zenodo 6804040, CC-BY-4.0, archive MD5
`a7fd8b481c3087090093106382dc217d`, 315490804 bytes. All 130 surface-code
conditions, 6.5M shots: four d3 patches and one d5 patch, X/Z basis, odd rounds
1 through 25. Do not include repetition-code data in the aggregate.

Within each condition, even indices are development data: first 19880 even
shots training, remaining 5120 validation. All 25000 odd shots are test.
Published even-fitted DEMs include validation syndromes, but no validation
logical labels; disclose this. Never select weights/hyperparameters from odd
labels or stored test predictions. Model inputs exclude logical labels and
raw final data-bit parities. Existing public test data is not a blind dataset.

Synthetic pretraining uses the released noisy circuits or corresponding DEMs,
independent fixed seeds and matched initial/temporal conventions. A model
trained/fine-tuned under our fixed split must be labeled as such, not assigned
paper scores or claimed identical to the publication's protocol.

Primary scientific metric: fitted logical error per round, independently by
patch and basis across rounds 3–25 (exclude r1 boundary effect). Report group
values, fit quality, equal-group distance aggregates, error suppression,
uncertainty and per-condition shot failures. Probabilities additionally allow
NLL/calibration; offline batch throughput is not streaming latency.

## Live hypotheses and falsifiers

H1: A faithful AlphaQubit recurrent Transformer can learn Sycamore d3/d5 decoding
within the remaining compute budget. Falsifier: adequately trained models plateau
below useful decoding after verified architecture/input/objective checks; or a
measured learning/throughput curve exceeds the resource ceiling. A tiny pilot is
not a test of ultimate accuracy. Do not substitute a generic Transformer or LSTM
while retaining the AlphaQubit name.

H2: Fine-tuning on allowed even hardware shots can close the sim-to-real gap
relative to the pretraining-only reference. Falsifier: independently selected
validation checkpoints and held-out paired scoring show no robust improvement.
Preserve and report both stages. For AlphaQubit, hardware fine-tuning is part of
the published method; the fixed split, sample budget, seed count and ensembling
must still be reported explicitly.

H3: Classical evaluation is cheap enough that neural training dominates cost.
Evidence ladder: full baseline replay and detailed job time records.

## Evidence ladder and pivot gates

1. Verify source pin, architecture/config differences, dataset checksum and
   detector-to-neural-input mapping; test real labels against Stim conversion.
2. CPU unit/contract checks, then one GPU forward/backward and tiny overfit.
3. Full-size JAX/Haiku GPU throughput pilot on both distances; choose batch/job
   budgets from measured throughput and memory. Use BF16 computation with
   float32 master weights, verified against FP32 and finite-gradient checks.
4. Train a representative d3 and d5 pair; diagnose sample efficiency before
   committing the entire array. Train across all ten patch/basis groups.
5. Select on validation, freeze packages, evaluate all held-out shots once;
   compare paired errors, rerun relocated inference and audit selected weights.
6. Preserve logs/weights outside Git; publish task-scoped code/provenance/results
   and an editable HTML report, with all reproduction deviations disclosed.

AlphaQubit allocation gate: full-size d3/d5 forward/backward, tiny overfit,
source audit, data isolation, optimizer/resume tests, and portable replay must
pass. Start with two representative 2-hour training allocations. Inspect complete
full-noise validation across all thirteen durations and real learning curves
before starting the remaining groups. The subsequent allocations may be queued
behind a strict CPU gate; dependency failure cancels unstarted work. The gate
independently reconstructs and hashes the full-noise validation data, scores
standard MWPM on it, and verifies finalized checkpoint/selection identities.
For each representative it requires at least10,000 updates,4M examples and
three complete validation snapshots, an accepted selected fit, at least10%
shot-error reduction over no correction, and selected LER no more than twice
MWPM LER. In addition, main BCE must improve by10% over the first complete
snapshot, or selected LER must already be within1.25× MWPM. These thresholds
justify a larger training allocation only; they do not establish convergence
or published accuracy. Failure preserves budget for diagnosis and a possible
representative extension. A failed LER fit is not a valid score,
and a short run is not evidence against the model's eventual accuracy. Preserve
validation losses and checkpoint rejection reasons for diagnosis.

If GPU memory/compatibility blocks the preferred RTX PRO 6000, profile a supported
alternative with smaller microbatches and equivalent accumulated gradients.
If the learning curve is too slow, audit inputs, curriculum, optimization and
reconstruction choices before spending the training reserve. Do not silently
substitute a different architecture or claim original-paper equivalence.

## Current AlphaQubit decisions and claims

### PR handoff and supplementary completion verified 2026-09-22 05:01 UTC

Continue the existing PR24 on `codex/surface-code-baseline`, with an English
description covering data, baseline, compute and proposed RSI delivery. Merge
the current main branch, preserving both independently added README task rows.
The fixed v1 reference is recorded in `alphaqubit-summary.json` with evidence
hashes; model/data assets remain outside Git.

Supplementary d5 train/replay array12679372 completed both30k-step runs.
Score12679373 wrote complete valid scores, then failed an overly strict
post-score check: shared bootstrap RNG draws changed d3 CI endpoints when d5
counts changed, despite unchanged d3 predictions, counts and point estimates.
The repaired guard checks hashes, all d3 condition results and deterministic
macro fields while retaining recomputed CIs. Actual-artifact checks passed and
mutated d3 point/count controls were rejected. CPU scoring repair12681093 then
completed0:0. Preserve both the original failure and repair in the ledger.

The supplementary d5 LER is0.032185776622954604, a small improvement over v1
0.032288885908884055. V1 remains the accepted benchmark reference. Total actual
allocation is350.048611 GPU-hours (3.711389 for the supplementary run), with
zero active GPUs and33.951389 remaining. No further training is planned.

### Benchmark reference accepted 2026-09-22

Latest user steering: “没事的。我觉得差不多可以了。所以整理下，我们的benchmark中，
这个task的 数据集（train test validation之类的）；复现的baseline是什么（model
结构 大小之类的）；训练时长；输入给rsi agent是什么、提交是什么；之类的；告诉我吧”.

Treat the audited v1 as the fixed benchmark reference. Reaching the paper's
headline accuracy is no longer a prerequisite for this deliverable. The already
running six-GPU-hour-bounded d5 continuation remains a supplementary experiment;
do not use its test score to replace the accepted reference or launch further
accuracy campaigns. The current deliverable is the benchmark specification in
`benchmark-spec.md` and an editable external HTML summary, not a claim that
Harbor packaging is already implemented.

The proposed agent track supplies allowed train/validation data, simulator,
source and pretraining/fine-tuning weights for model development. Its portable
contract accepts arbitrary batches and opaque IDs on the known geometries.
Propose balanced per-shot logical failure as the ranking metric, with fitted
LER retained as the scientific metric; the former avoids rewarding pathological
free-intercept fits. This proposal is derived from fixed v1 counts and does not
alter its checkpoint selection or reproduction scores. Physical split isolation,
the generic adapter and a separate real Harbor verifier remain packaging work.

### First full test completed 2026-09-22 02:16 UTC

All ten pretraining and fine-tuning groups, package freeze, relocated test replay
and scoring completed with Slurm exit 0:0. Both stages cover all 130 conditions
and 3,250,000 odd test shots. An independent audit reconstructed every prediction
count from the original labels, checkpoint/input/prediction identities, all group
and macro fits, and all paired classical bootstrap intervals. Evidence is
`alphaqubit-v1/verification/final-score-independent-audit-20260922.json`.

| Test LER per round | d3 | d5 |
| --- | --- | --- |
| Pretraining | 0.03105183 | 0.03366771 |
| Fine-tuning | 0.03012653 | 0.03228889 |
| Fine-tuning 95% shot-bootstrap CI | [0.02995034, 0.03027945] | [0.03195628, 0.03259113] |
| Released tensor network, same odd test and fit | 0.03026392 | 0.02908539 |
| Official AlphaQubit paper reference | 0.02901 | 0.02748 |

The fine-tuned model beats standard and correlated MWPM at both distances. It
slightly improves over released TN at d3, but remains worse than TN and belief
matching at d5. The numerical gaps to the paper references are +3.85% and +17.50%
relative, with different fold/ensemble protocols; these are descriptive gaps,
not a matched-protocol statistical comparison. Error suppression is 0.9330,
95% CI [0.9227, 0.9441], so the paper's suppression result is not reproduced.
The initial training/evaluation campaign is complete; official-level accuracy
across both distances remains unachieved. Do not describe the task as fully solved.

Actual total use is 346.337222 GPU-hours, of which 145.901944 belongs to the
earlier LSTM campaign. No original-chain GPUs remain active; 37.662778 GPU-hours
remain. Preserve this v1 result, selected weights, raw probabilities and report.

Next bounded experiment: finish the originally declared 30,000 fine-tuning
updates for both d5 groups in a separate `alphaqubit-d5-30k` directory. Their
two-hour stages stopped at 15,042 and 14,949 updates; X selected its final step,
and Z last improved at 12,461. This is development evidence for a bounded
continuation, not evidence that more updates will close the gap. Copy exact
optimizer/RNG/EMA/anchor state, retain the same LR, architecture and validation
selection, and cap each new allocation at three hours including replay (six
GPU-hours total). Cumulative stage limit is 4.5 hours or 30,000 updates. All d3
predictions remain unchanged. The v1 test has already been inspected; report
the continuation as a post-test follow-up on the same public fold, not a fresh
confirmation, and never select weights from test performance. This extension
leaves at least 31.66 actual GPU-hours for diagnosis and validation; it does not
authorize silently exhausting the reserve on repeated test-driven tuning.

The follow-up was submitted as GPU array12679372 (tasks8/9), scoring12679373
and terminal status12679374. Both GPU tasks started at 02:36 UTC; no additional
queue wait is currently needed. Exact cloned states, retained d3 predictions,
decision and submission identities are recorded under `alphaqubit-d5-30k/`.
The shared ledger ceiling is now365.036944 GPU-hours, below384. At the measured
previous d5 throughput the extension should take roughly two to three hours
including inference; this estimates runtime, not achievement of paper accuracy.

### Status verified 2026-09-21 20:22 UTC

All ten pretraining groups completed with Slurm exit 0:0. Hardware fine-tuning
has completed two groups, with three running and five queued for priority.
Finalized weights, source/data identities and recomputed selected validation
fits passed for all ten pretraining and both completed fine-tuning stages;
evidence is `alphaqubit-v1/verification/completed-stages-audit-20260921T2022.json`.
The completed d3/X/center(3,5) and center(5,3) hardware validation LERs are
0.03142795 and 0.03199257; the running center(5,7) selected validation LER is
0.02785890. These are selected checkpoints, not necessarily the final update.
No d5 hardware validation or held-out AlphaQubit score is available yet.

The paper reports aggregate experimental LER 0.02901 for d3 and 0.02748 for d5
(Nature 2024, Fig. 3). Our partial per-patch validation values cannot establish
parity with those full experimental test aggregates. This reproduction uses one
seed and one fold, with approximately 158–253 million synthetic examples per
model, versus up to two billion and a 20-model ensemble in the paper. Ensembling
accounts for only 0.0003/0.0008 absolute LER improvement at d3/d5 in the paper;
it cannot explain an arbitrarily large gap. Full frozen odd-shot evaluation
and paired classical comparisons remain required before judging success.

Actual allocation at 20:20:59 UTC is 331.564167 GPU-hours, including the earlier
145.901944 for LSTM. There are three active GPUs and 52.435833 GPU-hours remain
under the total limit. The committed ceiling stays at 359.036944. Queue delays
prevent a firm completion time; the earlier 7–9-hour estimate assumed no added
queue delay. Freeze, replay, scoring and reporting dependencies remain queued.
The external accounting and editable HTML report were refreshed.

### Status verified 2026-09-21 14:03 UTC

Eight of ten pretraining groups completed their cumulative 18-hour stages with
Slurm exit 0:0. Their finite FP32 selected weights, source/data identity,
finalized metadata and accepted validation fits were independently checked;
evidence is retained in `alphaqubit-v1/verification/completed-pretrain-audit-20260921.json`.
The remaining d3/Z/center(7,5) and d5/Z/center(5,5) groups are running at about
15.1 and 15.0 hours, with approximately three hours left each. Fine-tuning,
freeze, odd-test replay and scoring still await the full pretraining array.
No experimental held-out AlphaQubit score is available. The representative
selected synthetic validation LER remains 0.0226949 for d3/X and 0.0318215 for d5/X.

Actual allocation is 320.0825 GPU-hours, including 174.180556 for AlphaQubit
and the earlier 145.901944 for LSTM. Two GPUs are currently active; 63.9175
GPU-hours remain under the task limit. The committed ceiling is unchanged at
359.036944. With no additional queue delay, the remaining pretraining, two
fine-tuning waves and replay/scoring are estimated to take roughly 7–9 hours.
The external accounting and editable HTML report were refreshed.

### Status verified 2026-09-20 15:35 UTC

Both representative runs completed successfully, and expansion gate 12649332
passed at 04:39 UTC. Eight full-size models are now training on eight typed
RTX PRO 6000 GPUs; two additional groups are pending the array concurrency
limit. Active models have run approximately 10.1–11.7 cumulative hours and seen
101.7–160.6 million simulation examples each. Fine-tuning and odd-test replay
remain pending; no experimental held-out AlphaQubit score exists yet.

On the same fixed synthetic validation populations used by the verified gate,
the representative d3/X/center(3,5) selected LER is 0.0226949 versus standard
MWPM 0.0345093 (34.24% relative reduction); d5/X/center(5,5) is 0.0318215 versus
MWPM 0.0479207 (33.60%). These are validation-selected point estimates, not
hardware test results or proof of matching the published paper. Current fits
were independently recomputed from selected per-condition error counts.

One group, d3/X/center(7,5), currently retains an earlier accepted checkpoint:
newer fits have intercept below the paper's selection cutoff. No selection
rule was relaxed; the raw newer fits and rejection reasons remain in history.

Actual allocated use at this snapshot is 231.879444 GPU-hours, including the
145.901944-hour retired LSTM campaign (AlphaQubit portion 85.9775). Eight GPUs
are active. The committed ceiling remains 359.036944; the total budget remains
384. The external accounting and editable HTML evidence report were refreshed.

### Earlier implementation and submission evidence

- Primary sources are the Nature article, its architecture pseudocode and
  supplementary information. Local source hashes and page renders are saved in
  the external references directory. Community repositories were reviewed but
  their code was not copied; missing licenses and method differences ruled them
  out as the primary implementation.
- Independent reconstruction uses the published 320-dimensional, three-block
  Sycamore recurrent Transformer and convolutional/readout structure. Unspecified
  details and our choices are listed in `alphaqubit-source-audit.md`.
- AlphaQubit predicts official path-observable flips directly. The historical
  QRENND global-data-parity conversion does not apply to this model.
- The QRENND chain is retired. Its actual consumed cost was 145.901944 GPU-hours.
  Eight active runs were interrupted by a Slurm signal that reached the batch
  shell; they did not complete gracefully. All sixteen existing saved weights
  were checked and remain valid, but recent uncheckpointed updates were lost.
  `native-v1/pivot-stop.json` is the independent termination audit.
- FP32 full-size d3/d5 probes passed on RTX PRO 6000 Blackwell Max-Q. BF16 batch256,
  25-round probes measured 2442.40/1526.06 samples per second and peak memory
  9.61/24.46 GB. These are bounded steady-state optimizer probes, not real-time
  decoder latency or end-to-end campaign throughput.
- A fixed d3, three-round, batch128 diagnostic reached 2/128 main errors and
  main BCE 0.093509 after 88 updates. This only establishes trainability on the
  diagnostic batch; it does not establish held-out decoding quality.
- Training uses streamed even-fitted pij noise, LAMB, curriculum, EMA, and fitted
  validation LER selection. The batch-switch counter defaults to the literal
  optimizer-step reading of Extended Data Fig.8; the noise curriculum counter
  is examples seen. All schedule choices enter the resume identity.
- Integrated independent checks passed96 tests in the JAX environment; the
  skipped legacy TensorFlow inference test passed with the retained environment
  (all33 prediction-contract tests passed). Relocation and a fresh locked
  environment reproduced66,560 synthetic validation probabilities exactly.
- Representative jobs submitted:12649316, array0/8 with at most two RTX PRO6000
  GPUs, cumulative2h/model and2h15m hard per-allocation cap (4.5 GPU-hours total
  reserved). Current semantic source hashes are recorded with the submission.
  Later stages are prepared as a dependency chain behind the expansion gate.
  Their original hard allocation caps are counted even while dependencies wait.
- Settled actual use before representative jobs:146.036944 GPU-hours, including
  0.135 AlphaQubit probe GPU-hours; remaining237.963056.
- Planned AlphaQubit ceiling: representative2×2.25h; eight other groups
  8×18.25h; two representative resumes2×16.25h (18h cumulative/model);
  hardware fine-tuning10×2.25h; relocated test replay10×0.75h. Together with
  settled prior use, the maximum committed ceiling is359.036944 GPU-hours,
  leaving24.963056 uncommitted. Full pretraining arrays use max6 plus max2
  concurrent GPUs, fine-tuning/replay each max8, with strict stage dependencies.
  A terminal CPU status job runs even if a prerequisite fails.
- Submitted chain: representatives12649316 → gate12649332 → other groups
  12649333 + representative resumes12649334 → fine-tuning12649335 →
  freeze12649336 → replay12649337 → score12649338 → terminal status12649339.
  Slurm's actual typed-GPU requests, pending states and dependencies were
  independently checked and saved in alphaqubit-v1/submission-audit.json.
- The gate preserves versioned portable representative weights and validation
  history before full resumes can overwrite those artifacts. Gate-source
  amendment after submission is recorded in the ledger; model/data/trainer
  sources are unchanged and the gate is still pending its prerequisites.
- Current result scope: single seed and even-to-odd fold, d3/d5 hardware data.
  There are no trained AlphaQubit held-out scores yet. Pauli+ d7–11, soft readout,
  twenty-seed selection and the paper's ensemble are outside the first result.

## Historical QRENND decisions (retained audit; not the active procedure)

- Initial CS inspection: A5000 nodes available; RTX PRO 6000 nodes present;
  no H200. Current CPU node belongs to existing job 12473780.
- Source mirror pin: `c2ed73e3ec6ee843293c8839bf059c63b2960e61`.
- Data moved from temporary audit storage to persistent artifact directory;
  archive checksum verified before extraction.
- Source audit found bundled d5 width128/batch64 vs paper width253/batch256;
  use paper values, retain exact upstream model implementation and document
  discrepancy.
- Preserve QRENND's native global-data-parity target for both heads. An exact
  GF(2) certificate derived from ideal circuit annotations converts official
  path labels for training and converts probabilities back for evaluation.
  Verified independently against raw hardware measurement flips and synthetic
  data across all 130 conditions. Missing initial off-basis detectors are still
  zero-padded, an explicit input-convention adaptation.
- Full classical replay completed in 393.3 s on two allocated CPU cores, job
  12642029. Validation selected even-calibrated DEMs for both matching methods.
  Test macro LER d3/d5: MWPM 0.038780/0.043167; correlated 0.034162/0.034503;
  released TN predictions 0.030264/0.029085. These are our specified fits, not
  copied paper headline numbers. Counts and paired errors are saved.
- All ten fixed caches (280M training shots total) completed and were upgraded
  to provenance schema v2 only after payload hashes and deterministic first-block
  sampling checks against their source circuits. Validation uses separate seeds.
- Native model/pipeline tests: 74 passed before GPU pilots. A5000 runtime check
  passed; the subsequent A5000 pilot was pending and cancelled without compute
  (12642088). Resubmitted two native pilots on available V100s (12642089).
- Native pilots completed and independent full-validation CPU replay matched
  error counts exactly. Tiny-overfit gate passed after 325 updates. Real pilot
  packaging, fresh-environment detector-only replay and trusted validation
  scoring passed, explicitly returning unfit LER for the weak pilot weights.
- Expanded suite: 105 tests pass. Representative training is job 12642101
  (two V100s, two hours each). Full training is conditionally queued behind a
  gate requiring completed representative stages, at least one full epoch,
  finite complete synthetic validation, 5% combined-BCE improvement over the
  throughput pilot and selected validation shot error below 46%. This gate
  only justifies further training allocation, not convergence or test quality.
- Full chain IDs: gate 12642244; pretrain other groups 12642245 (max6 concurrent)
  and representative resumes 12642246 (max2); fine-tune 12642247 (max8); freeze
  12642248; replay 12642249 (max8); score 12642250; terminal report 12642251.
  Pretrain cap is cumulative 24h/model, including the representative sessions;
  fine-tune is 2h or 100 epochs. Including allocation overhead and diagnostics,
  all submitted caps sum to 274.5 GPU-hours, leaving 109.5 hours uncommitted.
- Generic GPU requests exclude mixed GPU nodes so the a5000/v100 node feature
  constraint cannot accidentally allocate a P100 or RTX2080 on those nodes.
  Typed V100 representative resumes can still use mixed-node V100 slots.

## Completion gate

Actual trained weights, full held-out scores, raw evidence, compute accounting,
reproduction limitations, portable inference replay, and independently reviewed
scoring are required. Submission of Slurm jobs is progress, not completion.
