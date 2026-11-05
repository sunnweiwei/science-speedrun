# Agent integration validation, 2026-10-14

## Scope

The scientific reference source is zifeng/drugclip-reference-reset at
52c1c56a9bd058f5ec92d87c8be3682de7ef4b61. The integration follows
TASK_STANDARD.md and protein-folding-prediction's prompt/build/mount shape.
It retains the reference model, pretrained policy, data population, split,
and raw EF1% diagnostic. Current primary scoring is dataset-balanced
BEDROC(alpha=80.5), version drugclip-dataset-bedroc/v2; older validation
entries below retain their original metric definitions. Active branch/PR state is recorded by GitHub;
each acceptance record must pin its own actual checked-out commit.

This document separates CPU integration evidence, reproducibility cleanup,
permutation/handoff fixtures, and the later operator-reported A100 scientific
verification. The GPU result does not establish current Harbor/agent acceptance.

## Initial CPU integration checks

The following commands ran separately to avoid the existing starters' generic
data/model module-name collisions. Python 3.12, CPU PyTorch 2.7.1,
NumPy 2.5.4, RDKit 2026.9.1, and LMDB 3.0.0 were used locally. The images
pin their own dependencies; no container build is claimed.

    pytest -q tests/test_agent_task.py environment/evaluation/tests
    # 11 passed, 3 skipped
    pytest -q tests/test_data_pipeline.py tests/test_hf_release.py
    # 11 passed; existing synthetic materialization and HF release checks
    pytest -q environment/starter/tests
    # 1 passed
    pytest -q environment/starter/reference/drugclip/tests/test_starter.py
    # 3 passed
    pytest -q environment/starter/reference/drugclip/tests/test_science_reference.py -k 'not two_process_gloo_stop_broadcast'
    # 14 passed, 1 deselected

New integration checks use real synthetic LMDBs and cover:

- exact field whitelisting with coordinate/order preservation and label removal;
- all-target/ligand coverage, duplicated targets/indices, invalid paths and NaNs;
- expected ROC-AUC/BEDROC over tied-score permutations and EF1%=1 for constants;
- atomic package replacement, checksum rejection, and retention of the current
  package after invalid publication;
- complete 117-target grader control flow, inference error and timeout handling
  with privilege hooks mocked explicitly;
- infrastructure failures producing no reward and removing a stale reward;
- actual minimal-starter CPU training, model export, standalone inference on
  label-free records, and scoring across both benchmark-shaped fixtures;
- controller view staging, allowlisting, immutable source-copy separation,
  excluded benchmark weights/metadata, and synthetic-release rejection.

The actual unprivileged replay tests remain enabled for a normal root verifier
container but skipped here: /proc/self/uid_map and gid_map map only UID/GID 0;
chown to 65534 fails with EINVAL. The mocked control-flow checks do not prove
privilege isolation. Running the original two-process Gloo test also failed
because this host forbids its TCP device; the focused reference run excludes it.

TOML and all three compose files parse. Both main-service build contexts resolve
to the task root; every Docker COPY source exists. Agent/verifier compose files
do not set network/GPU policies, and task.toml sets no-network on both sides.
bash -n and git diff --check pass for the affected scripts and patch.

## Reproducibility cleanup

Staged development/private receipts now include file paths, byte sizes,
SHA256 hashes, and a sorted-manifest digest. A controller-owned runtime
report records observed package versions and GPU visibility from each actual
container, separate from host versions. See REPRODUCIBILITY.md for the
run record and RUN_HANDOFF.md for the required controller sequence.

Focused cleanup checks passed:

    pytest -q tests/test_agent_task.py -k 'controller_views or runtime_report'
    # 2 passed, 12 deselected

The view test checks every recorded file size/hash and the file-manifest
digest for both views, plus source-copy separation and excluded data.
The runtime-report test executes the recorder, checks its identifiers and
observed-version fields, and verifies it does not dump environment variables.
Python compilation, Docker COPY-source checks, and git diff --check pass.
No container image or real GPU/agent run is inferred from these checks.

## Private permutation and controller handoff fix (historical)

A synthetic 200-record fixture with the first 20 labeled active yields
ROC-AUC=1 and EF1%=10 for score=-ligand_index. A constant predictor receives
ROC-AUC=0.5 and EF1%=1. Field redaction and expected tied-score metrics alone
did not prevent this index-only ranking exploit.

This demonstrates the vulnerability under actives-first ordering. The actual
released DUD-E/LIT-PCBA LMDB order has not been audited. Benchmark acceptance
still requires a real-data audit. The verifier now creates an independent
private permutation for each target, supplies freshly numbered label-free
records and maps output indices back to private labels. Its root-only output
retains the 32-byte seed, NumPy/algorithm identity, index maps and checksums;
The root-only details.json records the private manifest's SHA256, while
the original reward.json contained only the numeric Harbor reward. See
[ACCEPTANCE.md](ACCEPTANCE.md) for the current output format. Neither exposes the seed
or mapping. Original LMDBs are unchanged. Corrupt mappings are infrastructure
failures.

The following are historical commands and results from that revision. The
standalone controller and its dedicated `test_handoff.py` were retired on
2026-11-05; current session orchestration uses the shared Harbor runner.
The private-permutation and privilege-separation tests remain in `tests/`.

Affected suites at that revision passed:

    pytest -q tests/test_handoff.py tests/test_agent_task.py environment/evaluation/tests
    # 26 passed, 3 skipped (host UID/GID limitation, unchanged)
    pytest -q environment/starter/reference/drugclip/tests/test_science_reference.py -k 'not two_process_gloo_stop_broadcast'
    # 15 passed, 1 deselected (host TCP limitation, unchanged)

New permutation checks cover whitelist preservation, every copied record's
view-to-original mapping, exact correct-label scoring, private map permissions
and absence from inputs, seed repeatability/independence, and corrupt/nonbijective
mapping rejection. Across 32 fixed private-seed trials, index-only predictions
on an actives-first fixture return near-random mean AUC/EF1%, while true feature
predictions retain perfect scores. This is not a real-release order audit.

Reference molecule inference conformer/crop seeding now depends on configured
seed and SMILES, independent of temporary path or shuffled index. A dedicated
test checks all feature tensors under different paths and positions. Training
and pocket cropping retain their prior recipes; historical metrics are not
new-code reruns.

The retired controller/run.py implemented owned Docker environment creation, a wall-clock
driver deadline, bounded TERM grace, complete container stop, validated
read-only snapshot, changed-file detection, independent verifier budget,
failure statuses and scoped cleanup. Handoff fixtures run real foreground
drivers and a detached TERM-aware trainer, preserve resume state, replay the
frozen package after removing the live directory, and exercise expired budget,
driver error, partial publication, missing/bad candidates, unchanged/changed
fallbacks and verifier failure/timeout. Docker command checks cover ownership
labels, no-network, resource flags and private/development mount separation.
The container adapter is a fixture, so real Docker cgroup/GPU behavior remains
unverified in that fixture run. The former standalone reference driver was
not a Harbor lifecycle hook or a Codex adapter. Current run instructions are
in [RUN_HANDOFF.md](RUN_HANDOFF.md).

## Standard review and deliberate limits

- Prompt: all five required sections, explicit offline rules, one portable
  continuously updated model, and no development GPU/time budget.
- Workspace: every named path is copied or mounted. Generic initialization
  assets are prepared controller-side and checksum-pinned.
- Baseline: runnable DrugCLIP source, short training/publishing commands, and
  RESULTS.md. The minimal interface example is under references/.
- Verifier: separate image, label-free staged inference, root-only private data,
  explicit errors, normalized reward plus all raw metrics.
- Developer README: all eight sections, operator variables, materialization
  commands and limitations. Root readiness row added.

RESULTS.md now separates the operator-reported complete-data A100 reference
(full-suite EF1% 16.108) from the historical one-target pilot and author-checkpoint
evaluator control. The reported recipe and full dev curve are committed under
baseline/drugclip/reproduction/. Raw logs/checkpoint hashes and current isolated
verifier acceptance were not supplied and are not claimed by this update.

## A100 training and held-out verification reported 2026-10-15

Zifeng Wang supplied the PR #94 paper-scale baseline's GPU training and
corrected-inference evaluation. Data is pinned to the compact HF release
`zifeng-ai/target-based-drug-screening` at
`c8d2febc996e4c96b10f3aecd89fb7375150449f` (also now the downloader default).
Python 3.11 / PyTorch 2.6.0+cu124 were used, rather than the current declared
PyTorch 2.7.1 / CUDA 12.6 images.

- 95,305,418-parameter dual Uni-Mol, generic molecule initialization and a
  randomly initialized pocket encoder.
- ProFSA: 500,000 pairs, 100 completed epochs, 8 A100s, 19.0 h, per-GPU
  batch 48, LR 1e-4, molecule backbone frozen.
- BioLiP: 4 A100s, 19.6 h, batch 48 per GPU, LR 1e-3, BEDROC(alpha=80.5)
  checkpoint selection and patience 20. Stopped at epoch 34; selected epoch 14
  at development BEDROC 0.4288.
- The same epoch-14 model scored **16.108 equal-target mean EF1%** over all
  117 targets after correcting pocket element tokens, the 511-atom test crop,
  DUD-E median/MAD then max pooling, and LIT-PCBA max pooling. DUD-E EF1% is
  18.056; LIT-PCBA EF1% is 2.861. The earlier 1.830 is superseded.

The [reproduction record](../environment/baseline/drugclip/reproduction/README.md)
contains the reported recipe, benchmark metrics, and all 34 development values.
This closes the absence of a reported complete-data trained-reference/full-suite
score. It also provides the immutable release for the data-lock review item.
It does not close the missing development-receipt mount, malformed-artifact
classification, task-specific runner deviation, or real container/UID acceptance.
No raw training log, checkpoint hash, exact code commit/command/seed, per-target
metrics, or inference runtime was provided; these are not invented here.

The recording update checked the 34-row curve, epoch-14 maximum, the 20-epoch
patience interval, target-count-weighted EF1% rounding, JSON/TOML and local
record links. The existing HF release tests pass with the pinned default:

    PYTHONPATH=environment/data pytest -q tests/test_hf_release.py
    # 4 passed (mocked Hub operations; no dataset download)

`git diff --check` passes. These checks validate the record and release lock;
they do not rerun GPU training or container replay.

## Outstanding acceptance

- Resolve the current PR's merge conflict before main integration.
- Audit real released input order; the private permutation/mapping fix is implemented.
- Build both CUDA images and run Harbor with the real staged release.
- Exercise real UID/GID 65534 inference isolation and label-read denial.
- Repeat two-process distributed checks on a host supporting loopback sockets.
- Exercise the implemented controller's deadline/cleanup/snapshot/replay on real Docker/GPU.
- Connect and run a real agent adapter, or integrate equivalent behavior into Harbor's lifecycle.
- Verify full-suite wall time and memory against the one-H100 replay limits.
- Attach the reported A100 run's raw logs, commands/seed, code commit,
  checkpoint hash/location and evaluation outputs when available; replay the
  same checkpoint through the current private-permutation verifier.
- Fix the development-receipt mount and malformed-artifact classification
  findings, and resolve the task-specific runner format deviation.

The track remains integration-ready for review, not fully verified.
