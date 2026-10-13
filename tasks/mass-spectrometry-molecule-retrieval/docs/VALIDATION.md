# MassSpecGym: readiness and release acceptance

Moved from the README when the task adopted TASK_STANDARD.md. Paths below predate the move
(`/data` is now `/workspace/data`, `environment/starter` is now `environment/baseline/starter`,
`environment/reference` is now `environment/references`).

## Layout check after TASK_STANDARD.md (2026-09-27)

On one host, with rebuilt development and verifier images and the task's own compose files:

- the development container mounted the six `development/*` entries read-only under
  `/workspace/data` and an empty `/workspace/research`; the starter smoke trained, published
  atomically into `/workspace/submission`, relocated and replayed on staged validation queries;
- the verifier (`tests/docker-compose.yaml`, main + grader, CPU) replayed that smoke package on all
  17,556 test queries in 414 s: 17,556 valid rankings, 0 invalid, Hit@1 0.48 % (a few-step smoke
  model), `reward.json` `{"reward": 0.0}` (the scores were normalized then; Hit@1 was below the lower anchor);
- 47 unit tests pass (`tests/run_unit_tests.sh`).

## Current readiness

Version 1.3.0 adds prepared HF downloads and reference-normalized scoring to
the native GLACIER research starter, source/runtime build,
train/validation-only fragment preparation, resumable training, continuous
publication and validation diagnostics. No pretrained weights are supplied.
Fresh native runtime installation, CPU/GPU training probes, parameter-identical
CPU recovery and input-only namespace replay have been exercised. These checks
support a controlled pilot. The reviewer timing above adds single-H100 replay
evidence; acceptance of the current Docker/HF path remains pending. A cold CPU
agent pilot has completed full independent replay; the native scratch reference
has completed training and validation,
and full test scoring (40.322397% Hit@1). The generic agent-runner supplies GPU integration
and controls development API access through Harbor's allowlist, following the
clinical tasks' launch pattern. Actual runner staging and Harbor service
selection pass for 0/1/8 GPU allowlist and no-network profiles; both verifier
services remain offline. Production Docker/API transport and resource enforcement
still require deployment evidence; the reviewer timing does not certify those paths.
See [the agent evaluation protocol](AGENT_EVALUATION.md),
[implementation evidence](release_validation.json), and the
[original audit with implementation status](BENCHMARK_READINESS.md).

- Task design, pinned data interface, materializer, starter package, formal
  GLACIER/MIST hybrid lock, isolated verifier, and synthetic unit tests are
  included.
- The upstream files are not committed to Git.
- Real pinned-source materialization completed on 2026-09-01: all 231,104 rows
  were processed into a 533 MiB public view and a 3.7 MiB private-label view;
  test coverage was 17,556 queries and 3,170 deduplicated candidate pools.
- The original fifteen synthetic interface/security/statistical tests passed;
  the current regression suite and added acceptance checks are recorded in
  `release_validation.json`. A CPU plumbing run trained 128
  real spectra for one epoch, selected on 64 validation spectra, and published a
  relocatable 7.2 MiB package. It ranked all candidates for eight separately
  staged real validation queries; the deliberately undertrained model obtained
  Hit@1 0, Hit@5 0, and Hit@20 0.125. These are pipeline diagnostics, not a
  scientific baseline result.
- The historical September 2 full 50-epoch starter training completed on one RTX A5000 in 11 minutes 9
  seconds. Its complete operator-side test replay produced valid rankings for
  all 17,556 spectra and obtained Hit@1 5.126%, Hit@5 12.406%, and Hit@20
  27.108%, with MCES@1 22.185; see `starter/README.md` for hashes and
  validation results.
- The released MIST-CF-to-MIST reference replay ranked all 17,556
  test spectra and obtained Hit@1 30.257%, Hit@5 40.288%, Hit@20 54.130%, and
  MCES@1 17.719. Its ranking SHA-256 is
  `dbc135bcba3852f43839e1b16afdecd97db9a6e1669850a5b1f98c16828cc572`.
- The strict-clean 150-epoch MIST training completed on one RTX PRO 6000 in
  18:54:08. Validation selected epoch 128 at loss 0.2493356764; the exported
  28.5M-parameter encoder SHA-256 is
  `30715dbd198bb1d0d776a5dcb9d37488b8d5480c243a1572d62ed27b73e92c6b`,
  and all 59 exported tensors exactly match the selected EMA checkpoint.
- The former formal clean top-five-formula/max replay obtained Hit@1 39.872%, Hit@5
  51.487%, Hit@20 63.141%, and MCES@1 15.167. Its ranking SHA-256 is
  `78a781884f442039c66963eeab7187f27645440e644d3088f85dbb81e8fd7f3d`.
  The same clean encoder's top-one replay obtained 34.000% Hit@1. A 100,000
  resample query-paired bootstrap places the top-five Hit@1 gain at +5.873
  points, with a 99.9% interval of [+4.859, +6.892].
- The official GLACIER no-CF checkpoint was replayed on exactly 9,954 eligible
  test queries and combined with clean MIST on the other 7,602. The complete
  hybrid obtains Hit@1 49.391%, Hit@5 64.189%, Hit@20 74.436%, and MCES@1
  12.179. Its ranking SHA-256 is
  `b703afcdd0b543486d54b36b79e096de3259b25dbf0e29ebfb876b5d19e7a8c6`;
  an independent rescore of cached simulations is byte-identical. The
  constant-spectrum hybrid falls to 3.139% Hit@1.
- The agent-visible synthetic augmentation was independently materialized in
  19 minutes 45 seconds. It contains exactly 2,333,485 labels and matching
  SQLite rows, passes database integrity and checksum verification, and has
  zero 2D-structure overlap with validation or test. Its database SHA-256 is
  `d0ede6aa7162e4d12dcc2b7166e6d4b7f545d00ee047e4b816edd20234e62d77`.
- Replacing every MS/MS spectrum with the same constant two-peak input drops
  the clean top-five reference to Hit@1 4.386%, Hit@5 7.593%, Hit@20 16.615%,
  and MCES@1 25.429. This negative control rules out a high score driven only by
  precursor metadata or candidate-pool artifacts.
- Ignoring spectra entirely and preserving each answer-independent candidate
  order gives Hit@1 0.256%, Hit@5 2.820%, Hit@20 10.338%, and MCES@1 31.347.
  This additional null control rules out candidate position as an answer cue.
- The MIST constant-spectrum control, independent full training, reference
  packaging, and isolated-container replay are tracked in
  `reference/REPRODUCTION.md`; the GLACIER method replay, validation-first gate,
  resource failures/recovery, controls, and final promotion decision are in
  `reference/GLACIER_REPRODUCTION.md`.

## Materialized release acceptance (2026-09-22)

The public 38.0 GB HF release and separate private grader labels were downloaded
into a fresh directory and verified against every pinned payload hash and split
count. The default materializer uses these prepared files, with source rebuilding
retained explicitly. [Release evidence](release_acceptance.json) records the pins,
full frozen-hybrid rescoring, and actual full-test DeepSets inference in an
input-only offline Apptainer container. Scratch-hybrid Hit@1 remains 40.322397%,
normalized score 0.8002008032. All DeepSets scientific metrics and top-20 rankings
match the historical replay; 28 deeper rankings differ starting at rank 184
under a different PyTorch runtime. This is not a claim of complete-file bitwise
identity, new GLACIER inference, or a new Docker/H100 certification.
