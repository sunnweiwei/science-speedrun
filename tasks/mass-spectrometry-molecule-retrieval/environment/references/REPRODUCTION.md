# MIST-CF -> MIST reproduction state

Last updated: 2026-09-03

## Research contract

- **Question:** Can the public, data-safe MIST-CF -> MIST method be trained and
  replayed under this task's formula-free MassSpecGym v1.5 contract?
- **Contribution:** A task-owned, relocatable two-stage baseline with source and
  data pins, training commands, checkpoints, complete validation/test rankings,
  metrics, and leakage checks. A paper number without runnable artifacts does
  not satisfy the contract.
- **Audience:** ai4sci-tasks contributors and agents that must improve over a
  strong, established molecular-retrieval baseline.
- **Hard constraints:** Train learned components only on the official train
  fold; never expose true validation/test formulae; never formula-filter the
  supplied mass candidate pool; score every query on a fixed denominator;
  package offline inference behind `infer.sh INPUT_ROOT OUTPUT_ROOT`.
- **Compute envelope:** Up to eight GPUs for 48 hours. Reserve at least 25% of
  available wall time for full replay, leakage/shortcut checks, packaging, and
  rerun validation. Record actual GPU-hours and peak memory.

## Live hypotheses and falsifiers

1. **H1: exact audit replay.** The pinned audit/FRIGID integration plus public
   data are sufficient to recover approximately 26.64% test Hit@1.
   **Falsifier:** required model state, generated formula candidates, or hidden
   preprocessing cannot be reconstructed from public artifacts, or a faithful
   replay differs materially after controlled debugging.
2. **H2: method-level reproduction.** If an exact artifact replay is impossible,
   retraining public MIST-CF and strengthened MIST from their pinned recipes will
   produce a valid, strong task-owned baseline, even if its point estimate is
   different. **Falsifier:** it does not clearly exceed the reproduced DeepSets
   starter or fails a leakage/constant-spectrum control.
3. **H3: bounded fallback.** A public MIST-family checkpoint that is demonstrably
   train-fold safe can establish a runnable lower-bound baseline while the full
   synthetic-augmentation run is diagnosed. **Falsifier:** provenance or fold
   overlap cannot be established.

## Evidence ladder

1. Pin and inventory code, data, checkpoints, licenses, environments, and
   upstream commands.
2. Reproduce preprocessing on small train/validation slices; audit structure
   and formula boundaries.
3. Run inference with released components on a validation slice and verify the
   ranking contract and fixed-denominator scoring.
4. Train MIST-CF and report validation formula accuracy/coverage.
5. Train or faithfully replay MIST and report validation retrieval metrics.
6. Run all 17,556 test spectra, constant-spectrum and metadata/candidate-order
   controls, plus at least one rerun/seed or a justified deterministic replay.
7. Publish the relocatable package, hashes, environment lock, logs, and metrics.

## Pivot triggers

- If exact upstream assets remain unavailable after source/data inventory,
  switch from "exact audit replay" to an explicitly named method-level
  reproduction; do not present the audit point estimate as local.
- If synthetic-data preparation would consume the validation reserve, first
  establish a train-only experimental-data reproduction and then add synthetic
  augmentation as a separately measured phase.
- If MIST-CF formula accuracy is poor, stop downstream full training and audit
  SIRIUS decomposition, adduct handling, fast-filter construction, and split
  boundaries on a representative validation sample.
- If downstream retrieval does not beat DeepSets, test formula-oracle MIST only
  as a diagnostic; it must never be reported as the formula-free result.

## Decision and run log

- 2026-09-02: DeepSets is accepted only as the runnable starter. The user's
  baseline requirement makes a task-owned MIST-CF -> MIST run mandatory.
- 2026-09-02: Public code snapshots and artifact manifests were previously
  identified, but no local strong-baseline training result has yet been claimed.
- 2026-09-02: A clean-worktree patch replay found that an earlier edited
  MIST-CF compatibility patch had stale hunk counts and would be rejected by
  `git apply` on a new machine. The hunk was repaired and all three repository
  patches now apply in sequence from their exact pinned commits. The four
  resulting modified source files are SHA-256 locked in
  `environment-lock.json`, verified by `prepare_reference_assets.sh`, and
  covered by a patch-syntax/lock-linkage regression test.
- 2026-09-02: Contrary to the earlier inventory, the pinned Hugging Face data
  revision already contains both public train-fold MIST-CF checkpoints:
  `mist_cf_msg.ckpt` (SHA-256
  `53345744860a92ed2c3cb4a9326313d1d50d38011e178f33e02a12ac0dbea729`)
  and `mist_cf_fast_filter.ckpt` (SHA-256
  `7e26802251551a23904cd57f3f2f691dfd4aefc8fd4c8fba8e5ee2059c624791`).
  Both load strictly in the pinned original MIST-CF implementation after the
  import-only Lightning 2 compatibility patch in `patches/`.
- 2026-09-02: Downloaded and MD5-verified Zenodo record 15008938
  `msg_preprocessed.tar.gz`. Its 231,104 labels exactly match the v1.5 split
  counts (194,119/19,429/17,556). Extracted the official FRIGID `mist_msg.pt`
  encoder (SHA-256
  `09b4e93e8d5d35e7472f4d187d72ca316b125368232bb69148df101ec211b92a`);
  all 59 tensors load strictly into the pinned 28,545,444-parameter
  `SpectraEncoderGrowing` configuration.
- 2026-09-02: A one-query test-side end-to-end plumbing replay used a predicted
  (not oracle) formula and ranked the correct structure first among all 256
  unfiltered mass candidates. This is an interface check, not a scientific
  result.
- 2026-09-02: Extended the formula stage to consume the benchmark's unlabelled
  `queries.jsonl` directly. Its one-query output is byte-identical to labelled
  TSV audit mode, while reporting no formula diagnostic because no answer is
  present. `run_reference.sh` now composes this path with MIST and emits only a
  full contract ranking; `prepare_reference_assets.sh` verifies all external
  inference assets before use.
- 2026-09-02: Generated the full 17,556-row constant-spectrum control input by
  replacing every MS/MS spectrum with the same two positive peaks at 25 and 50
  m/z (SHA-256
  `819f66721d326bd04edf4ff8d3e6df0efe255ba6071d70a650b05aea68c71ffc`).
  Its two-stage replay is queued after the main run and uses the same mass,
  adduct, candidates, and weights.
- 2026-09-02: Started the complete 17,556-query MIST-CF replay as eight
  deterministic precursor-group shards (Slurm array 12474815), with an
  after-success full MIST retrieval job (12474828). Candidate enumeration at
  10 ppm produced 3,700 unique precursor/adduct groups and retained the true
  formula for 80.9125% of test spectra after the public 256-formula fast
  filter; the cache SHA-256 is
  `63082deff45af13067a515f36e804196153ca6913705ac2b11e134f3ede266a9`.
- 2026-09-02: Built a separate Python 3.11 / PyTorch 2.7.1+cu128 environment
  for the cluster's sm_120 RTX PRO 6000. The exact MIST architecture, EMA,
  cosine schedule, checkpoint selection/reload, and test path passed a
  3-epoch/100-example smoke run (Slurm 12474838) in 181 seconds, with peak host
  RSS 1,350,016 KiB. The selected encoder SHA-256 is
  `059c11c8e961088467e493db45df65a2645bccf378b5e4fa88ca9a3b3bb2ac4d`.
  The synthetic training archive subsequently downloaded and passed its
  published MD5 check.
- 2026-09-02: Full MIST-CF inference completed for every test query (Slurm
  array 12474815, eight RTX A5000 GPUs). The slowest shard took 1:38:05 and
  used approximately 13.1 GiB peak host RSS. The released train-fold model
  obtained 50.1937% formula Hit@1, 70.5058% Hit@5, and 80.9125% candidate
  coverage. The merged prediction SHA-256 is
  `0d3f7d90594e2f0bede41908445771940be6c7f03bb946867e530fd8e6fe12bd`.
- 2026-09-02: An initial MIST replay loaded all weights strictly but scored
  only 22.59%--23.32% Hit@1 across diagnostic Tanimoto variants. The cause was
  not the checkpoint: the hand-built inference constructor had inherited the
  model class's `set_pooling="intensity"` default, whereas the training CLI
  supplies `set_pooling="cls"`. Both modes have identical parameter shapes, so
  strict state loading cannot detect this error. The constructor now pins CLS
  pooling and the unit suite asserts the setting.
- 2026-09-02: Recalibrating the corrected released MIST checkpoint on all
  19,429 visible validation spectra gave a binary-fingerprint threshold of
  `0.20346222817897797`; predicted and reference on-bit density both round to
  0.0155492. The threshold report SHA-256 is
  `d822fca7558ae7826782f7fabd62c5f84d3c5ac36cc06b240d58afdb6f9e47b8`.
- 2026-09-02: Corrected MIST-CF-top-1 -> MIST replay completed in the task's
  exact anonymous candidate layout. All 17,556 rankings are valid. Hit@1 is
  0.3025746184, Hit@5 is 0.4028822055, Hit@20 is 0.5412964229, and MCES@1 is
  17.7194976077. The ranking SHA-256 is
  `dbc135bcba3852f43839e1b16afdecd97db9a6e1669850a5b1f98c16828cc572`;
  corrected query-fingerprint cache SHA-256 is
  `e96263dbcdad9709be51644c64574d23386a65997380210c34fb21749436a544`.
  Operator-only ground-truth-formula diagnosis reached 59.4270% Hit@1. A
  continuous-cosine diagnostic reached 31.3397% Hit@1, but it is not used for
  the frozen reference because the paper specifies thresholded Tanimoto for
  this pipeline.
- 2026-09-02: The exact released-weight replay is stronger than the paper's
  26.64% Hit@1 point estimate. That number is retained as a literature
  comparison, while 30.2574618% became the executable, hash-bound released-work
  comparator during clean retraining. Exact
  paper prediction files remain unavailable publicly (MassSpecGym issue 70),
  so the two point estimates must not be presented as identical artifacts.
- 2026-09-02: Direct extraction of 2.33 million synthetic JSON files exposed a
  shared-filesystem directory limit and unacceptable metadata latency. A
  lossless streaming packer now stores the same `spec -> JSON payload` mapping
  in a read-only SQLite table on node-local storage. A source patch changes
  only lookup storage; a one-record equality/featurization smoke test passed.
- 2026-09-02: The held-out overlap audit found a release-data defect rather than
  a false positive: `msg_aug_filt.tar.gz` contains 10 spectra of validation
  structure `CNZIQHGDUXRUJS` and 5 spectra of test structure
  `YTZALCGQUPRCGW`. The first formal training launch (12475232) therefore
  stopped before optimization. A deterministic sanitation step now removes
  exactly those 15 label rows and SQLite payloads, then re-audits all retained
  2D keys. A compatibility-only correction changed the obsolete Lightning
  seed call to the current top-level API after job 12475234 stopped before its
  first optimizer step. Job 12475237 then exposed the invalid-SMILES defect
  described below, also before optimization. The complete clean seed-1,
  150-epoch task-owned MIST training run is Slurm job 12475256 on one RTX PRO
  6000. Its completed result is recorded below.
  Because the released encoder was trained from the nominally filtered archive,
  its 30.2575% replay is labeled as the released-work result, not asserted to be
  strictly held-out-clean; the completed sanitized result recorded below
  determines the final clean reference.
- 2026-09-02: Job 12475237 passed the structure-overlap gate but stopped before
  its first optimizer step because the pinned synthetic corpus also contains 10
  spectra sharing one silicon-containing SMILES that RDKit 2023.09.6 cannot
  parse. The deterministic sanitation gate now validates each unique SMILES,
  removes these 10 rows and their SQLite payloads, and expects 2,333,485 clean
  simulation rows. This is a source-data compatibility defect, not stochastic
  sample dropping inside a data loader.
- 2026-09-02: The constant-spectrum negative control completed with the same
  MIST-CF and MIST weights, precursor metadata, and candidate pools. It obtained
  Hit@1 0.0230690, Hit@5 0.0689223, Hit@20 0.1924129, and MCES@1 26.9151572.
  Its ranking SHA-256 is
  `35c8d904cadc4d58c3c9cf702d67e3e32966ec094d1f561db4e936befb38b131`.
- 2026-09-02: Replaying the paper's second setting---the same released MIST
  encoder conditioned on the top five data-safe MIST-CF formula predictions,
  with an element-wise maximum across the five continuous fingerprints before
  thresholding---substantially improved the local result. All 17,556 rankings
  are valid; Hit@1 is 0.3567441331, Hit@5 is 0.4690704033, Hit@20 is
  0.5941558442, and MCES@1 is 15.7633002962. The ranking SHA-256 is
  `63f25cf84ddb0bdf2e19fe6a23c2121643d470a0020c82d4d296f25f6dd52468`.
  This agrees qualitatively with the paper's top-five improvement, but is not
  claimed to reproduce its 28.23% point estimate because the exact paper
  predictions remain unavailable. Since the released MIST encoder's nominally
  filtered augmentation contains the five test-overlap rows identified above,
  this result is recorded as a released-work comparison; the completed clean
  top-five replay recorded below determines the final benchmark target.
- 2026-09-02: The corresponding top-five constant-spectrum control obtained
  Hit@1 0.0240374, Hit@5 0.0615174, Hit@20 0.1664388, and MCES@1 26.2297505.
  Its ranking SHA-256 is
  `c5494f6946005448a398dd4f39ce9424beab8f35b7464de2558bf4574071b1d0`.
  The large real-to-constant gap remains intact under the stronger aggregation.
- 2026-09-02: A query-paired comparison confirms that the released top-five
  gain over top-one is not a point-estimate accident. Top-five alone is correct
  on 2,007 queries, top-one alone on 1,056, and both have the same correctness
  on 14,493. The Hit@1 delta is +0.0541695; a deterministic 100,000-resample
  paired percentile bootstrap gives the 99.9% interval
  [+0.0439166, +0.0643085].
- 2026-09-02: The normal agent-visible augmentation materializer completed an
  independent end-to-end run in 19:45 with approximately 1.9 GiB peak process
  RSS. It published 2,333,485 label rows and the same number of SQLite payloads,
  with database integrity `ok`, zero validation/test structure overlap, and
  post-publication database SHA-256
  `d0ede6aa7162e4d12dcc2b7166e6d4b7f545d00ee047e4b816edd20234e62d77`.
  This closes the data-access gap: agents receive the same strictly sanitized
  simulation corpus used by the task-owned clean retraining run.
- 2026-09-02: A source-level training-boundary audit confirmed that the MIST
  optimizer receives only `SpecDataModule.train`, while every saved candidate
  checkpoint is selected solely by `val_loss`. Cosine scheduling disables
  early stopping, so all 150 epochs run and validation only selects the final
  checkpoint. The upstream `trainer.test` call occurs after fitting and after
  the validation-best state is loaded; it computes a read-only oracle-formula
  loss but cannot select or update the model. The benchmark result is instead
  produced afterward from unlabelled queries with MIST-CF-predicted formulae.
- 2026-09-02: At epoch 10, the Slurm task cgroup reported
  `memory.peak = 205267886080` bytes (191.2 GiB). This is the relevant physical
  memory measurement; summed worker RSS double-counts fork-shared pages. The
  256 GiB agent allocation covers the observed upstream loader behavior with a
  useful margin.
- 2026-09-02: A candidate-order null model ignored spectra and emitted every
  pool in its published answer-independent order. All 17,556 rankings were
  valid; Hit@1 was 0.0025632, Hit@5 0.0281955, Hit@20 0.1033835, and MCES@1
  31.3470893. The ranking SHA-256 is
  `c2b58193ac4d5cef3dbc24748ce9e2c8b8176c6aefed4e1158ed75b92c71eb72`.
  This is far below both real-spectrum MIST settings and rules out candidate
  position as the source of their gains.
- 2026-09-03: Strict-clean seed-1 MIST training job 12475256 completed all 150
  epochs in 18:54:08 on one RTX PRO 6000 Max-Q. Validation selected epoch 128
  at loss 0.24933567643165588 and global step 1,222,791. The selected checkpoint
  is 114,278,572 bytes with SHA-256
  `ca6dc2a4c3e6bab47869bc40cc7a232cb776ec1d29c4bba11b6cd01369f8dd8d`.
  The exported 114,206,138-byte encoder has SHA-256
  `30715dbd198bb1d0d776a5dcb9d37488b8d5480c243a1572d62ed27b73e92c6b`;
  all 59 tensors exactly match the selected EMA checkpoint.
- 2026-09-03: Validation calibration of the clean encoder over all 19,429
  spectra fixed the binary fingerprint threshold at 0.22049389779567719. The
  threshold report SHA-256 is
  `8a3eba162b2289e6ee90f85e3c515028a6875f5fc21cfa864654fb03eeb037f7`.
  The same single threshold was reused for top-one, top-five, and both
  constant-spectrum controls.
- 2026-09-03: The clean top-one replay produced 17,556 valid full rankings and
  obtained Hit@1 0.3399977216, Hit@5 0.4431533379, Hit@20 0.5790612896, and
  MCES@1 17.4643141946. Ranking SHA-256 is
  `ad04ea5ea8b976a6eb1b35f8f49542f94e524618dfb4b374e44e389d214b8ff2`.
  Against released top-one its Hit@1 delta is +0.0374231 with a 100,000-resample
  99.9% paired interval of [+0.0290499, +0.0457963].
- 2026-09-03: The clean top-five/element-wise-max replay produced 17,556 valid
  full rankings and obtained Hit@1 0.3987240829, Hit@5 0.5148667122, Hit@20
  0.6314080656, and MCES@1 15.1668375484. Ranking SHA-256 is
  `78a781884f442039c66963eeab7187f27645440e644d3088f85dbb81e8fd7f3d`.
  It was the formal target until the later GLACIER hybrid replay documented in
  `GLACIER_REPRODUCTION.md` superseded it. Versus clean top-one, top-five alone
  is correct on
  2,058 queries, top-one alone on 1,027, and 14,471 tie. The Hit@1 delta is
  +0.0587264 with a deterministic 99.9% paired interval of
  [+0.0485874, +0.0689223]. Versus released top-five, the delta is +0.0419799
  with interval [+0.0325815, +0.0513784].
- 2026-09-03: Clean-model constant-spectrum controls retained the same formula
  predictions, precursor metadata, pools, weights, and threshold. Top-one fell
  to Hit@1 0.0317840 and MCES@1 28.4851902; top-five fell to Hit@1 0.0438596
  and MCES@1 25.4294543. Their ranking SHA-256 values are respectively
  `3d3ff20171207399f0dd78aa3ce51039c459d4fef3be0f289aeef0cb841792cc`
  and `ee0769a83db0124f5817d75cb1a470e2f96b7816667a3c3bde5289d50fdd4775`.
  The large real-to-constant gaps satisfy the shortcut gate.

## Claims ledger

| Claim | Required evidence | Current status |
| --- | --- | --- |
| DeepSets starter is reproduced | full test rankings and metrics | complete; see `../starter/commissioning.json` |
| MIST-CF is formula-data safe | split/provenance audit and test predictions | complete for released train-fold weights; 17,556-query replay hashed |
| Released MIST retrieval is reproducible | strict load, validation calibration, complete rankings | complete; top-1 30.2575% and top-5-max 35.6744% Hit@1, both ranking hashes recorded |
| MIST training is task-owned | train log, checkpoint, data audit, replay | complete; 150 epochs, selected EMA export and hashes recorded |
| Two-stage baseline is strong | complete metrics and constant-spectrum control | complete; clean top-five target is 39.8724% Hit@1 and control is 4.3860% |
| Result matches publication artifact | exact public predictions or controlled equality | numerically different; task-owned clean 39.8724% replay is the frozen target |
