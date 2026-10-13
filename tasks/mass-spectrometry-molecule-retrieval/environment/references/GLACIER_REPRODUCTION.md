# GLACIER reproduction state

Last updated: 2026-09-08

## Research contract

- **Central question:** Can the public GLACIER method be reproduced under this
  task's frozen, formula-free MassSpecGym v1.5 mass-retrieval contract and
  replace MIST-CF -> MIST as the benchmark baseline?
- **Intended contribution:** A task-owned, relocatable GLACIER baseline with
  immutable source/data/weight pins, an auditable train-only recipe, complete
  validation and 17,556-query test rankings, leakage and shortcut controls,
  measured runtime, and an offline reference runner over public contract input.
  A copied paper score or an upstream evaluator run is not sufficient.
- **Audience:** ai4sci-tasks contributors and research agents improving
  formula-free tandem-mass-spectrum molecular retrieval.
- **Quality bar:** The implementation must reproduce GLACIER at the method
  level; use the task's unchanged candidate pools and fixed denominator; beat
  the current clean MIST reference with a positive paired 99.9% interval; fit
  the operator construction envelope; disclose rather than assume single-H100
  verifier feasibility; and make every score traceable to pinned code, data,
  configuration, weights, rankings, and verifier output.
- **Hard constraints:** Learned components may update only from the official
  train fold and permitted train-structure-only auxiliary data. Validation is
  model selection only. Test labels, true test formulae, candidate order,
  PubChem popularity, and validation/test structures must not enter training,
  filtering, cache construction, or scoring. Every supplied mass-compatible
  candidate must be ranked. Inference must be offline and package-relative.
- **User instruction:** Reproduce GLACIER and make it the benchmark baseline.
- **Compute envelope:** Up to eight H100 GPUs for 48 hours, 32 CPU cores, and
  256 GiB RAM. Reserve at least 12 wall-clock hours and 25% of the available
  GPU-hours for full frozen replay, controls, packaging, and an independent
  rerun. Record actual GPU-hours, host memory, GPU memory, and inference time.

## Method and data lock

[GLACIER](https://arxiv.org/abs/2606.29161) is used as a forward spectrum
model, not an inverse fingerprint predictor. For each candidate molecule it
encodes the molecular graph with a five-layer Graphormer, conditions on adduct,
collision energy, and instrument, and uses a fixed set of 200 DETR-style
fragment queries to predict multi-bond fragmentation events and their peak
intensities in one pass. The released MassSpecGym configuration has hidden size
256, 200 query slots, MAGMa targets capped at six broken bonds, predicted
breakpoint cardinality from zero to three, four fragment-decoder layers, three
intensity-encoder layers, four intensity-decoder layers, and 150,000 bins over
0--1,500 Da. Retrieval ranks every candidate by entropy distance between its
simulated and observed square-root-intensity spectra at 0.01 Da resolution.

The base training objective combines a cosine spectrum loss with MAGMa-derived
fragment matching. The pinned recipe trains for 20--200 epochs with batch size
32, AdamW at learning rate `4e-4`, zero weight decay, 2,000 warmup steps, and validation-loss early
stopping. It initializes the Graphormer and prediction heads from scratch; it
does not use a generic pretrained molecular backbone. The optional second stage
contrastively finetunes against PubChem decoys with batch size 16, five sampled
decoys, entropy loss, and learning rate `2.7e-4`.

The public checkpoint replayed here is the base, non-contrastive model: its
stored hyperparameters are `learning_rate=0.0004`, epoch 23, and global step
74,352. It contains 15,108,154 parameters and strictly loads all 264 state
tensors into the pinned implementation. This work therefore reproduces the
released model and evaluator on the benchmark contract; it does **not** claim
that a new GLACIER checkpoint was trained independently.

The released GLACIER data table contains the 119,029
`simulation_challenge=True` rows with observable collision energy, instrument,
and `[M+H]+` adduct: 99,341 train, 9,734 validation, and 9,954 test spectra.
The official training program constructs gradient-updated data only from the
train indices and selects checkpoints by validation loss. MAGMa fragment labels
are supervision derived from each training molecule and its spectrum. The
contrastive PubChem map is irrelevant to the released non-contrastive weight.
The task adapter reads only public query metadata and candidate SMILES at
inference; it never reads target SMILES, target formulae, or labels.

The frozen benchmark is broader: it has 17,556 test spectra from all supported
MassSpecGym conditions. Consequently, the formal candidate must be an explicit
hybrid: GLACIER on its 9,954-query released domain and the existing strict-clean
MIST reference on the remaining 7,602 queries. Both branches rank the unchanged
mass-compatible candidate pools, and the branch predicate depends only on
public adduct/collision-energy/instrument fields.

## Live hypotheses and falsifiers

1. **H1: released-weight exact-contract replay.** The public MassSpecGym
   GLACIER checkpoint and pinned `coleygroup/ms-pred` code can score the task's
   anonymous mass candidates without answer information and recover a result
   near the paper's no-CF 69.69% Hit@1 on the released simulation subset.
   **Falsifier:** weights or preprocessing are
   unavailable, provenance cannot establish a train-only boundary, required
   features are absent, or complete frozen replay differs materially after
   controlled debugging.
2. **H2: official train-only method reproduction.** If a released checkpoint
   cannot serve as the final artifact, the official architecture can be
   trained from scratch using MassSpecGym train spectra, train-only MAGMa
   supervision, and fully documented train-only negatives within the compute
   budget. **Falsifier:** the paper result depends on unavailable/licensed data,
   validation/test structures, non-reconstructible labels, or training exceeds
   the budget without a representative learning signal.
3. **H3: the forward simulator fits the benchmark contract.** Deduplicated and
   condition-aware candidate-spectrum prediction can rank all candidates
   offline within the reference-construction envelope. Single-H100 verifier
   feasibility is a separate deployment measurement, not inferred from an
   eight-way construction run. **Falsifier:** exact conditioning prevents safe
   cache reuse or full replay exceeds the operator envelope.
4. **H4: GLACIER is a stronger baseline rather than a protocol artifact.** Its
   gain persists on the exact fixed denominator and collapses under a
   constant-spectrum control. **Falsifier:** the gain disappears under the
   frozen evaluator, relies on a different candidate protocol, or a
   spectrum-free/candidate-order control explains a substantial fraction.

## Evidence ladder

1. Pin and inventory the official paper, source commit, license, checkpoints,
   configurations, preprocessing, data dependencies, and published commands.
2. Audit split handling, collision-energy processing, MAGMa labels, PubChem
   negatives, and retrieval candidate construction at source level.
3. Materialize a small train/validation slice and run one candidate, one pool,
   and several-query end-to-end probes through the task contract.
4. Replay released weights on a representative validation subset; compare the
   upstream and task scorers and profile condition-aware inference.
5. Run complete validation, tune only permitted similarity/normalization
   choices, and freeze the inference configuration before test replay.
6. Run all 17,556 test queries, paired comparison with MIST, constant-spectrum
   and candidate-order controls, plus a deterministic rerun or hash equality.
7. If released weights fail provenance or accuracy, execute the bounded
   train-only reproduction and repeat steps 4--6.
8. Publish the relocatable package, exact hashes, environment lock, commands,
   logs, metrics, confidence intervals, and updated benchmark documentation.

## Pivot triggers

- If the public checkpoint lacks train-only provenance, use it only as a
  diagnostic and pivot to task-owned training; never make it the baseline.
- If the reported 69.95% uses a materially different MassSpecGym protocol,
  retain that number only as external literature context and define the local
  result exclusively from the frozen verifier.
- If collision-energy imputation touches validation/test spectra during model
  fitting, redesign it as train-fitted inference-time preprocessing or encode
  missingness directly before any scientific run.
- If full candidate simulation exceeds the verifier budget, test exact
  condition-aware caching, batched inference, and a safe two-stage reranker.
  Do not silently prune candidates.
- If a representative released-weight probe does not beat clean MIST, stop the
  full test run and audit binning, adducts, collision energy, spectrum
  normalization, checkpoint/config pairing, and candidate semantics.

## Resource ledger

| Phase | Planned maximum | Actual | Reserve decision |
| --- | ---: | ---: | --- |
| Inventory and audits | 4 h CPU | complete | no GPU |
| Slice preprocessing/replay | 8 GPU-h | about 0.18 P100 GPU-h plus one failed RTX launch | probe gate passed |
| Full candidate simulation | 64 GPU-h | about 32.9 aggregate P100 GPU-h across validation/test, retained partial work, recovery, and one cancelled retry | complete |
| Optional train-only reproduction | 192 GPU-h | not run | released weight passed the registered method-replay gate |
| Scoring, controls, packaging | 32 h CPU | complete frozen scoring plus deterministic rescore | no additional GPU |

## Decision and run log

- 2026-09-04: Opened a separate GLACIER reproduction track. The existing
  39.8724% clean MIST reference remains authoritative until GLACIER passes the
  complete frozen-contract and integrity gates.
- 2026-09-04: The paper reports 69.95%/86.47%/93.52% Recall@1/5/20 for the
  MassSpecGym mass challenge with contrastive finetuning, but this is registered
  only as an upstream claim pending local replay.
- 2026-09-07: Pinned upstream source commit
  `ed8311f22958cb37f055b663b5f56c5c77a2ee33` (MIT). Downloaded the public
  `GLACIER_checkpoint.zip` (181,602,669 bytes, SHA-256
  `7e0fa24a19163513fe7cbeea3bf51a48cdd33bf2dbe8ac7a0457cbbf28d1258b`)
  and extracted `best.ckpt` (181,602,423 bytes, SHA-256
  `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`).
  All 264 state tensors load into the current upstream `JointModel`; the model
  has 15,108,154 parameters. Its epoch/global-step and hyperparameters match
  the base MSG recipe (`epoch=23`, `global_step=74352`, `lr=0.0004`) rather
  than the published contrastive-finetuning recipe (`lr=0.00027`), so the
  released weight is provisionally classified as the paper's no-CF model.
- 2026-09-07: Found a protocol-scope mismatch that prevents importing the paper
  number as the full-task result. Upstream `data/spec_datasets/msg/labels.tsv`
  contains exactly 119,029 `simulation_challenge=True` spectra, all `[M+H]+`
  with known collision energy: 99,341 train, 9,734 validation, and 9,954 test.
  The frozen benchmark instead contains 231,104 spectra and evaluates 17,556
  test queries. Thus the released GLACIER domain covers 56.70% of the frozen
  test denominator; 7,602 test queries require an explicitly validated
  extension or fallback. The released instruments are Orbitrap and QTOF. The
  paper text names the full 231,104-row dataset but
  does not disclose this simulation-subset denominator.
- 2026-09-07: Added answer-free adapters from public contract JSONL to the
  upstream candidate table and from upstream PredSpecDB output back to full
  candidate rankings. A strict 10-query/2,560-candidate smoke run completed on
  one P100 in 65.57 seconds of program time (Slurm job 12540889); 2,552 valid
  candidate structures were simulated and eight RDKit/element-invalid
  candidates received deterministic worst scores. The first 10 eligible rows
  cover only two candidate pools and are not representative: entropy Hit@1 was
  2/10. A one-query-per-pool 100-query probe is running before any full replay.
- 2026-09-07: A first smoke launch on RTX PRO 6000 Blackwell failed before a
  model forward pass because upstream's pinned PyTorch 2.6/CUDA 12.4 wheels do
  not contain `sm_120` kernels. This is an environment-compatibility result,
  not a model failure; A5000/P100/H100 remain supported by the pinned wheel.
- 2026-09-07: The diverse validation probe completed 25,371 valid candidate
  simulations on one P100 in 542.13 seconds (Slurm elapsed 9:19, process
  `MaxRSS` 47,068,380 KiB). Under the frozen exact-candidate contract, entropy
  obtained 52/75/89% Hit@1/5/20 after reproducing the official observed-spectrum
  preprocessing; cosine obtained 45/68/83%. The earlier all-peak entropy result
  was 52/75/88%, so top-50 preprocessing does not explain the gap from the
  paper. Entropy is frozen for complete validation. The task's exact-SMILES,
  exact-InChIKey, and paper-style 2D-InChIKey metrics were identical on this
  probe; 93% of its pools nevertheless contain at least one 2D-key duplicate,
  so the distinction remains explicit in the complete audit.
- 2026-09-07: The registered constant-spectrum negative control on the same
  100-query probe collapsed entropy Hit@1/5/20 from 52/75/89% to 0/1/8%.
  This is diagnostic rather than final evidence, but it verifies the control
  path and shows that the probe gain is not reproduced from candidate metadata
  alone. The complete constant-spectrum replay is queued behind full inference.
- 2026-09-07: Reordered the same 100-query probe using the upstream
  answer-keyed candidate-table order and used a stable input-order tie break.
  It remained exactly 52/75/89% Hit@1/5/20, identical to the task's
  content-derived candidate-ID tie break. Candidate order therefore does not
  explain the probe result; the full task continues to use only the latter.
- 2026-09-07: Reconstructed the released domain exactly. Validation contains
  9,734 eligible rows. Test contains 9,954, not the 9,957 rows selected by only
  `[M+H]+` and finite collision energy: the additional three rows have missing
  instrument type and `simulation_challenge=False`. The adapter and hybrid
  selector require the released Orbitrap/QTOF vocabulary rather than silently
  mapping missing or unseen instruments to Orbitrap.
- 2026-09-07: Verified set equality rather than relying only on row counts: the
  9,954 public test queries selected by the adapter exactly equal the source
  TSV's `fold=test AND simulation_challenge=True` identifiers. The pinned
  upstream `msg/labels.tsv` has SHA-256
  `0b29f4b150fa043f2ea382df3d7a02ddfdf5dc9baee9790299b3890d3582f244`,
  119,029 unique spectra, no missing collision energies, 81,323 Orbitrap rows,
  and 37,706 QTOF rows.
- 2026-09-07: Audited the raw v1.5 split independently of model inference. It
  contains 194,119/19,429/17,556 train/validation/test spectra and
  22,746/3,185/2,998 unique 2D structures, with zero pairwise 2D-structure
  overlap. The public-field GLACIER predicate exactly reproduces all 119,029
  `simulation_challenge=True` identifiers. The pinned upstream training entry
  constructs its optimization DataLoader from `train_df` and its selection
  DataLoader from `val_df`; this source audit cannot independently attest the
  history of a third-party released checkpoint, which remains labeled as an
  official-weight replay rather than a new training run.
- 2026-09-07: Materialized 2,489,222 validation and 2,538,918 test candidate
  rows, then split each into four query-intact shards. Eight first array jobs
  failed before data loading because the batch shell did not support Bash's
  `printf -v` (about 0.08 failed GPU-hour total). They were replaced with eight
  explicit jobs `12541227`--`12541234`, four validation and four test P100
  shards. This is an orchestration failure, not model evidence.
- 2026-09-07: A bounded attempt to migrate the slowest validation shard to an
  A5000 was cancelled when the scheduler estimated a two-day queue delay. The
  shard restarted immediately as P100 job `12541915`; the other seven shards
  were uninterrupted. Eight dependent CPU scorers, real and constant-spectrum
  merges, and an independent all-shard rescore are registered to start only
  after successful prediction completion.
- 2026-09-07: The 64 GiB shard jobs reached roughly 95--98% before their
  size-sorted large-molecule tail caused DataLoader workers to exceed the host
  memory cgroup; the first observed failed shard peaked at 78,849,376 KiB RSS.
  This is a reference-construction resource failure, not a model or scoring
  result. Partial HDF5 files remain readable and contain every query group.
  Added an answer-free recovery helper that takes the exact set difference of
  public `(query, candidate InChIKey, collision energy)` keys, retries only
  missing valid structures with a smaller feature cache and 128 GiB host
  memory, and requires the scorer to combine the immutable partial and delta
  HDF5 files. The invalid dependency graph was cancelled before any label-side
  job ran; a fresh validation-first graph will be submitted after recovery.
- 2026-09-08: All eight answer-free candidate set-difference recoveries
  completed with batch size 2, two workers, a 64-entry feature cache, prefetch
  factor 1, 128 GiB allocations, and about 5.1--5.3 GiB process RSS. The eight
  jobs took 10:09--22:05 each. Including retained partial jobs, the probe, and
  one cancelled insurance retry, full validation/test candidate simulation
  consumed approximately 33.1 aggregate P100 GPU-hours. Scoring recovery used
  only public query IDs and strict disjoint merges.
- 2026-09-08: Before any private test-label access, the complete 9,734-query
  eligible validation replay passed the exact-permutation gate and obtained
  Hit@1/5/20 of 70.0945141%/86.8913088%/93.2401890%. Exact candidate ID,
  exact InChIKey, and 2D InChIKey metrics were identical. Relative to the paper
  no-CF numbers, these differ by +0.405/+0.311/-0.100 percentage points, so the
  registered entropy/top-50/0.01 Da configuration was frozen for test.
- 2026-09-08: After that gate, the 9,954-query GLACIER test-domain replay
  obtained exact-candidate Hit@1/5/20 of
  67.4904561%/84.4585091%/91.6817360%. All eligible targets are model-supported;
  across 2,538,918 candidate occurrences, 18,727 are explicitly unsupported by
  the model vocabulary and zero supported candidates lack predictions. The
  three local identity conventions remain numerically identical. The test
  result is lower than the paper's no-CF result by 2.200/2.121/1.658 points;
  because the public artifacts lack the paper-run ranking and a digest of its
  materialized mass-candidate table, this discrepancy is recorded rather than
  tuned away.
- 2026-09-08: The formal GLACIER/clean-MIST hybrid covers all 17,556 queries and
  obtains Hit@1/5/20 of 49.3905218%/64.1888813%/74.4360902% and MCES@1
  12.1789132. Its ranking SHA-256 is
  `b703afcdd0b543486d54b36b79e096de3259b25dbf0e29ebfb876b5d19e7a8c6`.
  An independent four-shard rescore and strict merge is byte-identical.
- 2026-09-08: Against the former 39.8724083% clean-MIST reference, the hybrid
  improves Hit@1 by 9.5181135 points. A 100,000-resample query-paired 99.9%
  bootstrap interval is [8.3447226, 10.6858111] points (2,918 hybrid-only
  hits, 1,247 MIST-only hits, and 13,391 ties). The constant-spectrum hybrid
  falls to 3.1385281%/6.1061745%/14.4964684% Hit@1/5/20 and MCES@1 26.4509854.
  The registered strength and spectrum-dependence gates therefore pass.
- 2026-09-08: GLACIER is promoted as an official-weight, method-level replay,
  not an independently retrained checkpoint and not an exact recovery of the
  unpublished paper run. The pinned source recipe uses train data for gradient
  updates and validation loss for selection; third-party checkpoint history
  cannot be independently attested. Single-H100 six-hour verifier timing also
  remains unmeasured and is disclosed rather than inferred from P100 shards.

## Claims ledger

| Claim | Required evidence | Status |
| --- | --- | --- |
| Official GLACIER assets are reproducible | immutable pins, hashes, strict load | complete for source and released no-CF weight |
| Training and negatives are held-out clean | source audit plus structure audit | source boundary complete; third-party checkpoint-history limitation disclosed |
| Task and paper protocols are equivalent | candidate/denominator/scorer comparison | falsified: paper/released domain is a 9,954-query test subset |
| GLACIER beats clean MIST | full rankings and paired 99.9% interval | complete: +9.518 points, interval [+8.345, +10.686] |
| The result uses spectral information | constant-spectrum and order controls | complete: constant-spectrum Hit@1 3.139% |
| Offline inference fits the contract | relocatable replay and resource profile | operator construction complete; single-H100 timing unmeasured |
| GLACIER is the formal benchmark baseline | all preceding gates plus docs/tests | complete as an official-weight hybrid replay with disclosed limits |

## Runnable reference path

The task-owned integration is split from the upstream repository so that the
benchmark contract remains reviewable. Prepare and verify the immutable assets:

```bash
bash environment/references/scripts/prepare_glacier_reference_assets.sh ASSET_ROOT
(cd ASSET_ROOT/ms-pred && uv sync --extra cu124)
```

The runner defaults every Python helper to `MASS_SPEC_GLACIER_PYTHON`; the
resolved environment contains the dependencies needed by both adapters. A
separate MIST-capable interpreter may instead be supplied through
`MASS_SPEC_REFERENCE_PYTHON`. The formal runner deliberately requires the
independently trained clean MIST weight and verifies both checkpoint digests;
it will not silently replace the fallback with the weaker released comparison
weight:

```bash
export MASS_SPEC_GLACIER_PYTHON=ASSET_ROOT/ms-pred/.venv/bin/python
export MASS_SPEC_MIST_CHECKPOINT=/absolute/path/to/clean/spectra_encoder.pt
bash environment/references/scripts/run_glacier_reference.sh \
  INPUT_ROOT OUTPUT_ROOT ASSET_ROOT
```

`OUTPUT_ROOT/rankings.jsonl` is a complete candidate permutation for every
query. `prepare_glacier_candidates.py` is answer-free;
`score_glacier_predictions.py` reproduces the pinned 0.01 Da, top-50,
square-root-intensity entropy protocol; and `merge_glacier_fallback.py` fails if
GLACIER is used outside the published domain or if the fallback is incomplete.
For large construction-time audits, `shard_glacier_candidates.py` and
`merge_partial_rankings.py` provide query-intact parallel replay without
changing rankings.

Audit the raw fold boundary and confirm that the upstream labels are exactly
the public-field released domain with:

```bash
python environment/references/scripts/audit_glacier_training_split.py \
  --source-tsv /absolute/path/to/MassSpecGym1.5.tsv \
  --upstream-labels ASSET_ROOT/ms-pred/data/spec_datasets/msg/labels.tsv \
  --output glacier-training-split-audit.json
```
