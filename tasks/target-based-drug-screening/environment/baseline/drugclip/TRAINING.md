# Paper-scale dual Uni-Mol DrugCLIP reference

This reference keeps the existing materialized ProFSA, BioLiP and held-out
DUD-E/LIT-PCBA data. It uses the [author's DrugCLIP model](https://github.com/THU-ATOM/Drug-The-Whole-Genome/blob/9d0db73915219763b5bcc3b7de86d7c4edb5ed03/unimol/models/drugclip.py)
architecture: independent molecule and pocket Uni-Mol encoders, each with 15
layers, width 512, FFN width 2048 and 64 heads. Both use Gaussian distance
features and a pair attention bias updated by every transformer layer. Each
encoder feeds a `Linear(512,512) -> ReLU -> Linear(512,128)` projection and L2
normalization. The contrastive scale is fixed at 14, matching the author's
`detach()` behavior. The reference has **95,305,418 parameters**; the 50M
rejection is removed and the exact count is written into every recipe/package.

The molecule encoder loads the pinned official Uni-Mol v0.1 checkpoint before
optimization. The pocket encoder starts randomly and is trained on ProFSA
while the molecule backbone remains frozen and in eval mode. BioLiP
fine-tuning loads the completed ProFSA state and trains both encoders and
projection heads. The existing materialization contains a 500k ProFSA subset
and a reconstructed BioLiP/Q-BioLiP population, so this reproduces the model
and training protocol on *this benchmark's data*, not the paper's exact data
population or reported result.

## Fixed pretrained assets

Put the following in `<data-root>/pretrained/` (or pass `--pretrained-dir`):

```text
mol_pre_no_h_220816.pt
dict_mol.txt
dict_pkt.txt
manifest.json
```

Copy `pretrained_manifest.example.json` to `manifest.json` after placing the
assets. It pins source revisions, sizes and SHA256 hashes. Training validates
all three assets and strictly loads the molecule backbone. It performs no
network download. The pocket dictionary comes from the pinned
[author repository](https://github.com/THU-ATOM/Drug-The-Whole-Genome/blob/9d0db73915219763b5bcc3b7de86d7c4edb5ed03/dict/dict_pkt.txt).
The published inference package includes its trained weights and does not need
these original files.

## Formal two-stage recipe

The defaults reflect the [author fine-tuning launcher](https://github.com/THU-ATOM/Drug-The-Whole-Genome/blob/9d0db73915219763b5bcc3b7de86d7c4edb5ed03/drugclip.sh)
and [ProFSA configuration](https://github.com/THU-ATOM/ProFSA/blob/a056baed91277059acba3836d5fbf3fccfbee605/conf/model/drugclip.yaml):

| | ProFSA | BioLiP fine-tuning |
| --- | ---: | ---: |
| GPU count | 8 | 4 |
| Per-GPU batch | 48 | 48 |
| Maximum epochs | 100 | 200 |
| Adam LR | 1e-4 | 1e-3 |
| Adam betas / eps / weight decay | (0.9, 0.999) / 1e-8 / 0 | same |
| LR scheduler | polynomial, power 1, end LR 0 | same |
| Warmup ratio | 0.06 | 0.06 |

The scheduler computes total updates from the actual distributed DataLoader and
optimizer steps; warmup updates are `int(0.06 * total_updates)`, as in
[Uni-Core](https://github.com/dptech-corp/Uni-Core/blob/ace6fae1c8479a9751f2bb1e1d6e4047427bc134/unicore/optim/lr_scheduler/polynomial_decay_schedule.py).
Training uses seed 1, FP16 on CUDA, clip norm 1.0, and gradient checkpointing.
There is no default wall-time cutoff. Gradient accumulation changes the
optimizer batch, but each pocket's contrastive candidate pool remains the
**48 local ligands**; the author's forward does not gather negatives across
GPUs. Required resources are 8 GPUs for ProFSA and 4 for fine-tuning, with
memory sufficient for a 95M-parameter dual encoder, 48 structures/GPU and up
to 256 atoms per structure. Provision high-memory GPUs and benchmark memory
before a full run; this repository does not claim a measured minimum VRAM.

From this directory, run the two stages on one 8-GPU host:

```bash
bash run_paper.sh /data/target-based-drug-screening \
  /runs/drugclip-paper /models/drugclip-paper
```

Or launch each stage separately (the second may use a 4-GPU host sharing the
completed checkpoint):

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --stage profsa --input-root /data/target-based-drug-screening \
  --work-dir /runs/drugclip-paper/profsa --batch-size 48

torchrun --standalone --nproc_per_node=4 train.py \
  --stage finetune --input-root /data/target-based-drug-screening \
  --profsa-checkpoint /runs/drugclip-paper/profsa/profsa_complete.pt \
  --work-dir /runs/drugclip-paper/finetune \
  --output-dir /models/drugclip-paper --batch-size 48 \
  --validation-batch-size 128
```

Use `--resume /runs/.../last.pt` with the same stage, world size and recipe
after an interruption. The checkpoint restores model, Adam, scheduler, AMP,
per-rank RNG, selected BEDROC and patience counter. An explicit `--max-seconds`
stops at the next epoch boundary after writing a resumable checkpoint. The
formal run is only marked `formal_recipe: true` if all default settings,
complete data and expected GPU counts are used.

## Development selection and inference

Fine-tuning uses the fixed `processed/biolip_dev/data.lmdb` in its recorded
order with epoch-0 conformer/crop choices. Each validation batch is a local
pocket-to-ligand retrieval task: the diagonal ligand is positive, duplicates
are masked, and BEDROC (alpha 80.5) is averaged over queries. A final
singleton is merged into the preceding batch. Metrics are summed across DDP
ranks without padded validation replicas. `valid_bedroc` is maximized; ties
keep the earlier checkpoint. Validation loss, AUC and top-1/top-3 accuracy are
auxiliary. Patience is 20 epochs. DUD-E and LIT-PCBA remain test-only and do
not affect selection.

The work directory receives `profsa_complete.pt`, `finetune_best.pt`,
`last.pt`, metrics and `training_summary.json`. The published model package
contains the selected checkpoint, config, exact source, checksums and
provenance. Score full held-out tests with:

```bash
bash /models/drugclip-paper/infer.sh \
  /data/target-based-drug-screening /predictions/drugclip-paper
python ../../../evaluation/score.py \
  /data/target-based-drug-screening \
  /predictions/drugclip-paper/scores.csv \
  /predictions/drugclip-paper/metrics.json
```

For a wiring smoke, use `--stage both`, two records per LMDB, one epoch per
stage, batch size two and separate output/work directories. These overrides
are recorded and cannot be reported as a completed formal run.

## Pocket cropping and distributed stopping

Pocket inputs use the author's distance-weighted `CroppingPocketDataset`
policy after hydrogen removal: atoms nearer the pocket centroid have larger
sampling probability. The sampled order is retained, then the selected
coordinates are centered. A local NumPy RandomState uses Uni-Core's
`hash((seed, epoch, index)) % 1e8` mix; the default seed is 1, dev and test
use epoch 0, and test uses a stable LMDB record index. Test inference permits
511 pocket atoms, matching the author's `test.sh`, while training uses 256.
Benchmark pocket LMDBs contain atom names such as `CA` and `N.am`; inference
converts these to element tokens before hydrogen removal, as in the author's
`AffinityPocketDataset`. DUD-E applies per-pocket median/MAD normalization
before max pooling; LIT-PCBA max-pools raw similarities. The policy and seed are
locked into the recipe and package provenance, so an older uniform-crop
checkpoint cannot be resumed. Every DDP rank receives rank 0's epoch-boundary
stop decision and elapsed time before RNG collection and checkpoint writing.

The released `test.py` constructs a DataLoader without setting a crop epoch,
but Uni-Core's `numpy_seed` requires an integer epoch when cropping occurs.
The held-out set has one such pocket (`LIT-PCBA/PPARG`, 523 heavy atoms after
hydrogen removal). This reference fixes its test epoch at 0 and records that
choice in inference metadata. All 4,241,148 frozen test LMDB molecules have
exactly one conformer, so the epoch choice cannot change their coordinates.

## Reproducible single-GPU acceptance smoke

With NVIDIA Docker support and the pinned local data/assets, run from the
repository root into an empty output directory:

```bash
bash target-based-drug-screening/environment/starter/reference/drugclip/run_single_gpu_smoke.sh \
  /mnt/c/github/datasets/target-based-drug-screening \
  /mnt/c/github/datasets/target-based-drug-screening/reference-smoke-v3
```

Set `DRUGCLIP_SMOKE_IMAGE` if the image differs from
`drugclip-paper-smoke:local`; it needs PyTorch/CUDA, NumPy, RDKit and LMDB.
The script uses the complete 95,305,418-parameter model and 256-atom limit.
It runs two ProFSA epochs on eight records, then two fine-tuning epochs on
eight train/eight dev records (batch 2, validation batch 4). Fine-tuning
stops at epoch 1 under an explicit 0.001-second budget and resumes from
`last.pt` to finish epoch 2. The independent inference container sees only a
copy of the model package and one complete DUD-E target; no training work
directory or pretrained files are mounted. The official scorer checks all
ligands in that target. This is a **one-target smoke evaluation**.

On 2026-10-06 the script passed with one NVIDIA GeForce RTX 3070 (8192 MiB).
Wall time was 239 seconds through final audit; peak allocated CUDA memory in
fine-tuning was 1,952,905,216 bytes. ProFSA completed two epochs/eight
optimizer updates. Fine-tuning logged `time_budget` after epoch 1, restored
optimizer, scheduler, AMP scaler, update count and best state, and finished
epoch 2 with `max_epochs` and eight stage updates. The audit verified that
ProFSA left the molecule backbone unchanged while updating pocket/projection,
and fine-tuning updated both backbones. Dev BEDROC rose from 0.2500000005
to 0.5; epoch 2 was selected. Losses and dev metrics were finite.

The full DUD-E `aa2ar` target had 32,010 scored ligands and 482 actives:
ROC-AUC 0.3411345, BEDROC(80.5) 0.0046597, EF1% 0.2068872. This tiny
training slice tests wiring, not benchmark performance. Standalone `model.pt`
SHA256: `3be8991006309e135b6b736dee309c94e94ed31eddf32c40de2c2805fe2c96b7`.
The script saves logs, `smoke_report.json`, scorer output, hashes and
timestamps. `formal_recipe` is `false` for this historical smoke. The later
full-data A100 training and early-stopped fine-tuning result are recorded below.

## Longer single-GPU pilot

To test whether additional training changes screening behavior, run a separate
medium-size trial with the same fixed materialization and pretrained assets:

```bash
bash target-based-drug-screening/environment/starter/reference/drugclip/run_single_gpu_pilot.sh \
  /mnt/c/github/datasets/target-based-drug-screening \
  /mnt/c/github/datasets/target-based-drug-screening/reference-pilot-512p-2048f
```

This uses the **first** 512 ProFSA records for 3 epochs and the first 2,048
BioLiP train records for 6 epochs, with 256 BioLiP dev records, one GPU,
batch 8 and validation batch 32. These prefix subsets are fixed by the
existing `--max-*-records` flags; the data population and splits are not
changed. The run is resumable at epoch boundaries. Only dev BEDROC selects the
checkpoint; `aa2ar` is scored afterward, using a copied model package and
the official scorer on all 32,010 ligands. The script saves logs, optimizer
checkpoints, predictions, `pilot_report.json` and the model SHA256.

The 2026-10-06 run took 1,398 seconds on one RTX 3070; peak allocated CUDA
memory was 4,475,086,336 bytes. ProFSA made 192 optimizer updates and BioLiP
made 1,536. Dev BEDROC by fine-tuning epoch was **0.249, 0.250, 0.201, 0.197,
0.194, 0.179**, so epoch 2 was selected. Dev AUC at epoch 6 was 0.655,
showing that the AUC and early-retrieval BEDROC trends differed.

The selected model scored the complete DUD-E `aa2ar` target at ROC-AUC
**0.7239**, BEDROC(80.5) **0.02858**, EF1% **0.6207**. The earlier eight-record
wiring smoke scored 0.3411, 0.00466 and 0.2069 on the same target. This
exploratory comparison changes training sample count and batch size, and
evaluates only one target. EF1% remains below the random-ranking value of 1;
it does not establish full-benchmark performance. The pilot model checkpoint
SHA256 is
`673fafb62f585f52ed1a35670048ee260a38f3d2330c32a6cd0410f6ac7d1203`.
Both historical runs have `formal_recipe: false`. A later complete-data,
multi-GPU run finished 100 ProFSA epochs and early-stopped BioLiP fine-tuning;
see the GPU training verification below.

## Released-checkpoint evaluator control

`official_checkpoint_eval.py` strictly maps the score-producing tensors from
the author's released DUD-E and LIT-PCBA 90%-identity checkpoints into this
reference's Uni-Mol forward path. Unused auxiliary heads are excluded by name.
The author [benchmark checkpoint release](https://huggingface.co/datasets/THU-ATOM/DrugCLIP_data/blob/main/README.md)
identifies these as the models for the two benchmark populations. From this
directory, with `hf` installed:

```bash
hf download THU-ATOM/DrugCLIP_data \
  benchmark_weights/dude_identity_90.pt \
  benchmark_weights/litpcba_identity_90.pt \
  --type dataset --revision 1025437db09a31d956dfede6a9a4bec81131713d \
  --local-dir /data/official-weights
python official_checkpoint_eval.py /data /runs/official-control \
  --dude-checkpoint /data/official-weights/benchmark_weights/dude_identity_90.pt \
  --pcba-checkpoint /data/official-weights/benchmark_weights/litpcba_identity_90.pt
python ../../../evaluation/score.py /data \
  /runs/official-control/scores.csv /runs/official-control/metrics.json
```

The script writes one atomic prediction file per target and resumes completed
targets. The final scorer validates every prediction against the frozen LMDB
order. The source SMILES index has 4,242,789 rows, while the author-released
test LMDBs contain 4,241,148 records; the scorer uses the latter population.

On 2026-10-12, a one-RTX-3070 run scored all 117 targets and 4,241,148 LMDB
ligands with these released checkpoint SHA256 values:
`dude_identity_90.pt` =
`cafcf98ecee190de7f1461fabcef877e3c93ce71026600d8c4622de88610452e`;
`litpcba_identity_90.pt` =
`997d343b0ba4b9436a30d76b7b3a0281f99d7e7f78f2e1fbdef5dc3bb0c67a31`.
The model's molecule and pocket forward paths matched the pinned author
Uni-Core code exactly on real test records (maximum absolute difference 0),
including a padded molecule batch.

| Benchmark | Targets | AUC here / paper | BEDROC(80.5) here / paper | EF1% here / paper |
| --- | ---: | ---: | ---: | ---: |
| DUD-E | 102 | 0.77465 / 0.7742 | 0.40126 / 0.3986 | 24.8276 / 24.61 |
| LIT-PCBA | 15 | 0.59540 / 0.5954 | 0.07295 / 0.0729 | 5.3661 / 5.36 |

The paper values are from [Science supplementary Tables S4 and S5](https://www.science.org/doi/10.1126/science.ads9530).
The equal-target mean EF1% over all 117 targets is 22.3325. The near match
supports the frozen test data and RDKit scorer once author-compatible pocket
preprocessing is used. Before this correction, 17,704 of 27,928 non-hydrogen
DUD-E pocket atoms and all 39,677 non-hydrogen LIT-PCBA pocket atoms were
mapped to `<unk>` by the reference inference code. The one-target pilot
checkpoint's `aa2ar` EF1% remained 0.6207 after the correction, so that
pilot's weak result is not explained by this preprocessing issue alone.


## GPU training verification reported 2026-10-15

The PR #94 dual Uni-Mol baseline was trained on the compact materialized
HF release `zifeng-ai/target-based-drug-screening` at
`c8d2febc996e4c96b10f3aecd89fb7375150449f`, using Python 3.11 and PyTorch
2.6.0+cu124. ProFSA completed 100 epochs on 8 A100s in 19.0 h. BioLiP
fine-tuning stopped at epoch 34 on 4 A100s in 19.6 h, after 20 epochs without
improving development BEDROC. Epoch 14 was selected at BEDROC(alpha=80.5)
0.4288. Both stages used batch 48 per GPU; learning rates were 1e-4 and 1e-3.
Molecule initialization was generic Uni-Mol v0.1, the pocket tower random,
and the molecule backbone stayed frozen during ProFSA training.

The same epoch-14 checkpoint's initial full-suite EF1% 1.830 was superseded
by **16.108** after the pocket element-token, 511-atom test crop, and author
pooling fixes. DUD-E (102 targets): ROC-AUC 0.769, BEDROC 0.295, EF1% 18.056.
LIT-PCBA (15 targets): ROC-AUC 0.577, BEDROC 0.038, EF1% 2.861.

See [the baseline reproduction record](../../../baseline/drugclip/reproduction/README.md)
for the supplied report and all 34 dev BEDROC values. This operator-reported
scientific verification does not prove current Harbor image/startup, shuffled
UID-separated replay, or H100 timing; raw run artifacts were not supplied.
