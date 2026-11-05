# Paper-scale DrugCLIP reference on the existing screening dataset

The existing target-based-drug-screening materialization and train/dev/test
split remain unchanged. This change replaces the reference's compact
Uni-Mol-plus-small-pocket default with the paper-scale **dual Uni-Mol** model
and the author's two-stage optimization and development checkpoint selection.
No full 100+200-epoch model is claimed here.

Both encoders have 15 layers, width 512, FFN width 2048 and 64 attention
heads, with Gaussian distance features and layer-by-layer pair attention bias.
The pinned official Uni-Mol molecule checkpoint is loaded before training;
the pocket encoder starts randomly. Both sides use the author's two-layer
128-dimensional projection, L2 normalization and fixed contrastive scale 14.
The track's old 50M guard no longer applies to this reference; its exact
parameter count is recorded. Historical scratch presets remain as code only.

The ProFSA stage freezes the molecule backbone, uses 8×48 batches, Adam at
1e-4 and runs up to 100 epochs. Fine-tuning loads its completed checkpoint,
unfreezes both encoders, uses 4×48 batches, Adam at 1e-3 and runs up to 200
epochs. Each stage has a fresh Uni-Core-style polynomial LR schedule (power 1,
end LR 0, warmup ratio 0.06). FP16, seed 1 and clip norm 1.0 remain the
formal defaults. A `torchrun` launcher and resumable stage checkpoints are
included; there is no default time cutoff.

BioLiP development selection now maximizes author-style `valid_bedroc`:
within each fixed validation batch, every pocket retrieves its paired ligand
from the local ligand pool with duplicate masking; per-query BEDROC uses
alpha 80.5. Loss, AUC and top-1/top-3 accuracy are diagnostic only. Equal
BEDROC keeps the earlier checkpoint and patience is 20. DUD-E/LIT-PCBA stay
held-out, author-released final tests and are never used for model selection.

The data population still differs from the Science paper: the available
materialization uses a 500k ProFSA subset and reconstructed BioLiP/Q-BioLiP
training data. This PR reproduces the model/training method on that fixed
benchmark dataset, not the paper's published data population or results.

Verification includes strict pretrained-asset hashes, numerical comparison of
both encoders against pinned author Uni-Mol/Uni-Core source on identical
weights and inputs, freeze/update assertions, polynomial/resume and BEDROC
fixtures, and an abbreviated real-LMDB two-stage train/package/inference run.
Pocket cropping now uses the author's distance-weighted probabilities after
hydrogen removal, with Uni-Core-compatible seed mixing and stable dev/test
record indices. Rank 0 broadcasts each epoch's stop reason and elapsed time
before all ranks collect RNG state and synchronize after checkpoint writing.
The crop policy/seed are recorded in the recipe, blocking resume of an older
uniform-crop checkpoint. Tests include a >256-atom author-reference fixture
and a two-process Gloo stop broadcast.

The reproducible single-GPU script completed on one RTX 3070 (8192 MiB) in
239 seconds with 1,952,905,216 peak allocated CUDA bytes in fine-tuning. It
ran ProFSA for 2 epochs/8 records and BioLiP for 2 epochs/8 train and 8 dev
records, using the full 95,305,418-parameter model, batch 2 and validation
batch 4. Fine-tuning stopped at epoch 1 with `time_budget`, resumed from
`last.pt` with optimizer/scheduler/AMP/update/best state, and completed epoch
2 with `max_epochs`. Dev BEDROC rose from 0.2500000005 to 0.5 and selected
epoch 2. The audit confirmed ProFSA's frozen molecule backbone and trainable
pocket/projection, followed by updates to both backbones in fine-tuning.

An independent copy of the model package, without mounts for the training
directory or pretrained assets, scored all 32,010 ligands in DUD-E `aa2ar`.
The official scorer returned ROC-AUC 0.3411345, BEDROC(80.5) 0.0046597 and
EF1% 0.2068872. This is **one complete target of smoke evaluation**, performed
after checkpoint selection; it is not the full DUD-E benchmark. The model
checkpoint SHA256 is
`3be8991006309e135b6b736dee309c94e94ed31eddf32c40de2c2805fe2c96b7`.
`formal_recipe: false`; multi-GPU and formal 100+200-epoch training remain
unverified.

Run the formal 8-GPU then 4-GPU sequence with
`environment/starter/reference/drugclip/run_paper.sh`; see the reference
README for separate stage and resume commands.

A separate resumable one-GPU pilot uses the same data and pretrained assets
with 512 ProFSA records for 3 epochs, 2,048 BioLiP train records for 6
epochs, 256 dev records and batch 8. It completed in 1,398 seconds on an RTX
3070; peak allocated CUDA memory was 4,475,086,336 bytes. Dev BEDROC selected
fine-tuning epoch 2 (0.24986); later epochs fell to 0.17911 by epoch 6.
After selection, the independent package scored all 32,010 DUD-E `aa2ar`
ligands at ROC-AUC 0.72387, BEDROC(80.5) 0.02858 and EF1% 0.62066, up
from 0.34113, 0.00466 and 0.20689 in the eight-record wiring smoke. This is
an exploratory single-target comparison, not full-benchmark evidence;
EF1% is still below 1. The pilot package SHA256 is
`673fafb62f585f52ed1a35670048ee260a38f3d2330c32a6cd0410f6ac7d1203`.
Both runs record `formal_recipe: false`.

## One-hour agent task and full offline evaluation

A separate one-hour agent-runner session completed on the local RTX 3070 on
2026-10-16. It ran from 03:01:15 to 03:56:10 UTC (54 minutes 54 seconds) and
produced a self-contained, 5,322,247-parameter
`geometric-descriptor-dual-encoder-ensemble` submission. This is an agent
candidate, **not** the 95,305,418-parameter paper-scale reference described
above. Its reported BioLiP dev retrieval BEDROC(80.5) was 0.22607 over 55,215
records; this development metric was not the final test score. The frozen
submission snapshot is `20261016T035437.149766Z-f84412c4f546` (tree SHA256
`f84412c4f54608e93bd866b4c6c004d22f3096975e62af1622fd1b3929db2061`);
its model checkpoint SHA256 is
`05c7408be8b5969f7228b1bd57e18e9d4217aafa6c248d3ebce77cf8596a0dd3`.

Independent offline evaluation of that unchanged snapshot completed with
`valid: true` across all 102 DUD-E and 15 LIT-PCBA targets and all 4,241,148
ligands. The primary equal-target mean EF1% was **2.96496** (DUD-E **3.21094**;
LIT-PCBA **1.29226**), giving agent-runner reward **0.74779** under
`EF1% / (1 + EF1%)`. Equal-target mean ROC-AUC was 0.54661 for DUD-E and
0.50528 for LIT-PCBA; mean BEDROC(80.5) was 0.06246 and 0.02162,
respectively. These are full-suite test results obtained only after the agent
session and snapshot freeze. The local run ID is
`drugclip-test-1h-20261015-200050`; the completed evaluation receipt is
`evaluations/manual-20261016-100058/evaluation.json` within that run.

The first verifier replay exposed an inference scratch-directory conflict and
a Harbor reward-format mismatch. Commit `065334e` fixed both; the corrected
verifier passed regression tests and the full offline replay above. This
validates the one-hour agent task, submission handoff and end-to-end scoring
path. It does not establish statistical performance across multiple agent
seeds or completion of the formal 100+200-epoch reference training.
