# GPU training verification record

Recorded on 2026-10-15 from the GPU training verification and held-out rescore
supplied by Zifeng Wang for the PR #94 DrugCLIP reference. This records an
operator-reported run; the documentation update did not rerun training.

- [gpu_verification.json](gpu_verification.json): reported data revision,
  environment, training recipe, checkpoint selection, and held-out scores.
- [development_bedroc.csv](development_bedroc.csv): all 34 reported fine-tuning
  epochs, transcribed from the supplied table; these are development metrics.

The dual Uni-Mol model has 95,305,418 parameters. Training used Python 3.11,
PyTorch 2.6.0+cu124, generic Uni-Mol v0.1 molecule initialization, and a random
pocket encoder. The materialized dataset is
`zifeng-ai/target-based-drug-screening` at
`c8d2febc996e4c96b10f3aecd89fb7375150449f`.

ProFSA completed 100 epochs over 500,000 pairs on 8 A100s in 19.0 h, with
per-GPU batch 48, learning rate 1e-4, and the molecule backbone frozen.
BioLiP fine-tuning used 4 A100s, batch 48 per GPU, learning rate 1e-3, and
BEDROC(alpha=80.5) checkpoint selection. It stopped at epoch 34 after 20 epochs
without improvement, in 19.6 h. Epoch 14 was selected at dev BEDROC 0.4288.
This is completed training under the early-stopping policy, not 200 completed
fine-tuning epochs. The observed training environment differs from the
PR's PyTorch 2.7.1/CUDA 12.6 container image and H100 resource declaration.

The same epoch-14 checkpoint initially scored mean EF1% 1.830. Correcting
pocket element tokens, the 511-atom test crop, DUD-E median/MAD then max
pooling, and LIT-PCBA max pooling produced the final full-suite EF1% **16.108**:
DUD-E 18.056 over 102 targets and LIT-PCBA 2.861 over 15 targets. The earlier
1.830 is superseded; no additional training or test-based checkpoint selection
is reported. Random ranking has expected EF1%=1. The historical v1 task reward formula
applied to the rounded final EF gives approximately 0.94155; this is a
calculation, not an observed `reward.json` from the isolated verifier.

The original report omitted raw training logs, exact seeds, training-code
revisions and per-target predictions. The author subsequently released the
selected weights with SHA256
`bb108dcec5608b04eee546a0ed57248ab5f956cb61799a8c5611ab9c8da7c772`.
The controller uses these weights to replay the frozen model. Only generic
Uni-Mol initialization is available to the agent.

Two controller-side replays on 2026-11-04 recovered mean EF1%
16.107934114858 and legacy v1 reward 0.9415475887803699. One used the original
verifier and one used Harbor 0.23 with the task's pinned image. The replay
record includes all aggregate metrics, source hashes and agreement under
two independent input permutations. [Independent replay](independent_replay.json)

The current v2 primary reward is dataset-balanced BEDROC(alpha=80.5).
Re-aggregating those same full-suite reports gives 0.166384906515 for Harbor
and 0.166385032576 for upstream. `independent_replay_v2.json` records both,
the original report hashes and old rewards. `independent_replay.json` remains
the original v1 record. No training or inference was repeated for this migration.
