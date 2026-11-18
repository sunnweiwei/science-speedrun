# rfd3-small: results

Under the 20-design protocol the author reports 47 novel enzyme designs out of 820,
4 small-molecule binders out of 80, 0 protein binders out of 100 and 7 unconditional
folds out of 20. The four-track mean was 0.114329. That run used binder-only
ProteinMPNN. Under `proteinmpnn-fixed-target/v2` the protein-binder component needs a
new measurement. The other three counts carry forward. The original 100 binder
backbones, or the retained epoch-100 weights for regenerating them, suffice for an
independent measurement. [Author's report and checkpoint identity](https://github.com/openssi-dev/ai4sci-tasks/pull/52#issuecomment-6618380001)

The figures in the tables below describe the historical protocols in their headings.

The reproduced numbers for this baseline. The criteria behind them, the enzyme-target
distribution and the full write-up of how the graded and operator paths were reconciled are
in [../../evaluation/RESULTS.md](../../evaluation/RESULTS.md).

## What this baseline is

Upstream RFdiffusion3, vendored at `rfd3/` from
[RosettaCommons/foundry](https://github.com/RosettaCommons/foundry) `b02eed6a` and reduced in
**layer widths only** to **12,445,554** parameters against upstream `pretrain`'s 168,038,994
(7.4 %). Every depth is upstream's — 18 diffusion transformer blocks, 2 + 2 pairformer,
3 + 3 atom encoder/decoder, `n_recycle` 2 — as are the crop, the losses, the AF-3
optimizer/schedule and EMA. See [README.md](README.md) for the eight width leaves that change
and how to run it.

## Transfer (verifier, 51-target evaluation view)

Checkpoint `epoch-0100.ckpt`, the real protocol of 51 targets × 100 designs.
`conditioning_satisfied` is 100 % on all four tracks.

| track | `pass_rate` | `diversity` | operator harness |
|---|---:|---:|---:|
| enzyme | 0.0624 | 0.341 | 0.054 |
| sm_binder | 0.0275 | 0.750 | 0.030 |
| ppi_binder | 0.002 | 0.200 | 0.004 |
| unconditional | 0.420 | 1.000 | 0.410 |

The two paths agree within sampling noise, on counts of 12 vs 11, 1 vs 2, 220 vs 256 and
41 vs 42 passing designs. Reaching that agreement required fixing five divergences in which
the graded path posed a different problem from the published specs; `input.yaml` is the source
of truth and residue renumbering is the only licensed difference. The five are written up in
[../../evaluation/RESULTS.md](../../evaluation/RESULTS.md).

**This row is accepted from the task author's report rather than reproduced in this setup**,
because the checkpoint is not distributable.

## Anchors

| system | enzyme | sm_binder | ppi_binder | unconditional |
|---|---:|---:|---:|---:|
| trivial generator (extended backbone, reduced view) | 0.000 | 0.000 | 0.000 | 0.000 |
| `rfd3-small` above | 0.0624 | 0.0275 | 0.002 | 0.420 |

The trivial row is a **partial** null system: one target per track and two designs each, run
through the real scorer, where nothing was designable on any track. The full
null row over all 51 targets is still open, and matters most on `enzyme`, where 28 of 41
targets score exactly 0.

## Reading the numbers

- `ppi_binder` is measurable, not good: 1 of 500 is not distinguishable from zero
  (P(0 of 500 | true rate 0.004) ≈ 13 %). What the cross-check establishes is that the number
  is produced by the model rather than by a spec bug, which was not true of any earlier run.
- Thresholds come from RFdiffusion3's `success.txt`, which was calibrated with AlphaFold;
  scoring here uses RoseTTAFold3. Numbers are internally comparable and **not** comparable with
  the published table.
- 5 designs of 5,100 never reached the grader, each carrying a single `UNK` residue that
  `rfd3_collect.py` rejects. They count as failed designs, which shifts pass rates by ≤ 0.1 %.
