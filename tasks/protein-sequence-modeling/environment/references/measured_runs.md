# Measured reference runs

Everything here was trained from random initialization under this task's own
rules and budget, and scored by `tests/grader.py` on the held-out split. These
are not literature numbers and are **not** comparable to the PEER paper; see
the task README for why.

## Run 1 — multitask transformer, 5M-sequence corpus

The first complete reference. A generic pre-LN transformer rather than the
ESM-2 architecture, trained on a 5M-sequence sample of UniRef50 before the full
corpus was built. Superseded as the recommended starting point, but it is a
real, fully scored number and the only one measured at 37M scale.

| | |
| --- | --- |
| Architecture | pre-LN transformer, learned position embeddings, untied LM head |
| Size | 12 layers, d=480, 20 heads, ffn 1920 — **37,344,769** parameters incl. heads |
| Pretraining corpus | 5M-sequence UniRef50 sample, 1,324,789,425 residues |
| Pretraining | **2,458,214,252 tokens** in 8.00 h — 1.86 epochs at 85K tok/s |
| Batching | fixed 64 sequences (before length bucketing was adopted) |
| Fine-tuning | 56,001 steps in 3.00 h, tasks sampled ∝ sqrt(rows) |
| Model selection | 29 `tune` evaluations; best at step 56,000, tune reward **0.6118** |
| Seed | 0 |
| Hardware | 1× A100-80GB, 11 h 03 m wall |

Held-out result, beside the trivial fixed-statistic control:

| task | run 1 | trivial | delta |
| --- | --- | --- | --- |
| stability | 0.7413 | 0.0000 | +0.7413 |
| fluorescence | 0.6001 | 0.0000 | +0.6001 |
| beta_lactamase | 0.4118 | 0.0000 | +0.4118 |
| subcellular_localization | 0.6711 | 0.2914 | +0.3797 |
| human_ppi | 0.8101 | 0.4979 | +0.3122 |
| binary_localization | 0.8714 | 0.5752 | +0.2962 |
| secondary_structure | 0.6833 | 0.4306 | +0.2527 |
| solubility | 0.6963 | 0.4997 | +0.1966 |
| yeast_ppi | 0.6574 | 0.4695 | +0.1878 |
| ppi_affinity | 0.4975 | 0.3688 | +0.1288 |
| fold | 0.1212 | 0.0279 | +0.0933 |
| **primary reward** | **0.6147** | **0.2874** | **+0.3273** |

The held-out reward (0.6147) came out slightly above the selected `tune` reward
(0.6118), so `tune` was not optimistically biased here — but `tune` is small for
the PPI tasks (95 / 315 / 203 rows) and single-task moves on those are noisy.

## Run 2 — ESM-2 `esm2_t6_8M`, 5M-sequence corpus

The current reference. The same corpus and budget as run 1, changing only the
architecture: ESM-2 (rotary embeddings, tied LM head) at **7.41M backbone
parameters — a fifth of run 1** — with length-bucketed token-budget batching.

| | |
| --- | --- |
| Architecture | `esm2_t6_8M`: 6 layers, d=320, 20 heads, rotary, tied LM head |
| Size | **7,408,351** backbone, 9,244,131 including heads |
| Pretraining corpus | 5M-sequence UniRef50 sample, 1,324,789,425 residues |
| Pretraining | **14,494,972,940 tokens** in 8.00 h at 508K tok/s (20.9 epochs) |
| Batching | length-bucketed, 32,768-token budget |
| Fine-tuning | 214,313 steps in 3.00 h |
| Model selection | best `tune` reward 0.6666 |
| Seed | 0 |
| Hardware | 1× A100-80GB, 11 h wall |

| task | run 1 (37.3M) | **run 2 (7.4M)** | trivial |
| --- | --- | --- | --- |
| binary_localization | 0.8714 | **0.9074** | 0.5752 |
| stability | 0.7413 | 0.7333 | 0.0000 |
| subcellular_localization | 0.6711 | **0.7328** | 0.2914 |
| secondary_structure | 0.6833 | **0.7211** | 0.4306 |
| solubility | 0.6963 | 0.6998 | 0.4997 |
| human_ppi | 0.8101 | 0.6962 | 0.4979 |
| fluorescence | 0.6001 | **0.6641** | 0.0000 |
| yeast_ppi | 0.6574 | 0.6218 | 0.4695 |
| beta_lactamase | 0.4118 | **0.5868** | 0.0000 |
| ppi_affinity | 0.4975 | 0.4400 | 0.3688 |
| fold | 0.1212 | **0.2019** | 0.0279 |
| **primary reward** | 0.6147 | **0.6368** | 0.2874 |

**A fifth of the parameters scores higher.** Run 2 wins on 7 of 11 tasks and on
the aggregate, which is the clearest evidence available here that the
architecture choices matter more than raw width at this scale. `fold` — the
hardest task — improved most in relative terms, 0.1212 to 0.2019.

**This run carried a defect, so treat it as a lower bound.** Its learning-rate
schedule was keyed to a nominal step count while the loop stopped on wall clock;
the two desynchronized, and the rate sat at exactly zero for 32.6% of
pretraining steps and 72% of fine-tuning steps. The starter now drives the
schedule from the same clock that ends the phase, so the same recipe should do
better than 0.6368.

Both runs used the 5M-sequence sample. The full 37.3M-sequence corpus is built
and shipped, but no reference has yet been trained on it.

## Measured throughput, for budgeting

One A100-80GB, length-bucketed token-budget batches, `--token-budget 32768`:

| model | backbone | throughput | peak memory | one epoch of the full corpus |
| --- | --- | --- | --- | --- |
| `esm2_t6_8M` | 7.41M | 495K tok/s | 2.9 GB | 5.6 h |
| `esm2_t12_35M` | 33.27M | 198K tok/s | 8.5 GB | 14.0 h |
| `esm2_t30_150M` | 147.73M | 68K tok/s | 28.3 GB | 41.0 h |

Two results worth not re-deriving:

- **Length-bucketed token batching is worth 4×.** Fixed 64-sequence batches
  waste 3.4× of the compute on padding over this length distribution; run 1 paid
  that cost, which is why its 85K tok/s at 37M looks so far below run 2's 508K
  at 7.4M.
- **Raising the token budget past ~65K buys almost nothing.** 32,768 → 524,288
  grows peak memory from 2.9 GB to 44.8 GB and throughput only from 493K to
  568K tok/s. At 8M scale the GPU is compute-bound, not occupancy-bound. Spend
  tuning effort elsewhere; larger models shift this.
