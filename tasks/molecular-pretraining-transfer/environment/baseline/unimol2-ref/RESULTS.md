# unimol2-ref: results

## Pretraining

`pretrain.sh` with the defaults: 30,000 updates, 3,000 warmup, batch 1,024 on 8 H100 (local batch 32, 4 accumulation
steps). The run took 5.1 h at about 1.65 updates/s. Loss on the 100k-molecule validation set (sum of the three
losses):

| updates | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---:|---:|---:|---:|---:|---:|
| validation loss | 0.120 | 0.092 | 0.085 | 0.071 | 0.068 | 0.063 |

At 30k updates the losses are masked token 0.0065, coordinate 0.048 and distance 0.0085, and they are still going
down. The run was made with the same code and flags as `pretrain.sh`, in a development image built from the same
pins. The first attempt hung at update 10,700 (the unequal-shard `IndexError` described in the README). After the
`unimol2x` fix it resumed from the update-10,000 checkpoint and finished without further incident.

## Transfer (verifier, test sets)

The package made by `publish.py` from `checkpoint_last.pt` was replayed through the isolated verifier.
The released 84M checkpoint and the untrained architecture were run
with the same fine-tune code (`evaluation/protocol`) on the same data.

| group | tasks | baseline MA-RAE | mean Spearman | released 84M MA-RAE | no pretraining MA-RAE |
|---|---:|---:|---:|---:|---:|
| QM9 | 6 | **0.183** | 0.965 | 0.192 | 0.413 |
| COMPAS-1D | 4 | **0.166** | 0.971 | 0.121 | 0.662 |
| ExpansionRx | 9 | **0.744** | 0.638 | 0.668 | 1.078 |
| ASAP | 5 | **0.636** | 0.640 | 0.644 | 1.315 |

The share of the gap from no pretraining to the released model that the baseline closes, (no pretraining −
baseline) / (no pretraining − released) on MAE, averaged over each group's tasks: QM9 1.03, COMPAS-1D 0.90,
ExpansionRx 0.81, ASAP 1.05. Over all 24 tasks it is 0.93.

| task | baseline MAE | RAE | Spearman | released 84M MAE | no pretraining MAE |
|---|---:|---:|---:|---:|---:|
| `qm9dft_v2` (HOMO/LUMO/gap, Ha) | 0.008386 | 0.301 | 0.929 | 0.008698 | 0.02702 |
| `qm9dft_v2_alpha` | 0.7089 | 0.133 | 0.985 | 0.777 | 1.479 |
| `qm9dft_v2_cv` | 0.3051 | 0.144 | 0.987 | 0.2649 | 0.4876 |
| `qm9dft_v2_mu` | 0.4519 | 0.354 | 0.914 | 0.5148 | 0.8729 |
| `qm9dft_v2_r2` | 9.503 | 0.080 | 0.990 | 10.03 | 17.21 |
| `qm9dft_v2_zpve` | 0.001809 | 0.085 | 0.987 | 0.00179 | 0.004659 |
| `compas1d_aea` | 0.02045 | 0.125 | 0.986 | 0.01889 | 0.1126 |
| `compas1d_aip` | 0.01673 | 0.113 | 0.995 | 0.0166 | 0.1265 |
| `compas1d_disp` | 0.0184 | 0.052 | 0.999 | 0.01355 | 0.03879 |
| `compas1d_dipmom` | 0.03083 | 0.372 | 0.903 | 0.01805 | 0.08236 |
| `exrx_logd` | 0.3766 | 0.464 | 0.877 | 0.3556 | 0.8647 |
| `exrx_ksol` | 0.3498 | 0.609 | 0.551 | 0.3086 | 0.5626 |
| `exrx_hlm_clint` | 0.3289 | 0.895 | 0.570 | 0.3097 | 0.3995 |
| `exrx_mlm_clint` | 0.3912 | 0.924 | 0.454 | 0.368 | 0.4241 |
| `exrx_caco_2_permeability_papp_a_b` | 0.3392 | 0.948 | 0.457 | 0.2945 | 0.4315 |
| `exrx_caco_2_permeability_efflux` | 0.4373 | 0.929 | 0.584 | 0.443 | 0.5007 |
| `exrx_mppb` | 0.2592 | 0.808 | 0.680 | 0.2075 | 0.3405 |
| `exrx_mbpb` | 0.1752 | 0.498 | 0.823 | 0.1486 | 0.3543 |
| `exrx_mgmb` | 0.2352 | 0.621 | 0.745 | 0.1865 | 0.4661 |
| `asap_hlm` | 0.3576 | 0.715 | 0.683 | 0.3838 | 0.5228 |
| `asap_ksol` | 0.146 | 0.776 | 0.135 | 0.1321 | 0.4111 |
| `asap_logd` | 0.4293 | 0.594 | 0.790 | 0.3777 | 0.7684 |
| `asap_mdr1_mdckii` | 0.1772 | 0.370 | 0.860 | 0.2297 | 0.5766 |
| `asap_mlm` | 0.4021 | 0.722 | 0.734 | 0.4161 | 0.5988 |

ADMET MAEs are in log10(x + 1) units (LogD in its own units). Without pretraining, the model predicts one constant
value on six ExpansionRx endpoints and on COMPAS dipole moment, so its correlation there is 0.

## Reading the numbers

- Fine-tunes are deterministic: the verifier replay reproduced the development run's MAE on all 24 tasks to the
  printed precision.
- Pretraining is still improving at 30k updates. Longer schedules, better use of the GPUs (throughput, batch
  size), and objectives or data choices that help ADMET are the obvious first levers.
- ExpansionRx and COMPAS-1D separate the baseline from the released model; QM9 and ASAP barely do.
