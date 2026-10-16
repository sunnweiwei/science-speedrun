# Scratch baseline results

**The 1M-frame mixed-source scratch trial scored 37.08/100 (Score v2) on the full eight-suite validation.**
It is the scratch baseline to compare against: random UMA-S, 10 epochs on eight H100s.
It has already been scored, so there is no need to train or evaluate it again.
The smaller packaged MP-compatible starter sample (the commands in `README.md`) has no end-to-end score.

## Validation metrics

All 4,312 validation cases were scored with the task's drivers and scorers. Lower is better except F1.

The other columns are the other baselines' validation results (`../*/RESULTS.md`). Those baselines start from pretrained weights, so they are comparisons, not scratch baselines.

| Metric | scratch 10 ep (this) | UMA-S-1.1 zero-shot | MP fine-tune 3 ep | mixed replay |
| --- | ---: | ---: | ---: | ---: |
| discovery F1 ↑ | 0.759 | 0.564 | 0.856 | 0.846 |
| discovery hull MAE (meV/atom) | 32.5 | 104.1 | 20.5 | 21.5 |
| phonon max freq MAE (K) | 91.1 | 10.2 | 48.7 | 53.2 |
| phonon mean freq MAE (K) | 37.2 | 3.859 | 18.2 | 17.7 |
| phonon entropy MAE (J/K/mol) | 51.3 | 10.3 | 33.2 | 24.6 |
| phonon heat capacity MAE (J/K/mol) | 20.0 | 4.468 | 12.4 | 10.5 |
| phonon free energy MAE (kJ/mol) | 23.7 | 2.465 | 12.5 | 11.2 |
| bulk modulus MAE (GPa) | 12.9 | 32.1 | 5.560 | 5.868 |
| shear modulus MAE (GPa) | 14.2 | 8.133 | 7.277 | 7.361 |
| κ SRME | 0.852 | 0.271 | 0.455 | 0.386 |
| MOF bulk modulus MAPE (%) | 39.4 | 27.5 | 26.6 | 28.7 |
| AM26 energy RelRMSE (%) | 25.4 | 4.124 | 8.780 | 7.335 |
| AM26 force RelRMSE (%) | 27.9 | 11.9 | 14.8 | 14.2 |
| cleavage relative MAE (%) | 8.411 | 4.960 | 6.598 | 7.306 |
| NEB barrier MAE (eV) | 0.147 | 0.191 | 0.147 | 0.157 |
| NEB converged fraction | 0.969 | 0.949 | 0.961 | 0.969 |

**Gaps to the MP fine-tune**, as the ratio of the scratch model's error to the fine-tuned model's:

| suite | scratch / fine-tune error | note |
| --- | --- | --- |
| AM26 amorphous energy | about 3× | |
| AM26 amorphous force | about 2× | |
| κ | about 2× | |
| elasticity | 2–2.3× | |
| phonons | 1.5–2× | 4–10× the zero-shot model's |
| discovery hull MAE | 1.6× | |
| MOF moduli | 1.5× | |
| cleavage | 1.3× | |
| NEB barriers | equal | |

**Failures.** All are included in the metrics.
- No case failed to execute.
- 2 of 40 κ cases received the scorer's maximum-error penalty.
- 7 initial crystal relaxations and 8 of 256 NEB paths did not converge within the fixed limits.

The 9,071-case test was not used.

## Recipe

| Role | MPtrj | sAlex | OMat24 | MatPES-PBE | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training frames | 250,000 | 250,000 | 400,000 | 100,000 | **1,000,000** |
| Labeled-dev frames | 1,250 | 1,250 | 2,000 | 500 | **5,000** |

**Sample.** The training frames are 2.36 % of the 42,409,826-frame pool.
- Shards and rows are sampled deterministically; OMat24 subset quotas follow the released counts.
- At most 128 atoms per frame, and at most four frames per parent within each source and subset.
- Reduced compositions and parent IDs are disjoint between training and labeled dev.

**Supervision.**
- Energy: MPtrj and sAlex only.
- Forces and stress: all four sources.
- Both source groups share one `omat` embedding.
- Elemental offsets and the force RMS use training labels only.

**Optimization.**
- Random UMA-S, 146.6M parameters, seed 41, eight H100s.
- 10 epochs = 312,500 updates, global batch 32.
- AdamW 4e-4 with a fixed 10-epoch cosine schedule. The E/F/S weights (20/2/1) and other settings are as in `templates/`.
- Resumed after each epoch.

**Selection.** The final step-312,500 checkpoint had the lowest labeled-dev loss and is the one validated.

## Training curve

`reproduction/training-loss.png` shows the training and labeled-dev loss and the per-source dev errors.
- The weighted normalized dev loss falls from 1.971 at update 1,000 to 0.270 at the end.
- It is still decreasing when the cosine schedule reaches zero.
- Training and dev loss meet in the last epoch, with no sign of overfitting.
- End-of-run dev errors, read from the plot:

  | dev error | MPtrj + sAlex | OMat24 + MatPES |
  | --- | ---: | ---: |
  | energy (meV/atom) | ≈ 12 | — |
  | force (meV/Å) | ≈ 60 | ≈ 107 |
  | stress (GPa) | ≈ 0.33 | ≈ 1.2 |

## Cost

- **Training:** about 17 h 55 min of eight-H100 allocation, ≈ 143 H100 GPU-hours; 19 h 53 min elapsed.
- **Data preparation:** 33 min.
- **Validation:** 2 h 14 min of eight-H100 allocation, of which inference took 2 h 11 min.

## What is not included

- The preparation script for this mixed sample, the raw validation predictions and the training logs are not included.
- The weights are withheld by design.
- `reproduction/validation_metrics.json` holds every metric and failure count above.
