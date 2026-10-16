# Results: UMA-S-1.1 zero-shot

`/workspace/data/models/uma-s-1p1.pt` with `calculator.py`, no training.

Best of the three baselines on phonons, κ, AM26 and cleavage, but its OMat-scale energies are not MP-compatible, so stability screening is poor (F1 0.54).

Scored with the task's own drivers and scorers (`/workspace/evaluation`) in this image, on eight H100s. Lower is better except F1.

| Metric | validation | test |
| --- | ---: | ---: |
| discovery F1 ↑ | 0.564 | 0.540 |
| discovery hull MAE (meV/atom) | 104.1 | 97.4 |
| phonon max freq MAE (K) | 10.2 | 7.926 |
| phonon mean freq MAE (K) | 3.859 | 4.293 |
| phonon entropy MAE (J/K/mol) | 10.3 | 8.199 |
| phonon heat capacity MAE (J/K/mol) | 4.468 | 1.942 |
| phonon free energy MAE (kJ/mol) | 2.465 | 2.404 |
| bulk modulus MAE (GPa) | 32.1 | 9.814 |
| shear modulus MAE (GPa) | 8.133 | 16.0 |
| κ SRME | 0.271 | 0.281 |
| MOF bulk modulus MAPE (%) | 27.5 | 25.2 |
| AM26 energy RelRMSE (%) | 4.124 | 3.827 |
| AM26 force RelRMSE (%) | 11.9 | 11.8 |
| cleavage relative MAE (%) | 4.960 | 5.267 |
| NEB barrier MAE (eV) | 0.191 | 0.126 |
| NEB converged fraction | 0.949 | 0.964 |

Every case of every suite completed (no failed cases). Comparison with the other baselines: the task README, §6.
