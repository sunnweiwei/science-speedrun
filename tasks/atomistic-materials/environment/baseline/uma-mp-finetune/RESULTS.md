# Results: MP fine-tune (3 epochs, the original task's recipe)

`/workspace/data/models/mp-finetune-3ep/`: the original task's exported 3-epoch checkpoint (trained 2 h 16 min on eight H100s before the v2 exclusions; 0.2 % of its frames touch v2 evaluation materials).

Fixes stability screening (F1 0.86) and elasticity but loses most off-equilibrium accuracy: phonon errors grow 4–6×, κ SRME 0.28 → 0.44, AM26 and cleavage errors grow.

Scored with the task's own drivers and scorers (`/workspace/evaluation`) in this image, on eight H100s. Lower is better except F1.

| Metric | validation | test |
| --- | ---: | ---: |
| discovery F1 ↑ | 0.856 | 0.863 |
| discovery hull MAE (meV/atom) | 20.5 | 19.9 |
| phonon max freq MAE (K) | 48.7 | 47.1 |
| phonon mean freq MAE (K) | 18.2 | 17.9 |
| phonon entropy MAE (J/K/mol) | 33.2 | 28.0 |
| phonon heat capacity MAE (J/K/mol) | 12.4 | 8.215 |
| phonon free energy MAE (kJ/mol) | 12.5 | 11.5 |
| bulk modulus MAE (GPa) | 5.560 | 7.746 |
| shear modulus MAE (GPa) | 7.277 | 13.7 |
| κ SRME | 0.455 | 0.444 |
| MOF bulk modulus MAPE (%) | 26.6 | 25.0 |
| AM26 energy RelRMSE (%) | 8.780 | 8.469 |
| AM26 force RelRMSE (%) | 14.8 | 14.7 |
| cleavage relative MAE (%) | 6.598 | 6.724 |
| NEB barrier MAE (eV) | 0.147 | 0.126 |
| NEB converged fraction | 0.961 | 0.967 |

Every case of every suite completed (no failed cases). Comparison with the other baselines: the task README, §6.
