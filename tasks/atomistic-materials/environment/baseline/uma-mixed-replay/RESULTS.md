> Historical pretrained comparison only. These measurements used `mixed-replay-v1`, whose dev
> hashing used unreduced atom counts: supercell representations could cross the train/dev split.
> The current selector reduces compositions and guards parent membership; it uses a new version
> key and has not been retrained here. Do not claim these numbers reproduce the corrected selector.

# Results: Mixed replay (MP E/F/S + OMat24 forces/stress)

`/workspace/data/models/mixed-replay/`, trained with `prepare_mixed.py` (2 h 17 min on eight H100s); test scores come from the full verifier run.

Keeps MP-level stability screening (F1 0.86) and recovers part of κ (0.44 → 0.39) and MOF (25 % → 21 %), but not phonon frequencies or cleavage. Off-equilibrium replay alone is not enough.

Scored with the task's own drivers and scorers (`/workspace/evaluation`) in this image, on eight H100s. Lower is better except F1.

| Metric | validation | test |
| --- | ---: | ---: |
| discovery F1 ↑ | 0.846 | 0.860 |
| discovery hull MAE (meV/atom) | 21.5 | 20.4 |
| phonon max freq MAE (K) | 53.2 | 55.5 |
| phonon mean freq MAE (K) | 17.7 | 18.5 |
| phonon entropy MAE (J/K/mol) | 24.6 | 24.7 |
| phonon heat capacity MAE (J/K/mol) | 10.5 | 8.544 |
| phonon free energy MAE (kJ/mol) | 11.2 | 11.1 |
| bulk modulus MAE (GPa) | 5.868 | 6.793 |
| shear modulus MAE (GPa) | 7.361 | 12.6 |
| κ SRME | 0.386 | 0.385 |
| MOF bulk modulus MAPE (%) | 28.7 | 21.4 |
| AM26 energy RelRMSE (%) | 7.335 | 7.175 |
| AM26 force RelRMSE (%) | 14.2 | 14.1 |
| cleavage relative MAE (%) | 7.306 | 7.378 |
| NEB barrier MAE (eV) | 0.157 | 0.134 |
| NEB converged fraction | 0.969 | 0.969 |

Every case of every suite completed (no failed cases). Comparison with the other baselines: the task README, §6.
