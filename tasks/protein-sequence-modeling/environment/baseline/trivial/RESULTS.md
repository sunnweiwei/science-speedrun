# Trivial control: results

Held-out primary reward **0.2874**. The integrator published this package with the README's
commands and replayed it through the task's independent verifier: 0.28737, the same as the
author's measurement. Per task:

| task | reward |
| --- | ---: |
| binary_localization | 0.5752 |
| human_ppi | 0.4979 |
| solubility | 0.4997 |
| yeast_ppi | 0.4695 |
| secondary_structure | 0.4306 |
| ppi_affinity | 0.3688 |
| subcellular_localization | 0.2914 |
| fold | 0.0279 |
| stability, fluorescence, beta_lactamase | 0.0000 |

Constant predictions collect most of the accuracy-based tasks, which is why the floor is near 0.3.
The same package is the reference solution's oracle (`solution/solve.sh` in the task repository).
