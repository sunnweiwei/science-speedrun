# Starter: reproduced results

The recorded command in `README.md` (`esm2_t6_8M`, 8 h masked-language pretraining, then 3 h joint
fine-tuning, best `tune` checkpoint) was run twice, seeds 0 and 1. Each run used one H100 and took
11 h. Each package was replayed by the task's independent verifier on the held-out PEER test
splits (primary reward: the equal-weight mean of the eleven task rewards).

| task | seed 0 | seed 1 | task author's run (A100) | trivial control |
| --- | ---: | ---: | ---: | ---: |
| binary_localization | 0.9017 | 0.9097 | 0.9074 | 0.5752 |
| stability | 0.7234 | 0.7388 | 0.7333 | 0.0000 |
| subcellular_localization | 0.7440 | 0.7256 | 0.7328 | 0.2914 |
| secondary_structure | 0.6935 | 0.7021 | 0.7211 | 0.4306 |
| solubility | 0.6788 | 0.6888 | 0.6998 | 0.4997 |
| human_ppi | 0.7468 | 0.8143 | 0.6962 | 0.4979 |
| fluorescence | 0.6760 | 0.6728 | 0.6641 | 0.0000 |
| yeast_ppi | 0.6878 | 0.6954 | 0.6218 | 0.4695 |
| beta_lactamase | 0.8283 | 0.8025 | 0.5868 | 0.0000 |
| ppi_affinity | 0.4694 | 0.4892 | 0.4400 | 0.3688 |
| fold | 0.2145 | 0.2117 | 0.2019 | 0.0279 |
| **primary reward** | **0.6695** | **0.6774** | 0.6368 | 0.2874 |

The author's run predates two changes: the learning-rate schedule now follows the same clock that
ends each phase (see `/workspace/references/measured_runs.md`), and an H100 sees more tokens in the
same hours. `fold` stays far behind every other task. The weights are not shipped.
