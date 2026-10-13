# Starter: reproduced results

The recorded command in `README.md` (30,000 steps, batch 8, width 32, crop 128, lr 1e-3, seed 0)
was trained on one H100 in the task's development image (24 minutes). The package was published with
`publish.py` and replayed by the task's independent verifier on all 1,085 held-out cases.

| group | mean VRMSE | group reward |
| --- | ---: | ---: |
| pretraining corpus (16 PDE families) | 0.950 | 0.341 |
| PDEArena Navier-Stokes | 0.872 | 0.353 |
| The Well: active matter | 1.458 | 0.279 |
| The Well: turbulent radiative layer | 1.464 | 0.278 |
| **reward** (mean of the four groups) | | **0.3127** |

Scale, for comparison (same protocol):

| predictor | reward |
| --- | ---: |
| persistence (repeat the last frame) | 0.2827 |
| spatial mean of the last frame | 0.3105 |
| starter (this directory) | 0.3127 |
| CNextU-net baseline (`../cnext_unet/RESULTS.md`) | 0.3447 |
| exact continuation | 1.0 |

The task author's own run of the same recipe (A100-40GB, 40 minutes) scored 0.3139. The
difference is one seed on different hardware. No seed band is claimed, and the package's weights
are not shipped.
