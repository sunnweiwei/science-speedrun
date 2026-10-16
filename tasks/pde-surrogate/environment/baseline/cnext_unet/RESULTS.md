# CNextU-net: reproduced results

The recipe in `README.md` was run from scratch on PDEArena Navier-Stokes on one H100, in the task's
development image with the pinned `the_well` source. All 200 epochs took 10.1 h, about 3 minutes
per epoch. The best-validation checkpoints were packaged with `package.py` and replayed by the
task's independent verifier on all 1,085 held-out cases.

| checkpoint | PDEArena VRMSE | reward |
| --- | ---: | ---: |
| best validation within the first 124 epochs (epoch 120; the task author's 12-hour A100 run stopped there) | 0.2968 | **0.3447** |
| best validation over all 200 epochs | 0.2821 | 0.3466 |

Per group, epoch-120 checkpoint:

| group | mean VRMSE | group reward |
| --- | ---: | ---: |
| pretraining corpus (16 PDE families; the PDEArena model or persistence) | 1.188 | 0.308 |
| PDEArena Navier-Stokes (the dataset the model is trained on) | 0.297 | 0.509 |
| The Well: active matter (persistence) | 1.523 | 0.272 |
| The Well: turbulent radiative layer (persistence) | 1.358 | 0.289 |
| **reward** | | **0.3447** |

The task author's run of the same recipe (A100-40GB, stopped by a 12-hour cap at epoch 124,
best-validation epoch 120) scored 0.3446, with PDEArena VRMSE 0.2978.

On the turbulent radiative layer, the author's run of `train_trl2d.sh` reached VRMSE 0.6743 (not re-run in this setup). Released checkpoints replayed under the task
protocol for comparison are in `/workspace/references/released_models.md`.

Weights are not shipped. Every model you submit must be trained from random initialization.
