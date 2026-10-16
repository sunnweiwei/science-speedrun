# Released checkpoints replayed under this task's protocol

These numbers are the task author's replays of released pretrained checkpoints through the task's
verifier. They were not re-run in this setup. They show where published models land on the same
cases. The checkpoints themselves are not in the workspace and may not be used: every submitted
model is trained from random initialization.

On the pretraining corpus (850 held-out cases, 16 PDE families; mean VRMSE, lower is better):

| released model | corpus VRMSE |
| --- | ---: |
| PDE-Transformer mc-s | 0.4242 |
| PDE-Transformer mc-b | 0.3832 |
| PDE-Transformer mc-l | 0.3415 |
| DPOT-Ti | 1.7217 |
| DPOT-M | 1.6345 |

Zero-shot on the three transfer datasets (235 held-out cases; mean VRMSE per dataset):

| released model | PDEArena NS | active matter | turbulent layer |
| --- | ---: | ---: | ---: |
| PDE-Transformer mc-s | 1.2247 | 1.3182 | 1.3995 |
| PDE-Transformer mc-b | 1.3628 | 1.3239 | 1.3332 |
| PDE-Transformer mc-l | 1.3833 | 1.5321 | 1.4440 |
| DPOT-Ti | 0.7455 | 2.1945 | 7.7740 |
| DPOT-M | 0.9589 | 2.0801 | 7.5972 |

For comparison, the from-scratch CNextU-net baseline reaches 0.2968 on PDEArena
(`/workspace/baseline/cnext_unet/RESULTS.md`). The released models were trained on this corpus
for 100 epochs on four H100s.
