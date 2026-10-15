# References

Material we have **not** re-run in this setup. Verified code and numbers are in
`/workspace/baseline/starter/`.

## Source code in the image

- `/opt/omg`: OMatG (Open Materials Generation) at `ed80cb86fd69d4f4d14164c6635b0a8d4939b34f`, with
  the task's packaging patch (no model, training or sampling change). It contains the other
  interpolants, encoders and configurations the starter does not use (for example EncDec-ODE and
  other stochastic-interpolant schedules).
- `/opt/lemat`: LeMat-GenBench at `58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6`, the scorer behind
  `/workspace/evaluation`, with the task's scheduling-only eight-GPU patch and small-batch guard.

## Historical results of the task author

Measured by the task author on Modal with an earlier (v0) model that also had conditional
adapters, 2,000 epochs from scratch on MP-20 train, then 2,500 unconditional structures through
the same UMA-200 relaxation and LeMat scoring. Those checkpoints do not load into the current
starter and were not re-run here.

| model | SUN + MetaSUN before relaxation | after relaxation |
| --- | ---: | ---: |
| Linear-SDE-Gamma | 1.88 % | 21.04 % |
| EncDec-ODE-Gamma | 2.64 % | 19.56 % |

## Papers and leaderboards

- P. Hoellmer et al., *Open Materials Generation with Stochastic Interpolants*, ICML 2025
  (arXiv:2502.02582). The model family of the starter.
- OMatG authors, *All That Structure Matches Does Not Glitter*, NeurIPS 2025 (arXiv:2509.12178). On
  structure matching, uniqueness and novelty in crystal-generation benchmarks.
- LeMaterial, LeMat-GenBench (arXiv:2512.04562) and its public leaderboard. The task's headline
  metric follows its unconditional SUN + MetaSUN ordering; leaderboard entries use their own
  training recipes and may submit already-relaxed structures, so their numbers are not directly
  comparable.
- T. Xie et al., *Crystal Diffusion Variational Autoencoder for Periodic Material Generation*,
  ICLR 2022. The source of the MP-20 split.

Published generator weights (OMatG or others) are not supplied and may not be used.
