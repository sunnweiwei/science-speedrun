# OpenDDE official-architecture 50M baseline

The unchanged official `opendde pred` CLI and released 655,791,538-parameter
general/ABAG checkpoints remain the inference references.

The trainable baseline is a 48,746,962-parameter configuration-only scale-down
of the official `OpenDDE` class. It retains the MSA/template modules,
Pairformer, structural-token expansion, structural refiner, all-atom
diffusion, distogram and confidence branches.

The preview forces recycling through a no-grad block and routes MSA residuals
through an inference-only in-place implementation. `trainable_forward.py`
keeps the same layer order and official submodules, replacing only those two
orchestration guards with autograd-safe functional residuals. The EDM loss,
distogram weight, optimizer and schedule are our transparent
`foldingtrain_edm_v1` reconstruction, not an official OpenDDE Recipe.

The synthetic H100 graph gate completed official all-atom featurization,
forward, diffusion+distogram loss, backward, clipping and an optimizer step;
45,605,828 parameters received gradients. The production runner adds an
official-featurizer fail-closed native view, deterministic distributed one-pass
sharding, EMA, atomic checkpoints, per-rank RNG resume and common validation.
The exclusive DDP7 gate and complete 7,947-step campaign both passed with
released activation checkpointing, BF16 execution and chunked attention.
Confidence is frozen for the v1 update stage. The portable EMA bundle passed
an offline/read-only gate and completed all nine controlled FoldBench tracks;
see [`RESULTS.md`](RESULTS.md).
