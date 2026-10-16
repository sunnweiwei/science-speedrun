# Shared PPO starter: status

The starter is runnable end to end: native training and resume of **one** shared PPO file across
cases, atomic publication as one zipapp, and inference through the contract. It has not been
trained into a competitive shared model; no shared-model benchmark result is claimed.

What is measured and verified:

- Per-case specialist controllers: 39 cases, PPO (DDPG for Pinball_2D_Re100), 4,000 or 20,000
  controls each.
  - Every published value matches the campaign's raw evaluation results, with matching hashes.
  - Five specialist packages were replayed on GPU through the verifier. Three were bit-identical;
    Channel_3D_Retau180 differed by 1.7e-7 relative; Nozzle2D_coarse varies about 0.5 % run to run
    because the solver is not deterministic.
  - The table, with each case's uncontrolled value, is in `/workspace/references/results`.
- These specialists are per-case models. A bank of them behind a dispatcher is not a valid
  submission, but they are the per-case numbers a shared model should approach or beat.
