# Baselines

| baseline | what it is | result |
| --- | --- | --- |
| [`starter/`](starter/) | one shared PPO actor/critic across all 39 cases, with padded observations/actions, an observation mask and task conditioning; native training and resume, atomic publication | runnable; no shared-model result yet |

The verified per-case numbers to approach (39 specialist controllers, not a valid single
submission) are in `/workspace/references/results`, with their training code and recipes in
`/workspace/references/specialists`.
