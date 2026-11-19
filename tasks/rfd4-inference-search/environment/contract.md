# Submission contract

Submit one UTF-8 file, `/workspace/submission/policy.json`, up to 64 KiB. The verifier interprets the policy and owns both model executions. See either baseline for a complete example. Use a temporary file followed by `mv` on the same filesystem to publish.

## Policy

`version` is 1. `default` is a complete plan. Optional per-length replacements go in the required `by_length` object, with keys `"100"`, `"150"` and/or `"250"`. A plan has three required fields and one optional solver setting.

| Field | Meaning |
| --- | --- |
| `solver` (optional) | `"stochastic-tsr"` (default) or `"ode-tsr"`. The ODE uses the native zero-diffusion schedule for both modalities. |
| `bands` | An ordered list of `[number_of_steps, trunk_passes_per_step]`. Each depth is 1–8 including the initial pass. Use 20–180 steps in total and at most 60 bands. |
| `width` | Concurrent surviving trajectories per independent group. Use 1, 2 or 5. The runner fixes the number of groups so their final capacity equals the slot count. |
| `checkpoints` | Up to eight checkpoints in increasing integration-step order. An empty list and width 1 gives independent sampling. |

Each checkpoint has an integer `step`, `branches` (1–8 children per parent on the incoming interval), and `score`. The value `"clean"` scores the intermediate clean estimate. The value `"rollout"` advances every sibling to completion for scoring, then continues surviving intermediate states. Use `"rollout"` at a terminal checkpoint. With no terminal checkpoint, each survivor finishes once. Only the final survivors occupy submission slots.

An optional `priority` at each checkpoint defines a new pruning rule. The default keeps the largest geometry scores. Features are `"quality"` (negative fraction of atoms clashing under the native 1.5 Å definition), `"rank"` (best geometry rank 1, worst 0, stable ties), `"parent"` (best previous survivor index 1, last 0, with 1 for width one), and `"draw"` (independent seeded uniform random number). A priority is a feature, a number in [−16,16], or `[operator, left, right]`, nested up to five levels. Operators are `+`, `-`, `*`, `min`, and `max`. Intermediate values are clipped to [−10000,10000]. Native selection keeps the highest priorities within each group. For example, `["+", "rank", ["*", 0.25, "draw"]]` randomizes near ties. The search runtime receives only the geometry score and these numerical features.

## Cost and outcome

Each length receives 50 final slots and 9,000 trunk passes in final evaluation. The development default is 10 slots and 1,800 passes per length. Use `--length 100`, `150` or `250` for a single-length development check. The private verifier always covers all three lengths. Each plan may spend at most 180 passes per slot. A denoising interval costs the sum of its step depths, multiplied by the number of children. Rollout scoring also charges each child the unused suffix of the schedule. After the last checkpoint, surviving branches pay for the remaining steps. This counts initial trunk evaluations, recycles and discarded work. Repeating a width-W group W fewer times keeps the accounting per delivered slot unchanged.

A second bound is 36 elapsed seconds per allocated slot on one GPU, per length, including sampler startup, decoding and search scoring. A cohort exceeding its limit earns zero for that length. Actual elapsed times are reported independently. Each final slot admits at most one common AF3 call. The returned timing diagnostic includes generation and AF3 startup/compilation on the allocated GPU and excludes CPU Foldseek. Compare it only on matched hosts.

The verifier refolds each complete canonical monomer with single-sequence, template-free AF3. It aligns N, CA, C and O atoms to the generated backbone and passes scRMSD ≤ 1.5 Å. Foldseek clusters the valid generated structures within each length. A cluster contributes one point if any member passes. Failed, incomplete or missing outputs stay in the denominator. Reward is the arithmetic mean of the three per-length clustered pass rates. The controller receives flat numerical metrics in `/logs/verifier/reward.json`. Full per-length records, the suite digest and any invalid-package reason are in `/logs/verifier/details.json`. Infrastructure errors abort evaluation. Malformed or over-budget packages receive zero.
