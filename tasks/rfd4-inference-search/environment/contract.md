# Submission contract

Submit a Python program at `/workspace/submission/infer.py`. Include additional modules and data in the same package. The controller starts one offline process per request on one GPU. The program must use the supplied frozen RFD4 model to produce fresh outputs from the current input and seed. It controls sampling, search, batching and stopping decisions. The controller measures cost and owns final refolding, clustering and reward. The output score verifies structural quality; verifying the implementation used the model as declared requires reviewing the submitted code.

## Request and response

Build and edit a complete candidate directory under `/workspace/research/`. From `/workspace`, publish it with `bash baseline/publish.sh "$candidate"`. The helper stages a copy on the submission filesystem and exchanges the complete directory at `/workspace/submission` in one operation.

The entry point is `python infer.py --request FILE --output DIRECTORY`. The output directory exists and starts empty. Resolve input files relative to the request file. `RFD4_MODEL` points to the frozen model and `DATA_PATH` to its assets parent. Private evaluation gives the worker a read-only package, request, runtime and model assets, writable temporary storage and its own output mount. Each request gets a fresh process.

```json
{
  "version": 1,
  "id": "example-117",
  "kind": "unconditional_monomer",
  "input": {"length": 117, "file": "monomer-117.cif.zst"},
  "num_outputs": 5,
  "seed": 7341,
  "budget": {"wall_seconds": 180}
}
```

Read the input, count, seed and time allowance on every invocation. Treat the id as opaque. Optional `metadata` carries controller-provided context. Exit 0 on completion, or 64 for an unsupported version or kind. Programs can ignore additive envelope fields. Future kinds define their payload and output semantics and need a scoring adapter. The current grader evaluates unconditional canonical protein monomers at a requested positive length within model and device capacity.

For slot 0 write `slot-000000.cif`, for slot 1 write `slot-000001.cif`, and so on. Write regular, uncompressed mmCIF files containing one canonical protein chain of the requested length. Missing and invalid slots stay in the denominator. The per-file limit is the larger of 1 MiB or 8 KiB per requested residue. Timeout or nonzero exit gives that request zero outputs.

## Baselines, cost and quality

The standard setup retains lengths 100, 150 and 250, with 10 slots per length in development and 50 in paper-size or private runs. The frozen weights, independent AF3 settings, Foldseek clustering and mean per-length clustered pass rate follow the original task. A candidate and a packaged human reference run sequentially on the same GPU with identical request data, seeds, slot counts, scoring and budgets. The reference policies retain their 180 trunk passes per slot on the standard suite. Free programs allocate their computation within the common elapsed generation/search limit, by default 36 seconds per requested slot. Startup, discarded trajectories and intermediate scoring count. The measured comparison uses GPU time; a program's trunk-call count is unverified.

The reward is the original quality metric at the stated budget. The paired report includes generation throughput, end-to-end throughput and distinct passing clusters per end-to-end GPU-hour. End-to-end cost sums generation/search and final AF3 time on the allocated GPU. CPU clustering is excluded. `quality_delta_pp` measures the quality difference in percentage points and `total_cost_ratio` divides candidate cost by reference cost. Dominance requires equal or higher quality and equal or lower actual cost, with at least one strict improvement and all requested slots successfully refolded in both arms. Higher quality at higher cost is a trade-off. Repeat at other `--seconds-per-slot` budgets on the same workload to measure additional curve points.

The default paired baseline is the reproduced stochastic TSR. Other choices for `--reference` are `geometry-beam`, `two-wide-beam`, `native-low` and `native-high`. Low and High replay the released monomer presets at 120 and 300 passes per sample. On standard 30- and 150-slot suites, `historical_baselines` retains the original public measurements from `baseline/*/reproduction/` and the corresponding available S15 and monomer comparison tables. Each archived point identifies its source. The paired difference uses the fresh measurement on the candidate's actual requests and hardware. An unexpected error in the trusted reference aborts the paired measurement.

Run `python evaluation/cost_curve.py budget1.json budget2.json --output points.csv` to export the measured curve. Supply full development reports or private `details.json` files from the same requests, seeds and device. The CSV includes both programs, timing, quality and observed frontier membership for complete cohorts. Historical points have their original timing, hardware and seeds and are available separately in the JSON.

The controller can pass `--suite FILE` containing `{"version":1,"requests":[{"id":"other-length","kind":"unconditional_monomer","input":{"length":117}}]}`. Harbor uses an optional `suite.json` beside the operator's private seed, plus the controller settings `RFD4_REFERENCE` and `RFD4_SECONDS_PER_SLOT`. The monomer grader weights distinct lengths equally. The default suite measures its three published lengths; a separate suite can test unseen lengths. The worker receives only its current request.

## Legacy policy packages

An `infer.py` entry selects program evaluation. Harbor wraps a valid legacy `policy.json` in the trusted program adapter so saved policies receive the same paired evaluation. Direct `run.py` replay of a bare policy retains the original v1 measurement; add `--program-track` for the paired comparison.

Submit one UTF-8 file, `/workspace/submission/policy.json`, up to 64 KiB. The verifier interprets the policy and owns both model executions. The same publisher accepts a directory containing the legacy policy and replaces the entire previous package.

### Policy

`version` is 1. `default` is a complete plan. Optional per-length replacements go in the required `by_length` object, with keys `"100"`, `"150"` and/or `"250"`. A plan has three required fields and one optional solver setting.

| Field | Meaning |
| --- | --- |
| `solver` (optional) | `"stochastic-tsr"` (default) or `"ode-tsr"`. The ODE uses the native zero-diffusion schedule for both modalities. |
| `bands` | An ordered list of `[number_of_steps, trunk_passes_per_step]`. Each depth is 1–8 including the initial pass. Use 20–180 steps in total and at most 60 bands. |
| `width` | Concurrent surviving trajectories per independent group. Use 1, 2 or 5. The runner fixes the number of groups so their final capacity equals the slot count. |
| `checkpoints` | Up to eight checkpoints in increasing integration-step order. An empty list and width 1 gives independent sampling. |

Each checkpoint has an integer `step`, `branches` (1–8 children per parent on the incoming interval), and `score`. The value `"clean"` scores the intermediate clean estimate. The value `"rollout"` advances every sibling to completion for scoring, then continues surviving intermediate states. Use `"rollout"` at a terminal checkpoint. With no terminal checkpoint, each survivor finishes once. Only the final survivors occupy submission slots.

An optional `priority` at each checkpoint defines a new pruning rule. The default keeps the largest geometry scores. Features are `"quality"` (negative fraction of atoms clashing under the native 1.5 Å definition), `"rank"` (best geometry rank 1, worst 0, stable ties), `"parent"` (best previous survivor index 1, last 0, with 1 for width one), and `"draw"` (independent seeded uniform random number). A priority is a feature, a number in [−16,16], or `[operator, left, right]`, nested up to five levels. Operators are `+`, `-`, `*`, `min`, and `max`. Intermediate values are clipped to [−10000,10000]. Native selection keeps the highest priorities within each group. For example, `["+", "rank", ["*", 0.25, "draw"]]` randomizes near ties. The search runtime receives only the geometry score and these numerical features.

### Legacy cost and outcome

Each length receives 50 final slots and 9,000 trunk passes in final evaluation. The development default is 10 slots and 1,800 passes per length. Use `--length 100`, `150` or `250` for a single-length development check. The private verifier always covers all three lengths. Each plan may spend at most 180 passes per slot. A denoising interval costs the sum of its step depths, multiplied by the number of children. Rollout scoring also charges each child the unused suffix of the schedule. After the last checkpoint, surviving branches pay for the remaining steps. This counts initial trunk evaluations, recycles and discarded work. Repeating a width-W group W fewer times keeps the accounting per delivered slot unchanged.

A second bound is 36 elapsed seconds per allocated slot on one GPU, per length, including sampler startup, decoding and search scoring. A cohort exceeding its limit earns zero for that length. Actual elapsed times are reported independently. Each final slot admits at most one common AF3 call. The returned timing diagnostic includes generation and AF3 startup/compilation on the allocated GPU and excludes CPU Foldseek. Compare it only on matched hosts.

The verifier refolds each complete canonical monomer with single-sequence, template-free AF3. It aligns N, CA, C and O atoms to the generated backbone and passes scRMSD ≤ 1.5 Å. Foldseek clusters the valid generated structures within each length. A cluster contributes one point if any member passes. Failed, incomplete or missing outputs stay in the denominator. Reward is the arithmetic mean of the three per-length clustered pass rates. The controller receives flat numerical metrics in `/logs/verifier/reward.json`. Full per-length records, the suite digest and any invalid-package reason are in `/logs/verifier/details.json`. Infrastructure errors abort evaluation. Malformed or over-budget packages receive zero.
