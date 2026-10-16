# Surface-code decoding: original benchmark specification

> Historical document: the warm-start proposals below describe the original v1
> release. Current v1.2 requires random initialization, supplies no trained
> agent weights, and allows 8 H100 GPUs × 48 hours. The current
> [runtime contract](../environment/contract.md) supersedes those proposals;
> the original measured results remain provenance.

Updated 2026-09-22. The measured reference is the independently audited
`alphaqubit-v1` run. The user accepted this level of performance as sufficient
for the benchmark. The completed, bounded d5 continuation is supplementary
and does not select or replace this reference. No further accuracy campaign is
required for this specification.

**Release update (2026-09-22):** this document preserves the accepted baseline
and its original design proposal. The executable v1 implementation now includes
physically separated HF role archives, generic inference and warm-start loaders,
Harbor development/verifier environments, sealed grading and fixed normalization.
See the current [runtime contract](../environment/contract.md),
[agent instructions](../instruction.md), and [score configuration](../score_config.json).
The user chose the strongest same-test reference (released TN) as the score's
upper anchor, replacing the unnormalized reward proposed below. Deployment
validation records distinguish actual engine runs from static configuration
checks; the historical packaging checklist below is not current readiness.

## Scientific task

Given the repeated stabilizer detection events of one Sycamore surface-code
memory experiment, predict the probability that its official logical-path
observable flipped. This is a binary probabilistic decoding task, not quantum
circuit execution or language-model question answering. At inference, threshold
the probability at `p >= 0.5` to obtain the logical correction decision.

The first release covers **distance 3 and 5, hard readout, known geometries and
experiment lengths of 1,3,...,25 rounds**. It does not claim soft-I/Q readout,
leakage-aware Pauli+ decoding, d7–11 scaling, new-hardware generalization or
real-time control latency. Those are separate potential tasks.

## Data and fixed split — measured

Source: [Google Quantum AI Sycamore data, Zenodo 6804040](https://zenodo.org/records/6804040),
CC-BY-4.0. Archive `google_qec3v5_experiment_data.zip`, 315,490,804 bytes,
MD5 `a7fd8b481c3087090093106382dc217d`; expanded local release 548,415,888 bytes.
Keep upstream attribution when redistributing allowed resources.

There are four d3 patches and one d5 patch, each measured in X/Z basis at
13 odd round counts: **10 patch/basis groups × 13 durations = 130 conditions**.
Each condition has 50,000 shots; the selected surface-code subset has 6.5M
shots. Repetition-code experiments in the upstream release are excluded.

| Split | Zero-based upstream row indices, per condition | Per condition | All 130 conditions |
| --- | --- | ---: | ---: |
| Train | `0,2,...,39758` | 19,880 | 2,584,400 |
| Validation | `39760,39762,...,49998` | 5,120 | 665,600 |
| Test | `1,3,...,49999` | 25,000 | 3,250,000 |

Train labels fit parameters; validation labels support evaluation and checkpoint
selection. Test labels belong to the trusted scorer. This is a within-experiment
held-out-shot split, not a split by geometry, device or experimental session.
The public historical test is not a previously secret dataset.

Synthetic pretraining is a separate, on-demand stream from the released
`pij_from_even_for_odd.dem` models through Stim, with the recorded noise curriculum
and independent training/validation seeds. It has no fixed downloaded training
row count. The baseline consumed about 248–253M synthetic examples per d3 model
and 158M per d5 model, 2,323,085,056 total. Its synthetic validation has 5,120
independent full-noise examples per duration and is separate from hardware
validation. Synthetic examples never replace the hardware test.

The published even-fitted DEM uses all even syndromes, including the hardware
validation syndromes, but not their logical labels. Preserve and disclose this
fixed calibration exception; it is not a strictly train-syndromes-only protocol.
Never provide the odd-fitted reverse-fold DEM to the agent.

## Reference decoder — measured

Reference paper: [Bausch et al., Learning high-accuracy error decoding for quantum
processors, Nature 2024](https://www.nature.com/articles/s41586-024-08148-8).
This is an independent JAX/Haiku reconstruction of its recurrent Transformer,
using documented choices where the pseudocode is incomplete. It is not official
Google code or weights. The paper is the influential method reference; do not
describe this implementation as matching the complete paper or as verified
current SOTA.

| Component | Implemented configuration |
| --- | --- |
| Recurrent state | One 320-dimensional vector per stabilizer; 8 stabilizers at d3, 24 at d5 |
| Per-round core | Three Transformer blocks, reused over experiment rounds |
| Attention | Four heads; key and value dimensions 32 per head; geometry/event-dependent bias |
| Feed-forward | Gated GELU dense layers, expansion setting 5 |
| Spatial processing | 3×3 convolutions, width 160; dilation sequence (1,1,1) for d3 and (1,1,2) for d5 |
| Readout | Spatial pooling and 64-wide residual readout, 16 residual layers; scalar flip probability |
| Training | Binary cross entropy plus next-stabilizer auxiliary loss (weight 0.02), LAMB-style updates, EMA checkpoint selection |
| Numerics | BF16 computation, FP32 master/inference checkpoint parameters |

Exact saved parameter counts are **8,449,990 per d3 model** and **8,456,070 per
d5 model**. One decoder package contains eight d3 specialists and two d5
specialists: 84,512,060 total parameters, 338,048,240 raw FP32 weight bytes
(approximately 338 MB). A shot uses one specialist, not a ten-model ensemble.
Each specialist supports all 13 durations. Both-stage archival packaging has
20 checkpoints and is approximately 682 MB including metadata/runtime files.

Input preprocessing derives measurement-frame values and previous events from
hard detector bits; it does not read raw final data-qubit parities or logical
labels. Geometry, basis and duration are legitimate inputs. The target is the
released `obs_flips_actual.01` convention, not the earlier LSTM's global parity.

### Complete held-out results

| Method | d3 fitted LER / round | d5 fitted LER / round |
| --- | ---: | ---: |
| Reproduced AlphaQubit, pretraining only | 3.105183% | 3.366771% |
| **Fixed reference: reproduced AlphaQubit, fine-tuned v1** | **3.012653%** | **3.228889%** |
| PyMatching MWPM, same test and fit | 3.877986% | 4.316659% |
| PyMatching correlated, same test and fit | 3.416151% | 3.450333% |
| Released belief matching, same test and fit | 3.117629% | 3.036428% |
| Released tensor network, same test and fit | 3.026392% | 2.908539% |
| Official AlphaQubit paper reference, different fold/ensemble protocol | 2.901% | 2.748% |

V1 uses one seed and one even-train/odd-test fold; the paper reference uses its
two-fold/ensemble protocol. The numerical paper gaps are +3.85% / +17.50%
relative. V1's error-suppression ratio is 0.9330, so it has not reproduced
improving LER with increasing distance. Its 95% shot-bootstrap CIs are
2.995034–3.027945% (d3) and 3.195628–3.259113% (d5); these do not measure seed
variance or hardware drift. Independent raw-prediction and fit audits passed.

The separately recorded 30k-step d5 continuation completed with LER3.218578%,
versus v1's3.228889%. It is a post-test follow-up on the same public fold and
does not replace the fixed reference above. Its additional allocation was
3.711389 GPU-hours. The scoring guard was repaired to check unchanged d3
prediction hashes, counts and deterministic results without requiring stochastic
bootstrap intervals to remain bitwise identical; CPU replay then passed.

## Training and evaluation cost — measured

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, one GPU per specialist,
at most eight simultaneous campaign GPUs. These are physical GPU-hours, not
H200-equivalent hours.

| Stage | Per specialist | Total actual allocation |
| --- | --- | ---: |
| Synthetic pretraining | About 18 hours | 179.9625 GPU-hours |
| Hardware fine-tuning | About 2 hours | 19.9467 GPU-hours |
| Relocated test inference, both stages | Ten jobs, 1.4–4.2 minutes each | 0.3911 GPU-hours |
| Statistical scoring, both stages | CPU, 2,000 bootstrap draws | About 11 seconds |

Main training plus test allocation is **200.3003 GPU-hours**; AlphaQubit probes
bring its campaign subtotal to 200.4353. The first full campaign took about
47.9 hours of wall time including scheduling delays and ended on 2026-09-22
at 02:16 UTC. Earlier interrupted QRENND work added 145.9019 GPU-hours to the
historical project total, but is not required to reproduce this baseline.

Fine-tuned checkpoint inference timers sum to 397.24 seconds for the complete
3.25M-shot test. This is offline batch timing on the observed runtime/cache
conditions, not guaranteed cold-start timing, a measurement on H200, or real-time
decoder latency. A fresh production verifier must be timed before release.

## RSI agent development resources — proposed release

Use a **warm-start model-development track**. Give the RSI agent:

1. Physically separated train and validation detector/label files and explicit
   manifests for all 130 conditions; do not mount the full upstream archive.
2. Ideal circuits, geometry metadata, authorized even-fitted DEMs, Stim sampler
   and the fixed noise-generation/split documentation.
3. Editable AlphaQubit training/preprocessing/inference sources, the pinned
   JAX/Haiku runtime, and standard/correlated PyMatching development controls.
4. The ten v1 fine-tuned inference checkpoints as the reference starting point;
   the ten pretraining checkpoints are also authorized optional initializers.
   Export only allowed weights/configuration/provenance, not the entire research
   artifact tree or raw test predictions.
5. A validation-only evaluator, the published reference results, and the
   model-delivery contract. The agent may inspect validation labels to diagnose
   errors and select checkpoints, but must not use them as training targets.

The agent may change architecture, objective, feature processing, simulation
noise curriculum, fine-tuning, calibration, distillation or decoder combinations.
One package may use ten specialists or a shared decoder, with one inference
entry point. It must cover the entire task. The reference checkpoint collection
is not a requirement to preserve its architecture or number of parameters.

Initial runtime proposal: **two 80–96 GB GPUs for 24 hours** of development,
with supplied weights; one GPU and a one-hour inference allowance in the
verifier. These are release-budget proposals pending Docker replay/timing,
not measured H100/H200 equivalence. The agent need not repeat the roughly
200-GPU-hour scratch reproduction. Runner scheduling and session management
remain outside the task implementation.

## Portable model submission — proposed release

Submit a complete artifact at `/workspace/submission/`:

```text
submission/
  infer.sh
  README.md
  model_config.json
  weights/                 # every selected specialist/shared model
  src/                     # model and preprocessing/inference code
  requirements.lock        # or another declared, pinned dependency specification
  training_provenance.json # allowed data/checkpoints, seed and selection record
```

The verifier invokes:

```bash
bash /path/to/submission/infer.sh INPUT_ROOT OUTPUT_ROOT
```

Inference must work after relocation, using only the package, declared runtime
and supplied inputs. It must not rely on the training process, optimizer state,
original workspace, undeclared caches, network access or stored test answers.
Weights are fixed during inference; recurrent state within an individual shot
is allowed. A CSV/NPZ of precomputed test predictions, a patch, or a report alone
is not a valid submitted model.

### Generic input/output schema

`INPUT_ROOT/manifest.json` lists independent requests. Each request gives an
opaque request ID, distance, basis, patch center, number of rounds, ideal-circuit
relative path, number of shots and detectors, and an input NPZ relative path.
The NPZ has evaluator-assigned `shot_ids: uint64[N]` and little-endian packed
`detectors_b8: uint8[N, ceil(num_detectors/8)]`. For this dataset,
`num_detectors = rounds * (distance**2 - 1)`. IDs do not encode upstream row
indices. All manifest/file paths are confined to the provided input tree.

Output one `<request_id>.npz` with exactly `shot_ids: uint64[N]` in input order
and `probabilities: float32[N]`. Each probability is the official-path logical
flip probability. The evaluator checks finite values in [0,1], exact ID and
condition coverage, missing/duplicate outputs and the fixed `p >= 0.5` decision.

The model must accept arbitrary batch sizes, request ordering and subsets within
the declared ten geometries and 13 durations. It must not require 50,000 input
rows, original odd row IDs or a complete 13-duration set in each invocation.
Unknown geometries/round counts are outside this version's supported domain.

## Ranking and scientific metrics — proposed release

Continue reporting the verified **fitted LER per round** separately at d3/d5,
using the existing free-intercept weighted log-fidelity fit on rounds3–25,
equal weight across patch/basis groups within each distance. Report fit quality,
all groups, paired uncertainty, error suppression, NLL/Brier and offline runtime.
Retain round1 measurements as diagnostics; they are excluded only from the LER fit.

For the agent leaderboard, prefer **balanced per-shot logical failure rate**:
threshold probabilities, calculate the error fraction for each condition,
average the 13 durations within each patch/basis group, average groups within
each distance, then give d3 and d5 equal weight. This avoids the 8:2 geometry
count making d3 dominate and avoids rewarding a deliberately flattened error
curve through its fitted intercept. It is a benchmark-specific ranking metric,
not the paper's LER. Fixed v1 counts give:

- d3 mean shot failure: 24.509231%.
- d5 mean shot failure: 24.478615%.
- **Balanced shot failure: 24.493923%**, lower is better.
- If Harbor requires a higher-is-better reward: `1 - balanced_failure`, with
  v1 reference **0.7550607692**. Invalid/incomplete submissions fail validity;
  do not drop failed conditions or award a score based on submitted prose.

The proposed primary averages all13 observed durations, including round1, and
therefore depends on this declared duration distribution. It does not directly
measure large-distance scaling, maximum memory lifetime or calibrated confidence.
Keep the scientific LER and both distance results visible to expose tradeoffs.
Use paired, condition-stratified uncertainty when comparing candidates.

This new scalar is derived from the already verified v1 counts; it has not yet
been installed as a Harbor reward or used to reselect reference checkpoints.
On the same proposed scalar, point-estimate balanced failures are28.395365%
for MWPM,25.968846% for correlated matching,24.472654% for released belief
matching and23.848404% for released TN. Thus v1 slightly trails belief matching
on this aggregation and is not the strongest decoder reference. These are point
comparisons; the new scalar's paired confidence intervals have not been computed.

## Isolation and remaining packaging work

The operator may read the complete original archive to materialize splits.
The agent may not access odd-shot inputs/labels, raw test measurements/sweep
bits, released per-shot decoder predictions, or `pij_from_odd_for_even.dem`.
Test labels must be physically absent from the predictor container, not merely
ignored by its loader. Run submitted inference without network in an input-only
environment, seal outputs, and score them in a separate trusted environment
that never imports submitted code. This controls direct access; it does not
make the public upstream test historically secret.

Before release:

- Materialize separately indexed train/validation/test inputs and private labels;
  publish only the allowed, versioned development bundle and checkpoint exports.
- Adapt the development loader to those physically separated files; its current
  reproduction version also assumes a full 50,000-row upstream condition.
- Generalize the current reproduction CLI, which still expects 50,000 upstream
  shots and fixed even/odd slicing, behind the proposed arbitrary-N interface.
- Add instruction.md, task.toml, development/verifier Docker environments,
  contract.md, trusted tests/test.sh and the fixed ranking metric.
- Run the real baseline through a clean relocated Harbor verifier with varied
  input ordering/chunk sizes, test data inaccessible during development and
  labels inaccessible during inference; record resource usage.

No additional model-accuracy reproduction is needed to explain this specification.
The remaining work turns the verified baseline into an independently executable
agent benchmark task.

## Evidence locations

External asset root:
`/home/users/xy200/yinxunjian/mycode/ai4sci-artifacts/surface-code-decoding/20260919`.

- `data/source.json`: upstream archive identity and license.
- `alphaqubit-v1/models/*/{pretrain,finetune}/`: weights, configuration and training history.
- `alphaqubit-v1/packages/both/manifest.json`: frozen model/source hashes.
- `alphaqubit-v1/scores-{pretrain,finetune}.json`: full held-out counts and fitted scores.
- `alphaqubit-v1/verification/final-score-independent-audit-20260922.json`: independent reconstruction checks.
- `reproduction/alphaqubit-summary.json` in Git: compact measured reference results and evidence hashes.
- `classical-v1/scores.json`: matching/TN/BP comparisons under the same split and fits.
- `accounting.json` and `campaign.json`: actual allocations and original hard caps.
- `benchmark-spec/summary.html`: editable presentation of this draft.
