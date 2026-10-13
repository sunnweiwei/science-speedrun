# Runtime and evaluation contract

The development container mounts only the `development/` HF role, under `/workspace/data`.
Its 130 conditions contain physically separate train and validation arrays,
ideal circuits, and the published `pij_from_even_for_odd.dem` calibration model.
No full 50,000-shot raw files, odd-fit DEMs, raw measurement/sweep records,
historical decoder predictions, or test labels are in this role. No trained
weights are shipped: `reference/` holds the reference AlphaQubit's results
(per-shot validation predictions, validation metrics, aggregate test score and
training cost) and a weightless submission template. Every submitted model is
trained from random initialization in the session. Synthetic draws are generated online.

There is no network during development. Final inference receives one GPU for at most one hour, with no
network and no package installation. The trusted grader, a separate CPU service, never runs submitted code.

## Artifact

`/workspace/submission` may be read at any moment, so replace it as a whole: assemble the new
package elsewhere and swap it in with one rename (`python -m qecbench.package ... --output
/workspace/submission` does this). It must include:

- `infer.sh`, invoked as `bash infer.sh INPUT_ROOT OUTPUT_ROOT`;
- `README.md`, `requirements.lock`, and ordinary relative source files;
- `model_manifest.json` with schema `surface-code-decoding-model/v1`, a nonempty
  `checkpoints` list (`path`, `sha256`, `format`), and relative `model_config` and
  `training_provenance` JSON paths;
- a JSON object describing the architecture/configuration and a truthful JSON
  provenance object describing initial weights, training data and selection;
- serialized learned parameters or decoder state. The package limit is 50 GiB.

No symlinks are accepted (hard links are fine: the verifier scores a copy). The trusted supervisor verifies declared
checkpoint hashes, copies the submitted tree into an immutable relocated
snapshot, and launches inference as UID 10001 with no new privileges. The model
can write its output and temporary files. Submitted code never runs in the
trusted grader. Dependencies must already be in the pinned runtime; the verifier
does not execute installation requests from `requirements.lock`.

## Input and output

The input manifest schema is `surface-code-decoding-input/v1`. Each listed
condition gives `name`, `group`, `distance`, `basis`, `rounds`, `shots`,
`num_detectors`, `arrays`, `arrays_sha256`, `circuit`, and `circuit_sha256`.
Canonical names are `surface_code_bX_d3_r01_center_3_5`, etc. Conditions and shots
may be reordered; any supported subset and positive N are valid inference
inputs. Official scoring requires all 130 conditions. NPZs contain only unique
uint64 sample IDs and packed uint8 detection events, with unused high bits zero.

Output files are `<condition-name>.npz` with exactly uint64 `sample_ids` and
float32 `probabilities`, both shape `[N]`. IDs are aligned by value, not row
position. NaNs, infinities, out-of-range values, duplicate/unknown/missing IDs,
missing files, incompatible dtypes, oversized archives, unsafe links, encrypted
ZIPs, and malformed NumPy headers are invalid predictions. Numeric NumPy v1/v2
arrays, stored or deflate ZIP encoding, are accepted. Pickle is forbidden.

The label meaning is `official_path_observable_flip`. Feature reconstruction
uses hard detection events and the ideal circuit; analogue readout, leakage,
and raw final-data parity are not part of this release.

## Isolation and reward

`tests/docker-compose.yaml` has two network-disabled services:

1. `main` sees the relocated submission and `verifier/` inputs only, plus a
   root-protected sealing volume. It runs the submitted program as UID 10001,
   kills surviving child processes, and seals regular prediction files.
2. `grader` sees `verifier/` inputs, `grader/` labels and sealed outputs, but no
   submitted source or development workspace. Its trusted fixed code checks
   input/snapshot/label checksums and computes the score.

A unique run token ties the sealed snapshot to the result. Root-protected result
paths prevent model-written `reward.json` from being accepted. Invalid model
artifacts, timeout, nonzero inference exit, or malformed predictions emit score
0 with `model_quality_measured=false`. Missing/corrupt mounted datasets, invalid
trusted calibration, network-isolation failure, snapshot corruption, and other
infrastructure errors remove reward files and emit `infrastructure_error`.
The Harbor verifier timeout is 3900 seconds, including at most 3600 inference
seconds and 300 grading seconds. Use fresh named volumes for each replay.

Only the trusted Git/Docker verifier's scorer and `score_config.json` define the
official result. A scorer copy inside a submission or the development release
is non-authoritative and is never imported by the trusted grader.

`score_config.json` fixes the anchors and version. P is one minus equal-distance,
equal-group, equal-duration failure, with a decision threshold of 0.5. Normalized
score = max((P-L)/(U-L),0) with no upper clip, L=0.7160463461538461 (MWPM), U=0.7615159615384615
(released TN rescored on identical shots). This upper anchor is not a newly
reimplemented TN decoder and is not an assertion of current literature SOTA.
Raw P, raw failure, unclipped score, fit-based scientific metrics and calibration
remain in `report.json`; `reward.json` contains only `{"reward": P}` (raw P; the normalized score is a
report diagnostic). Poor valid
models can lack a valid log-fidelity LER fit without losing their primary score.

The public historical Sycamore source prevents strong secrecy claims. Runtime
role isolation protects this run from direct answer access; it does not eliminate
prior memorization or establish prospective scientific generalization.
