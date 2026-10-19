# Fruit Fly Neuron Reconstruction (FFN / PATHFINDER)

## Overview

Develop a trained neuron-reconstruction model from public electron microscopy.
The primary metric is local sparse normalized expected run length (nERL, higher
is better). The selected track uses one FFN checkpoint, ascending seeds, no
consensus, shared SENSE proposals and disjoint two-node SHAPE merging.

Native GPU training, reconstruction and scoring have completed, including
matched single-window and stitched eight-window development comparisons.
The selected starter is legacy FFN step 6750 plus one production-rule update.
Its development nERL is 0.0722669 on the reduced-context proxy and 0.0350171 on
the stitched eight-window evaluation. These are different evaluation scopes,
not the historical full-volume score or a new private-test result.

The formal Docker verifier has run end to end on H100s (about 7 h on four GPUs,
about 4 h on eight); an agent package with the baseline FFN recipe scored nERL
2.2e-5 there. See the [run record](docs/validation/docker-formal-run.md) and
[validation scope](docs/READINESS.md).

## Data

MaleCNS v1.0 supplies spatial training/validation/test regions with shared-ID
exclusions. Images use a 16nm grid; the recorded test scores a 512³ core inside
a 1762³ prediction. Original source, license and split evidence remains in
[paper/data protocols](upstream/pathfinder/PAPER_PROTOCOLS.md).

The [materializer](environment/data/MATERIALIZE.md) checks a pinned resource
inventory and writes disjoint development and private directories. The
maintainer exporter packages only selected data/runtime dependencies, excluding
predictions, trained weights and historical ledgers. The 14.54GB prepared
release is publicly hosted on
[Hugging Face](https://huggingface.co/datasets/sanxing/ai4sci-fruit-fly-brain),
with an immutable commit and manifest hash pinned in the downloader.
This publishes campaign-prepared resources; a new upstream download remains
unverified.

```bash
python3 -m pip install -r fruit-fly-neuron-reconstruction/environment/data/requirements.txt
bash fruit-fly-neuron-reconstruction/environment/data/materialize.sh \
  --cache "$TASK_DOWNLOAD_CACHE" \
  --data "$TASK_DATA_DIR" --private "$TASK_PRIVATE_DIR"
```

## Workspace

The agent receives training/validation resources read-only under
`/workspace/data`, baseline commands, results and an editable copy of the
reference code under `/workspace/baseline/pathfinder-fast`, the submission contract,
public reference descriptions and development scoring documentation.
`/workspace/research` is fresh writable scratch; `/workspace/submission` holds
the current data-only model package. The original experiment directory and
private suite are not mounted into development. Detailed test receipts are
maintainer records and are removed from the development image.

## Evaluation

`tests/test.sh` runs the selected protocol through `verify_single.py`. It stages
one FFN, one SENSE and one SHAPE checkpoint, validating hashes and architecture.
Submitted Python/native libraries are not executed. The trusted pipeline starts
from raw EM, performs FFN inference and context preparation, then complete
SENSE, FFN-pair and SHAPE processing. Prediction runs in a namespace whose mount
allowlist excludes the private suite, original submission and experiment history.
Scoring runs afterward on the frozen reference. There is no unsafe fallback.

Each FFN window grows until its seeds are used or it reaches 1,000,000 calls
(segments grown before the cap are kept); stitching splits objects into
face-connected pieces and clears pieces under 10 voxels, the same minimum FFN
growth applies. `reward.json` reports SHAPE nERL and raw metrics for all three arms. Invalid
submissions score zero; infrastructure/timeout failures produce status without
a reward. [Contract](environment/contract.md), [verifier details](tests/README.md).

## Running

Set `TASK_DATA_DIR`, `TASK_PRIVATE_DIR`, `TASK_RESEARCH_DIR` and a separate fresh
`TASK_VERIFIER_SCRATCH`. The Compose files specify only the required data/scratch
mounts plus the verifier's namespace settings. Harbor supplies model snapshots,
logs, GPU allocation and offline policy. `tests/test.sh` is the verifier command.
See [which execution paths have run](tests/README.md#execution-evidence).

The agent's budget is 48 hours on 8×H100 (64 CPUs, 512 GiB, 1 TiB). The
verifier runs on 8×H100 with 64 CPUs, 512 GiB and a 12-hour timeout; stage caps
are in `tests/limits.json`. Prediction uses every exposed GPU: seeds on all CPUs,
four FFN workers per GPU under CUDA MPS with the exact `norm32` LayerNorm kernel
(about 925 views/s per H100, twice the eager speed), and four SENSE/FFN-pair/SHAPE
shards per GPU. A measured full run took about 7 h on four H100s and is about
4 h on eight; FFN time depends on the checkpoint. The FPS binary must match the
target Python/Torch/CUDA ABI and GPU architecture.

## Results and validation

### Current development results

Scores are comparable within a row, using the same evaluation region and
scorer. The selected starter is stronger than the corrected-from-start
16300-step candidate in each of these measured comparisons.

| FFN development evaluation | Selected starter nERL | Corrected-from-start step 16300 nERL |
|---|---:|---:|
| Reduced-context 311-cube proxy | 0.0722669 | 0.0349967 |
| Full-context single window | 0.0707893 | 0.0406383 |
| Stitched eight windows | 0.0350171 | 0.0190529 |

These results show measurable model differences; the historical near-zero
full-volume score does not establish that development feedback is uninformative.
Local proxy improvements still require larger-region and stitched confirmation.
Eight-window evaluation recomputes nERL on the original reference graph rather
than averaging window scores. It excludes full-125-window/private-test and
SENSE/SHAPE evaluation. Coverage remains incomplete (52.26% omitted reference
nodes for the selected starter's eight-window result).

Measured evaluation costs vary with the checkpoint and scope:

| Evaluation | Hardware allocation | Measured elapsed time |
|---|---|---:|
| Step 5800, eight windows | 4 GPUs | 90 minutes (6 GPU hours) |
| Step 16300, one full-context window | 1 GPU | 1h29m11s |
| Step 16300, eight windows | 4 GPUs | 4h31m50s |

These native timings exclude training, queueing and prior failed attempts; they
are not H100 estimates or a guarantee that every candidate finishes in 2.5 hours.
The reduced-context proxy was 3.46–5.50x faster than full-window evaluation in
its four-checkpoint calibration, with score differences up to 26%.

Sources: [starter results](environment/baseline/pathfinder-fast/RESULTS.md),
[step-5800 receipt](docs/validation/random-start-multiblock.json),
[step-16300 receipt](docs/validation/random-start-final.json).

## Limitations

Sparse references and shared-neuron exposure limit biological independence.
The FFN development scores above do not establish a new full-volume
SENSE/SHAPE result. The paper's complete
traditional proposals, higher-order motifs and graph search are omitted.
FIB-25/BANIS+ public checks and older ensemble results are separate protocols.

The formal entry has run once under Docker, with a weak agent package; a
trained baseline's private-test score and an independent replay of the selected
two-stage training/export recipe remain unrecorded. The pinned prepared resource
release is public; fresh upstream materialization remains unverified. The
10-voxel fragment minimum and the call cap were added to make the verifier
complete for such packages; scores before and after are not comparable.

## Citation and licenses

[Paper/data protocols](upstream/pathfinder/PAPER_PROTOCOLS.md),
[PATHFINDER reference](upstream/pathfinder/paper_reference.json) and
[BANIS+ reference](upstream/pathfinder/baseline_reference.json) record source
roles and license/provenance information. Runtime bundles retain dependency
sources/licenses. Data and weights are not redistributed in this Git PR.

Historical receipts use public aliases for execution locations and allocation IDs; see [public provenance and native configuration](docs/PUBLIC_PROVENANCE.md).
