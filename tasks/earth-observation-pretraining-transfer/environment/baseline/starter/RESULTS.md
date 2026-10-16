# Recorded starter results

The [Large WorldCover and moisture recipe](README.md) scored **77.034884%
AWF validation accuracy** and **21.459503 LFMC validation MAE** after
6,000 WorldCover updates and 1,000 moisture-adaptation updates. These are
measurements from a completed external run, not a reproduction inside the
current agent container.

## Validation metrics

The selected checkpoint is phase-two update 1,000, or 7,000 cumulative encoder
updates. Initialization/data seeds are 12536/3622. Inference used an H200,
batch size 16 and hash seed 0, with the same frozen encoder for both tasks.

| Metric | Matched Large random | Raw features | Trained starter |
|---|---:|---:|---:|
| AWF accuracy (%) ↑ | 63.372093 | 66.860465 | **77.034884** |
| LFMC MAE ↓ | 22.654226 | 22.319477 | **21.459503** |

Secondary metrics are AWF balanced accuracy **65.904544%**, LFMC
RMSE **29.747444** and R² **0.330995**. Validation selected logistic C=0.1
and ridge alpha=100. AWF uses 1,115 training / 344 validation rows; LFMC uses
31,175 / 8,262. Training labels fit frozen probes; neither task's labels
supervise the encoder or its auxiliary heads.

| Paired improvement | Point gain | Spatial/site 95% interval |
|---|---:|---:|
| AWF accuracy versus matched random, pp | 13.662791 | [7.739892, 19.572519] |
| LFMC MAE reduction versus matched random | 1.194723 | [0.619951, 1.766122] |
| AWF accuracy versus raw features, pp | 10.174419 | [3.618051, 16.201288] |
| LFMC MAE reduction versus raw features | 0.859974 | [0.018506, 1.702505] |

The bootstrap uses 5,000 replicates, seed 42, 76 AWF spatial blocks and
177 LFMC sites. These intervals condition on one training seed and adaptive
validation development. They do not adjust for recipe search or establish
across-seed replication.

AWF has no independent test split in the release, and public LFMC test was
exposed in earlier development. No untouched-test claim is made. Frozen point
probes also differ from the paper's trained downstream decoders. The fixed
[evaluation guide](../../evaluation/README.md) documents the development
protocol.

## Measured cost

| Recorded stage | Time |
|---|---:|
| Moisture adaptation, one H200 | 705.911 seconds (11.77 minutes) |
| CPU export | 25.234 seconds |
| Full validation | 1,575.988 seconds (26.27 minutes) |

The phase-two allocation used **0.644036 H200-hours**, including CPU/startup
work. These are incremental costs: they exclude the 6,000-update WorldCover
phase, shared data/cache preparation and baselines. End-to-end reproduction
must train both phases and costs more. Minimum GPU memory requirements and
equivalent performance on smaller cards have not been established.

## Recipe and reproduction limits

NDMI mean and standard deviation summarize observed temporal slots; they do
not directly supervise moisture at the LFMC measurement date. Monthly
pretraining and fortnightly LFMC cadence differ, and missing acquisition bins
can be compressed in LFMC inputs. Reversing the pretraining input order does
not remove these mismatches. There is no explicit pixel cloud mask.

The commands in [README.md](README.md) implement the scientific recipe with
the supplied code. They do not promise byte-identical checkpoints or numerical
identity to these measurements: source revisions, runtime details and GPU
nondeterminism can differ. Each fresh run records its actual implementation,
data identities, seeds, complete phase-one ancestry and cumulative updates.

Retain the runtime lock, cache/view manifests, checkpoint sidecars, training
sources, exported provenance and validation predictions with each reproduction.
CPU export and relocated-package checks verified batch independence, unchanged
features after perturbing masked inputs, and zero features for all-unobserved
samples. These synthetic checks use no downstream test labels and do not
establish an independent training reproduction or container verification.
