# Target-based drug screening

## 1. Overview

Develop one learned pocket-to-molecule ranking model that transfers across
DUD-E and LIT-PCBA targets. The primary reward is dataset-balanced
BEDROC(alpha=80.5), protocol drugclip-dataset-bedroc/v2. Architecture and training are open to research,
with no 50M parameter cap.

This track now has the native Harbor agent prompt, offline workspace,
portable model contract, atomic publication helper, and separate verifier.
The reference-reset DrugCLIP implementation and compact data protocol are
retained. CPU interface tests are distinct from GPU/container acceptance;
see [validation](docs/VALIDATION.md).

## 2. Data

The controller prepares ProFSADB (one fixed, filtered 500,000-record subset),
reconstructed BioLiP2/Q-BioLiP train/dev data, and author-released DUD-E and
LIT-PCBA test LMDBs. The compact training population differs from the paper's
exact population. Protein/ligand filtering and source revisions/hashes remain
in [data pipeline](environment/data/PIPELINE.md) and source_manifest.json.

The existing Hugging Face source is
[zifeng-ai/target-based-drug-screening](https://huggingface.co/datasets/zifeng-ai/target-based-drug-screening).
The default release is pinned to c8d2febc996e4c96b10f3aecd89fb7375150449f,
the revision used by the reported GPU-trained reference. The downloader
records its resolved commit; explicit revision overrides are supported and
must be identified separately in comparisons. The materializer requires passing
VALIDATION.json and .complete.json. Staging records every copied file's size
and SHA256 in separate development/private receipts.

[Prepare a run](environment/data/MATERIALIZE.md) provides the complete
controller commands: existing materialization, checksum-pinned generic
Uni-Mol assets, disjoint allowlisted development/private views, and run mounts.
The original source rebuild remains available through
TARGET_BASED_DRUG_SCREENING_DATA_SOURCE=upstream. The full-data launcher
conservatively requires 90 GiB free; the staging copies require additional
space equal to the selected training/test LMDBs plus about 191 MB of assets.

Development receives only processed training/dev records, split IDs, and
generic Uni-Mol initialization. Private receives the 102 DUD-E and 15 LIT-PCBA
target LMDBs. Author benchmark-trained weights, raw test label files, and
test-related metadata never enter the agent view. Source licenses remain
applicable; access to a public mirror does not grant additional rights.

## 3. Workspace

The image provides instruction.md, contract.md, data/README.md, baseline/,
references/, evaluation/, publish.py, and initially empty submission/ and
research/. Compose mounts only the controller-staged development subtrees
read-only and a fresh research directory writable.

The runnable scientific starting point is
[baseline/drugclip](environment/baseline/drugclip/README.md), whose training and inference sources are copied directly into
/workspace/baseline/drugclip at build time. Its dual Uni-Mol model has
95,305,418 parameters, uses generic molecule initialization, ProFSA pretraining
and BioLiP fine-tuning, and selects by BioLiP dev BEDROC. Its short train.sh
exports a package under research/; publish.py atomically installs the selected
package in submission/.

The minimal atom-histogram starter is copied into references/minimal-starter.
The eight-record smoke package scored v2 reward 0.028706 and EF1% 1.602231
over all 117 targets;
this is an interface check, not a full-data training result.
No trained benchmark model weights are baked into the agent image.

## 4. Evaluation

[contract.md](environment/contract.md) defines the infer.sh interface.
The controller snapshots the package and replays it in the separate offline
[verifier](tests/README.md). Labels are mounted beneath root-only /private.
The grader creates fresh field-whitelisted LMDBs with independent private
molecule permutations per target, runs candidate inference
as UID/GID 65534 with supplementary groups cleared, and then scores outputs
using its own trusted scorer. The package cannot access the labeled tree or
the reward file. Fresh numeric input keys refer to the shuffled view; the
original-index mapping and seed remain private and map predictions back to
the correct labels. Pocket order is preserved.

Every supplied ligand in every target must appear exactly once. The grader
checks complete 102+15-target coverage; omitted/duplicate targets, bad indices,
nonfinite predictions, inference failures, and malformed packages score 0.
Infrastructure failures produce no reward.

The primary reward is the mean of the DUD-E target-mean BEDROC(alpha=80.5)
and the LIT-PCBA target-mean BEDROC(alpha=80.5). Each dataset has weight 0.5;
targets within a dataset have equal weight. Both complete datasets are
required. This is already a 0–1 metric and receives no further transformation.
Random BEDROC depends on class prevalence, not a universal 0.5 anchor.

All six raw screening metrics and per-target/benchmark means are preserved
in `report.json`. Valid `reward.json` files contain eight numeric fields:
`reward`, `dataset_balanced_bedroc`, and the six `overall_mean_*` diagnostics
(ROC-AUC, BEDROC and EF at 0.5%, 1%, 2%, 5%). The latter remain equal-target
means for compatibility; `overall_mean_bedroc_80.5` is not the primary reward.
Reports identify `scoring_protocol: drugclip-dataset-bedroc/v2`. Invalid
packages receive zero; partial scientific evaluations have no primary score.
Exact ties are averaged over within-tie orders. BioLiP in-batch development
BEDROC is used for selection and is not the held-out screening score.

The verified epoch-14 baseline is **0.166385** (DUD-E 0.294588; LIT-PCBA
0.038182). [Metric rationale and validation limits](docs/METRIC_DESIGN.md)
explain the scientific choice. Earlier task versions used `E/(1+E)` on mean
EF1%; those reports retain their original values. To migrate a trusted full
report containing all per-target metrics, without retraining or overwriting it:

```bash
python target-based-drug-screening/tests/reaggregate.py old-report.json new-v2-report.json
```

The migration recomputes from individual targets, rejects incomplete or invalid
reports, and records the source hash, old score and old protocol. A report is
controller evidence, never an alternative to submitting an inference package.

## 5. Running

From the repository root, configure the [shared runner](../../runner/README.md),
then prepare a fresh data root and run the task:

```bash
export EVAL_ROOT=/large-disk/eval
task=target-based-drug-screening
bash runner/prepare.sh "$task" "$EVAL_ROOT/data/$task"
HOURS=48 bash runner/run.sh "$task" drugclip-research-001
```

The preparation step writes `TASK_DATA_DIR` and `TASK_PRIVATE_DIR` to
`$EVAL_ROOT/envs/$task.env`. Reuse that file across runs. The runner creates
a fresh `TASK_RESEARCH_DIR` per session. To score a saved package directly:

```bash
bash runner/run_verify.sh drugclip-replay "$task" /large-disk/frozen-package
```

Run orchestration belongs to the shared runner; this task has no separate
controller. [RUN_HANDOFF.md](docs/RUN_HANDOFF.md) explains the shared lifecycle.

task.toml declares the external development budget (8 H100s, 48 hours) and
separate verifier resources. These settings are absent from the prompt.
Inference has one H100, 16 CPUs, 64 GiB RAM, and 22 hours; the verifier's
24-hour total allows preparation and scoring. Full-suite H100 timing is
unmeasured, so validate operational limits on the deployment host.

environment/Dockerfile and docker-compose.yaml are now the agent environment.
The controller's former Dockerfile/compose are preserved as
Materialize.Dockerfile/data-compose.yaml. prepare.sh and run-data.sh retain
their entry points. Neither compose sets network/GPU policy; Harbor reads it
from task.toml.

## 6. Results and validation

[RESULTS.md](environment/baseline/drugclip/RESULTS.md) records an operator-reported
A100 verification of the PR #94 reference: ProFSA completed 100 epochs on
500,000 pairs (8 A100s, 19.0 h); BioLiP stopped at epoch 34 (4 A100s, 19.6 h)
with patience 20. Epoch 14 was selected by dev BEDROC 0.4288. Python was 3.11,
PyTorch 2.6.0+cu124. After the inference fixes, the same checkpoint scored:

| Benchmark | Targets | ROC-AUC | BEDROC(80.5) | EF1% |
| --- | ---: | ---: | ---: | ---: |
| DUD-E | 102 | 0.769 | 0.295 | 18.056 |
| LIT-PCBA | 15 | 0.577 | 0.038 | 2.861 |
| Combined equal-target mean | 117 | — | — | **16.108** |

The initial EF1% 1.830 is superseded by this corrected-inference rescore.
The [reproduction record](environment/baseline/drugclip/reproduction/README.md)
includes the reported recipe and all 34 development BEDROC values. The run
was reported by Zifeng Wang, not rerun in this integration update. Two independent full-suite replays on 2026-11-04 recovered the reported
EF1% and legacy v1 reward 0.9415475887803699. Their per-target scores now give
v2 rewards 0.166385032576 and 0.166384906515, respectively. One used the upstream verifier, and the
other ran through Harbor 0.23 with the task image. The latter completed in
90 minutes on one shared RTX A6000. [Acceptance record](docs/ACCEPTANCE.md)

Historical experiments are retained for context:

- An eight-record, two-stage smoke passed on one RTX 3070.
- A 512-ProFSA / 2,048-BioLiP pilot scored one full DUD-E aa2ar target:
  ROC-AUC 0.72387, BEDROC 0.02858, EF1% 0.62066.
- A full-suite author-checkpoint evaluator control scored DUD-E EF1% 24.8276
  and LIT-PCBA EF1% 5.3661, combined 22.3325. Those benchmark-trained weights
  are forbidden candidate initialization.

The trained pilot and released-checkpoint control are different experiments.
The control is not a full-suite score for the locally trained reference.
Exact historical recipes, hashes, and run descriptions remain in the
[training record](environment/baseline/drugclip/TRAINING.md).
New integration evidence and outstanding checks are in
[docs/VALIDATION.md](docs/VALIDATION.md). The reproducible run record is
specified in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), and the required
controller handoff is in [docs/RUN_HANDOFF.md](docs/RUN_HANDOFF.md).

## 7. Limitations

The training population differs from the Science paper. The author reported
the full training run on A100s. Raw training logs, the exact training commit
and the original training seed are still unavailable. The released checkpoint,
independent replay conditions and scope of the tests on this hardware are
documented in [acceptance](docs/ACCEPTANCE.md). Timing at the declared H100
resource budget remains unmeasured.

## 8. Citation and licenses

- [DrugCLIP, Science](https://www.science.org/doi/10.1126/science.ads9530).
- [THU-ATOM/Drug-The-Whole-Genome](https://github.com/THU-ATOM/Drug-The-Whole-Genome),
  pinned at 9d0db73915219763b5bcc3b7de86d7c4edb5ed03.
- [THU-ATOM/ProFSA](https://github.com/THU-ATOM/ProFSA),
  pinned at a056baed91277059acba3836d5fbf3fccfbee605.
- [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol);
  generic molecule initialization and dictionary hashes are pinned in
  pretrained_manifest.example.json.
- DUD-E, LIT-PCBA, BioLiP/Q-BioLiP, ProFSADB, and upstream code/assets retain
  their respective source licenses and citation requirements.

