# Earth Observation Pretraining and Transfer

## 1. Overview

Develop one from-scratch geospatial encoder whose frozen features improve AWF
classification and LFMC moisture regression. The agent may change the encoder,
training objective, auxiliary targets, schedule and implementation while keeping
the data restrictions and evaluation contract fixed.

The starting recipe is official OlmoEarth-v1 Large trained for 6,000 WorldCover
updates followed by 1,000 updates of WorldCover and NDMI moment prediction. Its
recorded validation scores are **77.034884% AWF accuracy** and **21.459503 LFMC
MAE**, from one adaptively selected seed. These are development scores, not a
measured final verifier reward or a reproduction of the new container workflow.

Read [the agent instructions](instruction.md), [baseline commands](environment/baseline/starter/README.md),
[baseline results](environment/baseline/starter/RESULTS.md), and
[submission contract](environment/contract.md). The workspace follows the
repository's [task standard](../TASK_STANDARD.md); deployment gaps are listed below.

## 2. Data

[Operator preparation](environment/data/MATERIALIZE.md) uses the pinned public
sources in [sources.json](environment/data/sources.json). The two pretraining
shards and evaluation downloads total about 148 GB before extraction, runtime,
development copies and checkpoints. No credentials are required. Geographic
exclusions retain 19,837 pretraining samples.

Provide read-only development mounts containing pretraining samples and complete
AWF/LFMC train/validation views. Keep LFMC test data in the verifier-only mount.
The [agent data guide](environment/data/README.md) describes the contents and
allowed uses. Datasets and weights are not committed to Git.

## 3. Workspace

The development image explicitly copies these files; it does not rely on a live
repository checkout:

| Agent path | Purpose |
|---|---|
| `/workspace/instruction.md`, `/workspace/contract.md` | Goal, rules, submission interface and score definition. |
| `/workspace/data/` | Data guide and source manifest, plus operator-mounted read-only datasets. |
| `/workspace/baseline/starter/` | Existing preparation/training/export code with a README and recorded RESULTS. |
| `/workspace/evaluation/` | Development replay, fixed probes, raw features and paired comparisons. |
| `/workspace/research/` | Writable derived data, checkpoints, experiment notes and evaluation outputs. |
| `/workspace/submission/` | One complete selected encoder package for independent replay. |

`/workspace/starter` is a compatibility alias for the baseline code. Source files
live in `environment/baseline/starter/` with its README and RESULTS. The pinned runtime is `/opt/olmoearth`, with its Python on PATH.

## 4. Evaluation

Use [development evaluation](environment/evaluation/README.md) for full AWF and
LFMC validation. The same encoder serves both tasks; frozen probe splits,
normalization, missingness handling, pooling and search grids remain fixed.
The supplied commands also compare matched random and raw-feature controls.

The [independent verifier](tests/README.md) replays the delivered package with
only unlabeled inputs visible to inference. It fits probes outside model
execution and reports the existing reward:

`0.5 * (AWF_validation_accuracy + max(0, 1 - LFMC_test_MAE / max(constant_test_MAE, 1e-12)))`.

The constant predictor is fitted from LFMC training labels. Development uses
LFMC validation; final test scoring follows candidate selection. The recorded
baseline validation pair must not be substituted into a reported final reward.

## 5. Running

Prepare the data and runtime outside the agent session using the
[materialization guide](environment/data/MATERIALIZE.md). The development build
context is the task directory:

```bash
docker build -f earth-observation-pretraining-transfer/environment/Dockerfile -t earth-observation-pretraining-transfer-agent earth-observation-pretraining-transfer
```

`environment/docker-compose.yaml` declares this build context. The development and verifier Compose files wire their separate data mounts.
Export `TASK_DATA_DIR`, `TASK_PRIVATE_DIR` and `TASK_RESEARCH_DIR` as documented
in the materialization guide. `task.toml` declares offline operation and a default
48-hour agent budget; the controller enforces these, allocates GPUs, snapshots
the published submission and collects verifier results. A complete Harbor launch
has not been exercised on this cluster. No cluster scheduler,
account or partition configuration is required by the documented workflow.

Inside the prepared workspace, the [baseline README](environment/baseline/starter/README.md)
walks through cache preparation, both training phases and export. The
[evaluation guide](environment/evaluation/README.md) consumes those packages.
Check the pipeline with a short run before starting a new research trajectory;
the recorded baseline scores need not be re-established as a prerequisite.

## 6. Results and validation

[RESULTS.md](environment/baseline/starter/RESULTS.md) records the selected recipe's
metrics, controls, uncertainty, seeds and measured costs. The moisture phase
ran for 11.77 minutes on one H200; validation took 26.27 minutes. These costs
exclude the 6,000-update parent phase and data preparation.

Previous CPU export and relocated-package checks passed batch-independence,
masked-input and fully missing-observation checks. They do not establish an
end-to-end container reproduction. Delivery checks cover atomic publication, explicit invalid-submission rewards,
configuration and repository paths; see [readiness](docs/READINESS.md). No new
training or benchmark experiment was run for the delivery changes.

## 7. Limitations

- Results are conditional on one adaptively selected seed; independent-seed
  replication and untouched-test generalization are unestablished.
- AWF has no separate test split. Public LFMC test was exposed during earlier
  development. The frozen point probes do not reproduce the paper's complete
  trained downstream heads.
- Docker/Harbor execution is unverified because this cluster does not support
  Docker acceptance. This is an accepted best-effort limitation, not a merge gate.
  The controller must enforce the declared policy and snapshot submissions.
  Unexplained inference exits require operator diagnosis and do not generate a
  synthetic score. Historical external-LFMC acquisition scripts are outside the
  delivered AWF/LFMC frozen-probe workflow; their metadata-selection policy file
  is not included.
- Minimum GPU memory and equivalent performance on smaller cards are unestablished.

## 8. Citation and licenses

[Sanxing Chen](https://github.com/sanxing-chen) contributed the original task
and official OlmoEarth replication. [James Cai](https://github.com/jchy20)
contributed the two-phase WorldCover/moisture baseline and the agent workspace
and baseline-guide organization.
The [current baseline results](environment/baseline/starter/RESULTS.md) are the
track's result reference; the [evaluation protocol](docs/PROTOCOL.md) defines
the released populations and frozen-probe semantics.

- [Official implementation](https://github.com/allenai/olmoearth_pretrain/tree/0497dfbb6711ded4e6bf10cf089fc1e4d58c186b).
- [Official pretraining data](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset), including WorldCover maps.
- [AWF release](https://huggingface.co/datasets/allenai/olmoearth_projects_awf/tree/da8eb6aab65b518cb8223912ba9091cbaa5dad2f) and [LFMC documentation](https://github.com/allenai/olmoearth_projects/blob/main/docs/lfmc.md).

Retain the [OlmoEarth Artifact License](docs/LICENSE-OlmoEarth) and the source
attributions in the data guide.
