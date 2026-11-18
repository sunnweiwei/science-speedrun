# Brain–Computer Interface Decoding (V4)

## Overview

The agent develops one EEG model for four clinical and brain–computer interface (BCI) classification
tracks with one jointly trained backbone:

- **TUEV** clinical EEG events (6 classes, 16 bipolar channels, 5 s windows);
- **AD65** dementia diagnosis: Alzheimer's disease, frontotemporal dementia, or healthy control (3 classes, 19 channels, 10 s windows);
- **OpenBMI** left/right motor imagery (2 classes, 62 channels, 4 s trials);
- **Longitudinal** resting-state eyes closed/open across multi-day visits (2 classes, 58 channels, 2 s windows).

Track-specific input adapters and heads are allowed. Every track is tested on held-out
subjects/patients (`TUEV`: the official patient-disjoint evaluation set) under **per-subject /
per-recording-session isolated inference**, and scored by **subject/session-stratified balanced
accuracy** $\text{BA}_k \in [0, 1]$. The primary V3/V4 benchmark score (`reward.json["score"]` and
`reward.json["reward"]`) averages chance-adjusted track skills across the four active tracks and
clips to $[0, 1]$:

$$\text{score} = \operatorname{clip}\!\left(\frac{1}{4}\sum_{k \in \{\text{tuev},\text{ad65},\text{openbmi},\text{longitudinal}\}} \frac{\text{BA}_k - c_k}{1 - c_k},\; 0,\; 1\right), \quad c = \left\{\text{tuev}:\tfrac{1}{6},\;\text{ad65}:\tfrac{1}{3},\;\text{openbmi}:\tfrac{1}{2},\;\text{longitudinal}:\tfrac{1}{2}\right\}.$$

**V4 Protocol Updates:**
1. **Removal of `FACED`**: The legacy `FACED` emotion track is removed from active training mounts, contracts, and verification because identical video clips recur chronologically across subjects (allowing stimulus-locked EOG/ocular artifacts to leak clip identity).
2. **Removal of `BIOT`**: `BIOT` (`braindecode/biot-pretrained-six-datasets-18chs`) is removed from authorized pretrained weights and blocked by metadata and SHA-256 in the verifier due to direct pretraining exposure to `TUEV`. Only `cbramod` and `codebrain` (plus from-scratch initialization such as `eegnet` and `contrawr`) are authorized.
3. **Per-Subject / Per-Session Isolated Inference**: `tests/main_verifier.py` invokes `infer.sh` independently for each held-out `(subject, recording_session)` unit using freshly materialized single-unit shards, preventing transductive cohort-wide feature whitening or cross-subject BatchNorm adaptation.

Braindecode 1.8.1 provides the authorized baselines: two pretrained shared references (`CBraMod`,
`CodeBrain`; the form a submission takes) and four single-task specialist families (`EEGNet`,
`ContraWR`, `CBraMod`, `CodeBrain`; per-track capacity references only). The best authorized shared
reference, `CodeBrain`, scores `55.77 / 59.75 / 63.31 / 76.89` (%, five-seed test mean across
`TUEV / AD65 / OpenBMI / Longitudinal`, V3/V4 composite `0.4174`).

## Data

| Track | Source and license | Train / val / test (windows; test subjects) |
| --- | --- | --- |
| TUEV | [TUH EEG Events](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml), TUH data use agreement | 68,445 / 15,487 / 29,421; official patient-disjoint evaluation set |
| AD65 | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) derivatives, CC0 | 3,774 / 1,343 / 1,376; 18 patients |
| OpenBMI | [GigaDB 100542](https://gigadb.org/dataset/100542) MI sessions, CC0 | 12,800 / 4,400 / 4,400; 11 subjects |
| Longitudinal | [OpenNeuro ds007176](https://openneuro.org/datasets/ds007176), CC0 | 28,523 / 8,418 / 10,028; 9 subjects |

`bash environment/data/materialize.sh release ROOT` installs the four active tracks and two
allowlisted pretrained checkpoints into `ROOT/runtime/development` (`TASK_DATA_DIR`) and
`ROOT/runtime/private` (`TASK_PRIVATE_DIR`) from the pinned sources in
[download-source.json](environment/data/download-source.json), verifying every file:

- AD65, OpenBMI and Longitudinal: the public prepared release
  [sunweiwei/ai4sci-brain-computer-interface-decoding-data](https://huggingface.co/datasets/sunweiwei/ai4sci-brain-computer-interface-decoding-data),
  built by `materialize.py` from the pinned public sources (about 10 GB);
- TUEV: the runtime tree of the controlled-access release
  [AI4Sci-BCI/AI4BCI_private_dataset](https://huggingface.co/datasets/AI4Sci-BCI/AI4BCI_private_dataset).
  Access requires TUH provider terms and owner approval (`HF_TOKEN`);
- `pretrained/`: the two audited initialization checkpoints (`bci-cbramod.safetensors`,
  `bci-codebrain.safetensors`) pinned in `environment/baseline/braindecode/pretrained.lock.json`
  from the public `braindecode/*` Hugging Face repositories (BSD-3-Clause).

Run it in an environment with `environment/requirements-data.txt` and `huggingface_hub`.
[MATERIALIZE.md](environment/data/MATERIALIZE.md) documents the rebuild from original sources.

## Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md`, `/workspace/contract.md` | task prompt, 4-track package contract, `infer.sh` schema, per-subject/session isolation rules, V3/V4 score formula |
| `/workspace/data/` | `README.md`; `{tuev,ad65,openbmi,longitudinal}/{train,val}` and `pretrained/` (read-only mounts) |
| `/workspace/baseline/braindecode/` | shared and single-task training code, inference, `publish.py`; `RESULTS.md`/`RESULTS.json` |
| `/workspace/references/` | provenance & overlap audit of pretrained checkpoints (`PRETRAINED_AUDIT.md`) |
| `/workspace/evaluation/` | the verifier's metric code (`bci_score.py`, `data_identity.py`) |
| `/workspace/submission/`, `/workspace/research/` | the current package (empty at start) and writable scratch |

## Evaluation

The separate verifier (`tests/Dockerfile`, `tests/main_verifier.py`) copies the package, enforces
the pretrained checkpoint policy (rejecting `BIOT` and unapproved weights), partitions each track's
held-out windows by `(subject, recording_session)`, stages single-unit shards without labels, and
runs `infer.sh` as UID 10001 with no capabilities and no network, up to two hours per track. Labels,
scoring code and the reward directory are root-only, and a readability probe fails closed before
any package code runs. Subject/session-stratified track metrics:

- **TUEV**: 6-class balanced accuracy across the official patient-disjoint evaluation set;
- **AD65**: 3-class balanced accuracy over patients after averaging window probabilities per patient;
- **OpenBMI**: mean of per-subject balanced accuracies across held-out subjects;
- **Longitudinal**: per-session balanced accuracy, averaged within subjects, then over subjects.

On a full 4-track run, `reward.json` emits `score`, `reward`, `signed_score`, and the four track
balanced accuracies (`tuev`, `ad65`, `openbmi`, `longitudinal`), matching
`harness/ssi_index_v3.py:bci()` to `< 1e-12`. `report.json` retains full diagnostics (window vs
subject/session balanced accuracy, weighted/macro F1, Cohen's $\kappa$, confusion matrices,
per-subject scores, isolated unit counts, data identities, and package hash).

## Running

```bash
bash environment/data/materialize.sh release /large-disk/bci
export TASK_DATA_DIR=/large-disk/bci/runtime/development
export TASK_PRIVATE_DIR=/large-disk/bci/runtime/private
export TASK_RESEARCH_DIR=/large-disk/runs/bci-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p brain-computer-interface-decoding -a <agent> ...
```

Operator tools (image builds, protocol tests, frozen-package replay) are in
[tests/README.md](tests/README.md).

## Results and validation

Five-seed test means (%) across the four V4 tracks:

| model | TUEV | AD65 | OpenBMI | Longitudinal | V3/V4 Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant class | 16.67 | 33.33 | 50.00 | 50.00 | 0.0000 |
| shared CBraMod, pretrained | 40.71 | 51.65 | 57.41 | 74.54 | 0.3006 |
| shared CodeBrain, pretrained | 55.77 | 59.75 | 63.31 | 76.89 | 0.4174 |
| best single-task specialist | 61.36 | 49.02 | 77.30 | 76.03 | 0.4596 |

## Citation and licenses

**Authorized baseline models**

- **EEGNet:** Lawhern, V. J., et al. (2018).
  [EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://doi.org/10.1088/1741-2552/aace8c).
  *Journal of Neural Engineering*, 15(5), 056013.
- **ContraWR:** Yang, C., Xiao, C., Westover, M. B., and Sun, J. (2023).
  [Self-Supervised Electroencephalogram Representation Learning for Automatic Sleep Staging: Model Development and Evaluation Study](https://doi.org/10.2196/46769).
  *JMIR AI*, 2, e46769.
- **CBraMod:** Wang, J., et al. (2025).
  [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://arxiv.org/abs/2412.07236).
  *International Conference on Learning Representations (ICLR)*.
- **CodeBrain:** Ma, J., et al. (2026).
  [CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model](https://arxiv.org/abs/2506.09110).
  *International Conference on Learning Representations (ICLR)*.

**Original datasets**

- **TUEV:** Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., and Picone, J. (2015).
  [Improved EEG Event Classification Using Differential Energy](https://doi.org/10.1109/SPMB.2015.7405421).
  *IEEE Signal Processing in Medicine and Biology Symposium (SPMB)*, pp. 1–4.
- **AD65:** [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504), CC0.
- **OpenBMI:** Lee, M.-H., et al. (2019).
  [EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy](https://doi.org/10.1093/gigascience/giz002).
  *GigaScience*, 8(5), giz002.
- **Longitudinal Eyes:** [OpenNeuro ds007176](https://openneuro.org/datasets/ds007176), CC0.
