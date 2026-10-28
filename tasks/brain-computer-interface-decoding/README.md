# Brain–Computer Interface Decoding

## Overview

The agent develops one EEG model for five classification tracks with one jointly trained
backbone: FACED emotion (9 classes), TUEV clinical events (6), AD65 dementia diagnosis (3),
OpenBMI left/right motor imagery (2) and eyes closed/open in a longitudinal resting-state study
(2). Track-specific input adapters and heads are allowed. Every track is tested on held-out
subjects (TUEV: the official evaluation set) and scored by its own balanced-accuracy metric; the
five rewards have equal standing and there is no aggregate.

Braindecode 1.8.1 provides the baselines: three pretrained shared references (CBraMod, BIOT,
CodeBrain; the form a submission takes) and five single-task specialist families (EEGNet,
ContraWR, CBraMod, BIOT, CodeBrain; per-track references only). The best shared reference,
CodeBrain, scores 19.58 / 55.77 / 59.75 / 63.31 / 76.89 (% , five-seed test mean).

**Status.** Maintainer acceptance 2026-10-28: all 15 shared references, the per-track best
specialists and the RSI package were rescored through the verifier on the released data and
reproduce the author's numbers (largest difference 0.0002); see
[docs/acceptance-2026-10.md](docs/acceptance-2026-10.md). The workspace follows
[TASK_STANDARD.md](../TASK_STANDARD.md).

## Data

| Track | Source and license | Train / val / test (windows; test subjects) |
| --- | --- | --- |
| FACED | [Synapse syn50614194](https://www.synapse.org/Synapse:syn50614194/datasets/), access terms of the provider | 6,720 / 1,680 / 1,932; 23 |
| TUEV | [TUH EEG Events](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml), TUH data use agreement | 68,445 / 15,487 / 29,421; official evaluation set |
| AD65 | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) derivatives, CC0 | 3,774 / 1,343 / 1,376; 18 |
| OpenBMI | [GigaDB 100542](https://gigadb.org/dataset/100542) MI sessions, CC0 | 12,800 / 4,400 / 4,400; 11 |
| Longitudinal | [OpenNeuro ds007176](https://openneuro.org/datasets/ds007176), CC0 | 28,523 / 8,418 / 10,028; 9 |

`bash environment/data/materialize.sh release ROOT` installs everything into
`ROOT/runtime/development` (`TASK_DATA_DIR`) and `ROOT/runtime/private` (`TASK_PRIVATE_DIR`) from
three pinned sources in [download-source.json](environment/data/download-source.json), verifying
every file:

- AD65, OpenBMI and Longitudinal: the public prepared release
  [sunweiwei/ai4sci-brain-computer-interface-decoding-data](https://huggingface.co/datasets/sunweiwei/ai4sci-brain-computer-interface-decoding-data),
  built by `materialize.py` from the pinned public sources (about 10 GB);
- FACED and TUEV: the runtime trees of the controlled-access release
  [AI4Sci-BCI/AI4BCI_private_dataset](https://huggingface.co/datasets/AI4Sci-BCI/AI4BCI_private_dataset)
  (about 20 GB). Access requires the providers' terms (Synapse, TUH) and the owner's approval;
  `HF_TOKEN` must belong to an approved account;
- `pretrained/`: the three initialization checkpoints pinned in
  `environment/baseline/braindecode/pretrained.lock.json`, from the public `braindecode/*`
  Hugging Face repositories (BSD-3-Clause).

Run it in an environment with `environment/requirements-data.txt` and `huggingface_hub`.
[MATERIALIZE.md](environment/data/MATERIALIZE.md) documents the rebuild from the original
sources (downloaders with pinned hashes, the CBraMod FACED/TUEV preprocessing, the subject
splits and the TUEV train/validation regrouping) and the author's model release.

## Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md`, `/workspace/contract.md` | the prompt; package, `infer.sh` input/output formats, inference limits |
| `/workspace/data/` | `README.md`; `{faced,tuev,ad65,openbmi,longitudinal}/{train,val}` and `pretrained/` (read-only mounts) |
| `/workspace/baseline/braindecode/` | shared and single-task training code, inference, `publish.py`; `RESULTS.md`/`RESULTS.json` (every seed's test scores); `reproduction/` (validation histories of the 15 shared references) |
| `/workspace/references/` | provenance audit of the pretrained checkpoints |
| `/workspace/evaluation/` | the verifier's metric code (`bci_score.py`, `data_identity.py`) |
| `/workspace/submission/`, `/workspace/research/` | the current package (empty at start) and writable scratch |

## Evaluation

The separate verifier (`tests/Dockerfile`, `tests/main_verifier.py`) copies the package, stages
each track's test inputs without labels and runs `infer.sh` as UID 10001 with no capabilities,
no network and one GPU, up to two hours per track. Labels, scoring code and the reward directory
are root-only, and a readability probe fails closed before any package code runs. Rewards:

- FACED, TUEV: balanced accuracy over windows;
- AD65: balanced accuracy over patients after averaging window probabilities per patient;
- OpenBMI: mean of per-subject balanced accuracies;
- Longitudinal: per-session balanced accuracy, averaged within subjects, then over subjects.

`reward.json` holds the five track rewards (`faced`, `tuev`, `ad65`, `openbmi`, `longitudinal`)
and no aggregate. `report.json` adds status, accuracy, weighted/macro F1, Cohen κ, confusion
matrices, per-subject scores, data identities and the package hash. An invalid package scores 0
on the affected track; missing evaluation data is an infrastructure failure with no reward. A
full verification of a shared CodeBrain package takes about 4 minutes on one H100.

## Running

```bash
bash environment/data/materialize.sh release /large-disk/bci          # section Data
export TASK_DATA_DIR=/large-disk/bci/runtime/development
export TASK_PRIVATE_DIR=/large-disk/bci/runtime/private
export TASK_RESEARCH_DIR=/large-disk/runs/bci-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p brain-computer-interface-decoding -a <agent> ...
```

Resources (`task.toml`): the agent gets one H100, 64 CPUs, 128 GiB and 500 GiB for 48 hours
(the standard 1× H100 budget; one shared reference trains in 15–55 minutes for BIOT and
CBraMod and 2.5–4 hours for CodeBrain on one GPU). The verifier
uses one H100, 8 CPUs and 64 GiB. Operator tools (image builds, protocol tests, frozen-package
replay, shared-model campaigns) are in [tests/README.md](tests/README.md).

## Results and validation

Five-seed test means (%), from
[RESULTS.md](environment/baseline/braindecode/RESULTS.md):

| model | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant class | 11.11 | 16.67 | 33.33 | 50.00 | 50.00 |
| shared CBraMod, pretrained | 14.91 | 40.71 | 51.65 | 57.41 | 74.54 |
| shared BIOT, pretrained | 16.75 | 50.69 | 59.71 | 59.12 | 74.22 |
| shared CodeBrain, pretrained | 19.58 | 55.77 | 59.75 | 63.31 | 76.89 |
| best single-task specialist | 42.22 | 61.36 | 49.02 | 77.30 | 76.03 |

The author (Jingying Ma) trained and scored all packages; the maintainer replay is in
[docs/acceptance-2026-10.md](docs/acceptance-2026-10.md). One author-run 40-hour agent research
session and its five-seed replication are recorded in [docs/rsi-experiment.md](docs/rsi-experiment.md),
outside the agent's workspace.

## Limitations

- FACED and TUEV cannot be redistributed; operators need the providers' approval and access to
  the controlled release, or must rebuild from the sources.
- The shared references are well below the single-task specialists on FACED (about 20% against
  40%), so the shared-backbone rule itself is a large part of the difficulty.
- Test sets are small for AD65 (18 patients), OpenBMI (11 subjects) and Longitudinal (9
  subjects); five-seed standard deviations reach 10 points on AD65.
- Public pretraining corpora of the checkpoints can overlap TUEV/TUEG; see the provenance audit.
- The UID boundary of the verifier is not a hardened multi-tenant sandbox.

## Citation and licenses

**Baseline models**

- **EEGNet:** Lawhern, V. J., et al. (2018).
  [EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces](https://doi.org/10.1088/1741-2552/aace8c).
  *Journal of Neural Engineering*, 15(5), 056013.
- **ContraWR:** Yang, C., Xiao, C., Westover, M. B., and Sun, J. (2023).
  [Self-Supervised Electroencephalogram Representation Learning for Automatic Sleep Staging: Model Development and Evaluation Study](https://doi.org/10.2196/46769).
  *JMIR AI*, 2, e46769.
- **CBraMod:** Wang, J., et al. (2025).
  [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://arxiv.org/abs/2412.07236).
  *International Conference on Learning Representations (ICLR)*.
- **BIOT:** Yang, C., Westover, M. B., and Sun, J. (2023).
  [BIOT: Biosignal Transformer for Cross-data Learning in the Wild](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html).
  *Advances in Neural Information Processing Systems*, 36.
- **CodeBrain:** Ma, J., et al. (2026).
  [CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Model](https://arxiv.org/abs/2506.09110).
  *International Conference on Learning Representations (ICLR)*.

**Original datasets**

- **FACED:** Chen, J., Wang, X., Huang, C., Hu, X., Shen, X., and Zhang, D. (2023).
  [A Large Finer-grained Affective Computing EEG Dataset](https://doi.org/10.1038/s41597-023-02650-w).
  *Scientific Data*, 10, 740.
- **TUEV:** Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., and Picone, J. (2015).
  [Improved EEG Event Classification Using Differential Energy](https://doi.org/10.1109/SPMB.2015.7405421).
  *IEEE Signal Processing in Medicine and Biology Symposium (SPMB)*, pp. 1–4. The official
  patient-independent training/evaluation split
  ([Section III](https://isip.piconepress.com/conferences/ieee_spmb/2015/papers/l01_04.pdf)) is
  kept; training and validation are partitioned by subject within the official development set.
- **AD65:** [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504), CC0.
- **OpenBMI:** Lee, M.-H., et al. (2019).
  [EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy](https://doi.org/10.1093/gigascience/giz002).
  *GigaScience*, 8(5), giz002.
- **Longitudinal Eyes:** [OpenNeuro ds007176](https://openneuro.org/datasets/ds007176), CC0.

**Software and access terms**

- Baseline implementations use [Braindecode 1.8.1](https://pypi.org/project/braindecode/1.8.1/);
  the pretrained checkpoints come from the `braindecode/*` Hugging Face repositories (BSD-3-Clause).
- The two vendored [CBraMod data-preparation scripts](https://github.com/wjq-learning/CBraMod) retain their MIT license.
- FACED and TUEV access, attribution and usage terms follow their original data releases; access
  to the controlled release grants no additional redistribution rights.
