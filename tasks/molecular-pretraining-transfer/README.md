# Molecular Pretraining Transfer (v2: Uni-Mol2)

## 1. Overview

A model-development task about molecular foundation models. The agent pretrains an encoder from scratch on the
Uni-Mol2 corpus: 126 M molecules with 3D conformers, with every evaluation molecule removed. The verifier
fine-tunes the encoder with one fixed short recipe on 24 downstream regression tasks and scores the complete test
sets:

- QM9: 6 fine-tunes, 8 properties;
- COMPAS-1D: 4 properties;
- OpenADMET-ExpansionRx: 9 ADMET endpoints, time split;
- ASAP antiviral-ADMET-2025: 5 endpoints.

- **Reference.** The official Uni-Mol2 code and 84M architecture, with the paper's pretraining recipe. The model is
  capped at 90 M encoder parameters.
- **Metric.** Raw metrics, with no aggregate yet (a unified score map will be defined later). Per group:
  MA-RAE (mean relative absolute error, the ExpansionRx/ASAP leaderboard metric) and mean Spearman. Per task: MAE,
  RAE, R², Spearman and Kendall.
- **Anchors** (test MA-RAE for QM9 / COMPAS-1D / ExpansionRx / ASAP):

  | encoder | QM9 | COMPAS-1D | ExpansionRx | ASAP |
  |---|---:|---:|---:|---:|
  | no pretraining | 0.413 | 0.662 | 1.078 | 1.315 |
  | baseline: 84M, paper recipe, 30k updates (5.1 h on 8 H100) | 0.183 | 0.166 | 0.744 | 0.636 |
  | released Uni-Mol2 84M (paper-length pretraining), known-answer reference | 0.192 | 0.121 | 0.668 | 0.644 |

- **Status.** Verified: see section 6 and [`docs/VALIDATION.md`](docs/VALIDATION.md).

This version replaces v1 (PR #22: a PCQM4Mv2 gap-transfer task with 5 M-parameter models by pj20). The v1 study is
summarized in `environment/references/README.md`.

## 2. Data

| source | revision | license | use |
|---|---|---|---|
| Uni-Mol2 pretraining set, HF `dptech/unimol2-dataset`, shards 0–11 | 4748959 | Apache-2.0 | pretraining corpus |
| Uni-Mol fine-tune data (`molecular_property_prediction.tar.gz`: QM9 DFT geometries, `qm9.csv`) | sha256 a59d465e | Uni-Mol release | QM9 |
| COMPAS-1D, gitlab `porannegroup/compas` | 72e61e3 | CC BY 4.0 | COMPAS-1D |
| OpenADMET-ExpansionRx, HF `openadmet/openadmet-expansionrx-challenge-data` | 6b898cc | CC BY 4.0 | ExpansionRx |
| ASAP antiviral-ADMET-2025, Polaris `asap-discovery/antiviral-admet-2025-unblinded` | sha256 3388480e | CC0 | ASAP |

**Leakage filter.** Any molecule matching an evaluation molecule by InChIKey connectivity is removed from the
corpus: 995 molecules against 145,619 keys. QM9 and COMPAS-1D are split 8:1:1 by scaffold similarity, because
Uni-Mol2 does not release its splits. ExpansionRx and ASAP use their official splits. The fine-tune training sets
are seeded subsets of at most 5,000 molecules; the test sets are complete. Details:
[`environment/data/README.md`](environment/data/README.md).

```bash
docker build -f environment/Dockerfile --target runtime -t mpt-runtime molecular-pretraining-transfer
docker run --rm -v $PWD/molecular-pretraining-transfer:/task:ro -v /large-disk/mpt:/out mpt-runtime \
  bash /task/environment/data/materialize.sh /out
export TASK_DATA_DIR=/large-disk/mpt TASK_PRIVATE_DIR=/large-disk/mpt
```

About 190 GB is downloaded. The release is 175 GB: `public/pretrain` 173 GB, `public/downstream` 0.3 GB,
`verifier/test-inputs` 0.2 GB, `private/labels` 3 MB. The procedure, timings and layout are in
[`environment/data/MATERIALIZE.md`](environment/data/MATERIALIZE.md). Every released file is checked against
`environment/data/CHECKSUMS.json`.

## 3. Workspace

```text
/workspace/
  instruction.md, contract.md
  data/README.md, data/pretrain/, data/downstream/     read-only mounts of public/
  baseline/unimol2-ref/    pretrain.sh, publish.py, unimol2x/ (equal-length shard fix), README, RESULTS
  references/              Uni-Mol2, the released models under this protocol, the v1 study
  evaluation/              check_submission.py, dev_eval.py, protocol/ (the verifier's fine-tune code)
  submission/              empty at start
  research/                empty, writable, large disk
```

| baseline | test MA-RAE QM9 / COMPAS / ExpansionRx / ASAP | cost |
|---|---|---|
| `unimol2-ref` (official Uni-Mol2 84M, paper recipe, 30k updates) | 0.183 / 0.166 / 0.744 / 0.636 | 5.1 h on 8 H100 |

## 4. Evaluation

Verifier code: [`tests/`](tests/).

**Main container.** It has the fine-tune data and the test molecules without labels, and no network. It:

1. freezes the package;
2. runs the contract check as an unprivileged user (≤ 90 M encoder parameters; every encoder weight present);
3. runs the 24 fine-tunes as that user: the official Uni-Mol2 fine-tune recipe, 15 epochs, seed 0, best-validation
   epoch; three fine-tunes per GPU; each is retried once;
4. seals the per-molecule test predictions with hashes.

**Grader.** It has the labels, no GPU and no network. It scores the sealed predictions and never runs package code.

**`reward.json`.** Flat numbers:

- `valid`;
- `{qm9,compas1d,expansionrx,asap}_ma_rae` and `_mean_spearman`;
- `<task>_{mae,rae,r2,spearman,kendall}` for all 24 tasks.

An invalid package gets `valid: 0` and the reason. There is no aggregate `reward` yet (the TASK_STANDARD.md
multi-metric exception). `report.json` adds the encoder check and fine-tune timings.

**Cost.** One evaluation is 24 fine-tunes, 31 min on 8 H100 in the acceptance replay (three per GPU).

## 5. Running

```bash
export TASK_DATA_DIR=/large-disk/mpt TASK_PRIVATE_DIR=/large-disk/mpt                 # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/mpt-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p molecular-pretraining-transfer -a <agent> -m <model> --allow-agent-host <model API host>
```

**Resources** (external budget, in `task.toml`):

- development: 8 H100, 96 CPUs, 768 GB RAM, 12 hours; the baseline's pretraining uses 5.1 h of it;
- verifier: 8 H100, 4-hour limit.

Unit tests (standard library): `bash molecular-pretraining-transfer/tests/run_unit_tests.sh`.

## 6. Results and validation

Acceptance on one 8 × H100 host:

- The release was rebuilt with `materialize.sh`, and its digests are pinned.
- The baseline's 5.1 h pretraining was run once, with the same code and flags as `pretrain.sh`. `pretrain.sh` and
  `publish.py` were smoke-tested inside the development container.
- The published baseline package was replayed through the isolated verifier on the rebuilt release. It reproduced
  the development numbers exactly on all 24 tasks.
- An invalid package was rejected.

Numbers and the acceptance log:
[`docs/VALIDATION.md`](docs/VALIDATION.md) and
[`environment/baseline/unimol2-ref/RESULTS.md`](environment/baseline/unimol2-ref/RESULTS.md).

## 7. Limitations

- **Short fine-tune.** 5k training molecules and 15 epochs rank encoders cheaply, but the numbers are not paper
  numbers. With the official 40-epoch recipe on all 107k molecules, the released model reaches QM9 HOMO/LUMO/gap
  MAE 0.0055 Ha; here it gets 0.0087 Ha.
- **Resolution.** QM9 and ASAP barely separate the baseline from the released model. ExpansionRx and COMPAS-1D do.
  ASAP test sets are small (106–126 molecules per endpoint), so single-seed ASAP numbers are noisy.
- **One seed.** Differences below a few percent in a group's MA-RAE should not be over-read.
- **Rebuilt splits.** QM9 and COMPAS-1D splits are rebuilt (scaffold similarity), because Uni-Mol2 does not
  publish its own.
- **Public labels.** All evaluation labels are public upstream. The agent has no network, and test inputs and
  labels are never in its container, so this is a benchmark-visible rather than secret test.
- **Fixed input pipeline.** The fine-tune input is Uni-Mol2's feature pipeline. Architectures must consume it
  (contract.md).

## 8. Citation and licenses

Task authors: pj20 (v1), sunnweiwei (v2).

Sources:

- Uni-Mol2 (Ji et al., 2024) and its dataset (Apache-2.0);
- Uni-Mol (Zhou et al., ICLR 2023; code MIT);
- QM9 (Ramakrishnan et al., 2014);
- COMPAS-1D (Wahab and Gershoni-Poranne, CC BY 4.0);
- OpenADMET-ExpansionRx (Expansion Therapeutics / OpenADMET, CC BY 4.0);
- ASAP Discovery antiviral-ADMET-2025 (CC0).
