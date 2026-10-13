# MassSpecGym v1.5 data

In this workspace, `/workspace/data` holds this file, `check_resources.py`, and the read-only
development view prepared before the session:

```text
/workspace/data/
  manifest.json
  train.jsonl
  validation.jsonl
  validation_candidate_pools.jsonl
  glacier/                  preparation.json, manifest.json, targets.sqlite, train.jsonl, validation.jsonl
  synthetic_augmentation/   manifest.json, labels.tsv, subformulae.sqlite, sanitation_report.json, training_data_audit.json
```

The test queries, their candidate pools and all test labels exist only in the verifier. The
sections below describe provenance, the official split and each file. The paths under
`TASK_DATA_DIR` refer to the controller's host layout, of which `development/` is what you see.

## Pinned upstream release

- Hugging Face dataset: `roman-bushuiev/MassSpecGym`
- Repository type: `dataset`
- Full revision: `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`
- License: MIT
- Upstream code commit used for semantics:
  `f259fe3780d5bd227fc6ece36ce6f397c2eef716`

The core experimental task uses these source files:

| Path below the HF repository | Bytes | SHA-256 |
| --- | ---: | --- |
| `data/MassSpecGym1.5.tsv` | 261,197,365 | `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a` |
| `data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json` | 454,362,310 | `9536a330cba24c399271364889119f90124579400dba14e9f78630c75c13c73a` |

The source release is public. Do not commit either file or a prepared view.

The development environment additionally receives the public MIST vFRIGID
synthetic augmentation corpus, downloaded from Zenodo record `20671842` under
the MIT license. The pinned `msg_aug_filt.tar.gz` is 6,533,892,821 bytes with
MD5 `32d8813d84d06364fc318e81f9cb4416`. Despite its upstream name, the archive
contains 15 spectra whose 2D structures overlap validation/test and 10 spectra
with one RDKit-invalid SMILES. The materializer removes those exact 25 rows and
publishes only the 2,333,485-row clean result.

The prepared HF release includes the sanitized augmentation. Its provenance
retains the pinned Zenodo record, archive digest and prepared SHA-256 manifests;
the runtime download is now bound to the task-owned HF commit as well.

## Official split

| Fold | Spectra | Unique 2D structures |
| --- | ---: | ---: |
| train | 194,119 | 22,746 |
| validation (`val` upstream) | 19,429 | 3,185 |
| test | 17,556 | 2,998 |

The 231,104 spectra contain on average approximately 38, 29, and 32 peaks in
train, validation, and test. The release contains `[M+H]+` and `[M+Na]+`
adducts and Orbitrap, QTOF, or missing instrument annotations. The official
split has no 2D InChIKey overlap across folds and was constructed with a
cross-fold MCES distance threshold of 10.

Use the complete train fold for scientific model development. The formal
GLACIER replay's published condition domain contains 99,341 train, 9,734
validation, and 9,954 test spectra: `[M+H]+`, finite collision energy, and an
Orbitrap or QTOF annotation. This is a model-domain restriction, not a different
benchmark denominator; the formal hybrid still ranks all 17,556 test spectra.
A contributor may use a deterministic subset for a plumbing smoke run, but
subset scores are not benchmark results. Validation labels are for model and
hyperparameter selection. The official test fold is replay-only.

## Prepared layout

```text
TASK_DATA_DIR/
  development/
    manifest.json
    train.jsonl
    validation.jsonl
    validation_candidate_pools.jsonl
    glacier/
      preparation.json
      manifest.json
      targets.sqlite
      train.jsonl
      validation.jsonl
    synthetic_augmentation/
      manifest.json
      labels.tsv
      subformulae.sqlite
      sanitation_report.json
      training_data_audit.json
  verifier/
    manifest.json
    queries.jsonl
    candidate_pools.jsonl

TASK_PRIVATE_DIR/
  manifest.json
  labels.jsonl
```

Train and validation rows contain the target SMILES, 2D InChIKey, and molecular
formula. Validation rows also name an anonymous candidate pool so agents can run
the same hit-rate logic during development.

`glacier` contains native MAGMa fragment masks and spectra derived only from
the public train/validation folds. The materializer requires the runtime built
by `environment/runtime/install.sh` (default `/opt/glacier/bin/python` and
`/opt/ms-pred`). For a host installation elsewhere, set
`MASSSPECGYM_GLACIER_PYTHON` and `MASSSPECGYM_GLACIER_SOURCE`; preparation uses
`MASSSPECGYM_PREPARE_WORKERS` CPU workers (default 16). It is resumable and
records file/source hashes, domain counts and exclusions. Selected spectra
without an assignable fragment are excluded from fitting with a recorded
reason; this never removes queries from the benchmark denominator. No test
file is read by this preparation. It runs before the agent budget begins.
The completed preparation retains **99,119 train** and **9,720 validation**
spectra from the eligible 99,341/9,734; 222/14 rows have no assignable fragment.
Those exclusions apply only to fragment-supervised fitting/validation.
Retrieval still evaluates all 19,429 validation and 17,556 test queries.

`synthetic_augmentation` contains the same public ICEBERG simulations used by
the strict-clean MIST fallback component of the formal reference after
task-owned sanitation. `labels.tsv`
provides spectrum IDs, formulae, SMILES, InChIKeys, ionization, and instrument;
`subformulae.sqlite` maps each spectrum ID to the losslessly preserved simulated
spectrum/subformula payload. It is legitimate training data for any method, not
a requirement to use MIST. Validation and test spectra or structures are not
included. The manifest binds every file by SHA-256.

The pinned archive was independently materialized on 2026-09-02 in 19 minutes
45 seconds with approximately 1.9 GiB peak process RSS. The complete published
view is:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `labels.tsv` | 398,064,763 | `8d6506976f9105bd49fc12c14e7fab2ddfb6d1130d871a10b651ece59c833705` |
| `subformulae.sqlite` | 36,743,610,368 | `d0ede6aa7162e4d12dcc2b7166e6d4b7f545d00ee047e4b816edd20234e62d77` |
| `sanitation_report.json` | 719 | `e23eef26f61ef7b4b3638f92945470e8983d31088e7db8bb15dfe6674833b7f0` |
| `training_data_audit.json` | 538 | `243faf58b90ab2b7fc1429ec7f3246159e947071a114d8ed3f0e0e62a7563d2c` |

The labels contain exactly 2,333,485 data rows. The database reports the same
row count and `PRAGMA integrity_check = ok`; its digest was recomputed after
the atomic publication step.

Verifier query rows omit SMILES, InChIKey, formula, parent mass, and fold. The
public candidate pools contain candidate IDs and SMILES because candidate
structures are inputs to retrieval. Private labels contain only the query ID,
correct candidate ID, and true SMILES needed for MCES@1.

The upstream benchmark and answers are publicly downloadable. This task's
offline split is an evaluation hygiene boundary, not a claim that MassSpecGym is
a secret benchmark. Report it as benchmark-visible evaluation.
