# Natural Disease Progression: anticipating serious new disease after discharge

## 1. Overview

The agent designs, trains and improves one jointly trained neural network. From a patient's
cutoff-bounded multimodal MIMIC-IV history at their second eligible live adult discharge, the
network predicts cumulative risk of first-recorded inpatient diagnosis for 262 diseases and of
death, at 13 horizons (`30, 60, ..., 360, 365` days). The outcome is a first recorded inpatient
diagnosis, not biological onset. The final model must be neural: trees, rate tables and ensembles
with them are not eligible (see the instruction).

**Reward: `panel_recall_at_5pct`.** This is case detection for serious new disease.
- The panel is 26 prespecified serious, actionable endpoints, among them heart failure, AF, MI,
  stroke, PE, CKD, AKI, sepsis, cirrhosis, dementia, COPD and several cancers.
- Each endpoint is restricted to strictly new onset.
- Patients are ranked by predicted 365-day risk. The score is the share of next-year cases found
  in the top 5 %, averaged over the endpoints with at least 20 cases. Chance is 0.05 and perfect
  is 1.0.

The earlier reward (disease-first macro time-averaged AUROC over all 262 diseases), AUPRC,
calibration and mortality are reported next to it.

**Status.** Verified on 2026-10-01 at evaluation v3. This integration redesigned the splits and the
reward; why is in [docs/EVALUATION_DESIGN.md](docs/EVALUATION_DESIGN.md). In short, v2 mixed
ICD-9-era patients, whose outcomes are under-recorded, into every split, so that a 0/1 coding-era
indicator alone scored 0.653 on the v2 reward, above the age/sex baseline's 0.630.

## 2. Data

MIMIC-IV 3.1 (PhysioNet, credentialed; DOI 10.13026/kpb9-mt58). Patients are split by
`subject_id`, with one sample at the second qualifying live adult discharge. Inputs are
demographics plus diagnoses, labs, prescriptions and administrations whose event and availability
times are at or before the cutoff. ICD-10 is reduced to three-character categories, and ICD-9 uses
the CMS GEM with ambiguous candidates masked. Labels are first later inpatient records with
death-aware censoring masks. The full definition is in [docs/protocol.md](docs/protocol.md) and
[docs/schema.md](docs/schema.md).

There are two stages; the operator procedure, sizes and digests are in
[environment/data/MATERIALIZE.md](environment/data/MATERIALIZE.md).

1. **longitudinal-v2 release** (task author, from raw MIMIC). The chain runs inside the dev image:

   ```bash
   python environment/data/audit.py --mimic-root /mimic --output build-intermediate \
     --gem environment/data/cms-2018-gem.zip
   python environment/data/materialize.py --mimic-root /mimic --output build-intermediate
   python environment/data/validate.py --materialized build-intermediate
   python environment/data/audit_risk.py --materialized build-intermediate
   python environment/data/materialize_longitudinal_v2.py --source-materialized build-intermediate \
     --output materialized
   python environment/data/validate_longitudinal_v2.py --materialized materialized
   python environment/data/stage_runtime_data.py --materialized materialized --destination runtime-data
   ```

   Its development tree is the controlled HF release
   `ScienceRSI/natural-disease-progression@f7da2d06295f1ed614de0ceb5619b3f72dbaedd7` (13 files,
   1.08 GB). The release excludes test data, controller identifiers and raw tables.

2. **Evaluated split.** `environment/data/build_heldout_split.py` re-splits that release
   deterministically into `development/` (the agent's mount) and `private/` (the verifier's):

   | split | patients | index admission |
   |---|---:|---|
   | train | 68,175 | every ICD-9-era (45,346) and mixed (19) patient, plus 22,810 ICD-10-era |
   | validation | 3,915 | ICD-10-era |
   | test (hidden) | 11,424 | ICD-10-era |

Access: every person handling these files must hold PhysioNet credentialing, CITI training and the
MIMIC-IV DUA ([docs/data-release.md](docs/data-release.md)). The superseded HF revision `6b9c2cd`
must not be used.

## 3. Workspace

The agent receives `/workspace/{contract.md, data, baseline, references, evaluation, submission,
research}`. The data mount is read-only: `train`, `public/validation`,
`trusted/validation` (validation labels, for aggregate scoring only), the train-only normalization
artifact and the manifest. Test inputs, test labels, raw MIMIC and controller files are never
mounted.

| directory | contents | reproduced |
|---|---|---|
| `baseline/temporal-gru/` | multimodal GRU with learned cumulative hazards; `package.sh`, `publish.py`, [RESULTS.md](environment/baseline/temporal-gru/RESULTS.md) | test reward **0.236**, validation 0.211 |
| `references/delphi/` | upstream Delphi (gerstung-lab, MIT) code with synthetic demo data | not adapted, not verified |
| `references/scaffold/` | architecture-neutral data, training and submission seams | not a model |
| `evaluation/` | `score.py`, `risk_scoring.py`, `panel_scoring.py`: the verifier's code | — |

Statistical, tree and legacy baselines stay in `docs/maintenance/` and are not copied into the image.

## 4. Evaluation

The verifier (`tests/`) is two-phase:
1. `main` runs the submitted `infer.sh` as uid 10001 against `public/test`, with no network and no
   labels mounted. It seals the predictions with SHA-256.
2. A separate `grader` service, which never executes submission code, checks the seal, scores with
   `environment/evaluation/risk_scoring.py` and `panel_scoring.py`, and writes `report.json` and
   `{"reward": panel_recall_at_5pct}`.

Invalid packages or predictions score 0. Infrastructure failures write no reward.

Anchors on the hidden test (single seed):

| system | reward |
|---|---:|
| constant prediction (chance) | 0.050 |
| statistical strata | 0.185 |
| temporal-gru baseline | 0.236 |
| LightGBM on 567 features (maintainer reference; not an eligible submission) | 0.268 |

## 5. Running

```bash
export TASK_DATA_DIR=/secure/ndp/runtime/development TASK_PRIVATE_DIR=/secure/ndp/runtime/private
export TASK_RESEARCH_DIR=/large-disk/runs/ndp-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p natural-disease-progression -a <agent> -m <model> --allow-agent-host <model API host>
```

Resources come from `task.toml`. The agent gets 1 GPU, 10 CPUs, 22 GiB and 128 GiB of disk for
24 h. The verifier gets the same resources with a 12 h limit; a full verification takes about
2 min. Both images set `OMP_NUM_THREADS=10`. Without it, PyTorch sizes thread pools by host cores,
and under a CPU quota training ran about 15× slower. Unit tests:

```bash
docker build -f tests/Dockerfile --target unit-tests -t ndp-unit-tests . && docker run --rm ndp-unit-tests
```

## 6. Results and validation

| run | result |
|---|---|
| author, v2 splits and reward, 1× RTX 3070 | reference GRU 0.8311 macro AUROC (validation); 3-seed ensemble 0.837; LightGBM 0.824 ([docs/longitudinal-v2-report.md](docs/longitudinal-v2-report.md), historical) |
| integrators, v2 (2026-10-01) | GRU retrained with the documented recipe: **0.8311172** (documented 0.83112). Two-phase verifier reward identical to the development score. Red-team packages score 0 |
| integrators, v3 (2026-10-01) | split built twice with identical inventories. Baselines on test: statistical 0.185, GRU **0.236**, LightGBM 0.268. GRU through the full verifier on the real private test: 0.2362567160, identical to `score.py`. Constant package 0.0501; empty and NaN packages 0. Unit tests 42/42 |

Full integration record: [docs/VALIDATION.md](docs/VALIDATION.md). An agent campaign on the v2
setup is described in PR #68.

## 7. Limitations

- **Single centre and retrospective.** MIMIC-IV is one Boston hospital system. The endpoint is a
  recorded inpatient diagnosis, which depends on care seeking and coding.
- **The split is not prospective.** Validation and test are ICD-10-era patients drawn at random. A
  genuine temporal split needs `anchor_year_group`, which the release does not carry.
- **Incomplete training labels.** The ICD-9-era training labels are incomplete by construction.
  This is disclosed to the agent.
- **Endpoint panel.** It was prespecified by the integrators from clinical relevance, not reviewed
  by a clinician. The 20-case minimum leaves 17 of 26 endpoints in the test reward; the rarer
  cancers are reported but not rewarded.
- **Small validation set.** 3,915 patients and 14 endpoints in reward, with a single-model interval
  of about ±0.03.
- **Calibration is not rewarded.** The baseline's calibration slope is about 3.
- **Inherited aggregates.** The 262-disease vocabulary and the normalization artifact were fit on
  the v2 train split, which contained some current test patients. Both are population aggregates,
  and neither enters the reward.

## 8. Citation and licenses

- MIMIC-IV 3.1: Johnson et al., PhysioNet, doi:10.13026/kpb9-mt58. PhysioNet Credentialed Health
  Data License 1.5.0 and DUA. The task repository contains no MIMIC rows.
- CMS 2018 General Equivalence Mappings (`environment/data/cms-2018-gem.zip`), US government work.
- Delphi: Shmatko et al., *Learning the natural history of human disease with generative
  transformers*, Nature (2025). Code MIT ([references/delphi/LICENSE.upstream](environment/references/delphi/LICENSE.upstream));
  Delphi-2M weights are not included.
- Task code: see the repository license.
