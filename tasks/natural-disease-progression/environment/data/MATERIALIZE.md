# Materializing the data (controller side)

What the data are and how the labels are defined is in [README.md](README.md) (agent-facing) and
[../../docs/protocol.md](../../docs/protocol.md). This page gives the procedure and the numbers an
operator needs. All of it is MIMIC-IV-derived credentialed data. Every person who handles it must
hold current PhysioNet credentialing, the required CITI training, and the MIMIC-IV DUA
([../../docs/data-release.md](../../docs/data-release.md)).

## Two stages

| stage | input | script | output |
|---|---|---|---|
| 1. longitudinal-v2 release | raw MIMIC-IV 3.1 | the command chain in [../../README.md](../../README.md#2-data) (`audit.py` → `materialize.py` → `materialize_longitudinal_v2.py` → `stage_runtime_data.py`) | `runtime/development`, published as the controlled HF release `ScienceRSI/natural-disease-progression@f7da2d06295f1ed614de0ceb5619b3f72dbaedd7` |
| 2. evaluated split | that release | `build_heldout_split.py` | `development/` (agent) and `private/` (verifier) |

An operator with access to the HF release needs only stage 2. It takes no raw MIMIC and about 1 min
of CPU:

```bash
# fetch the pinned release (private repository; a credentialed token)
huggingface-cli download ScienceRSI/natural-disease-progression --repo-type dataset \
  --revision f7da2d06295f1ed614de0ceb5619b3f72dbaedd7 --local-dir /secure/ndp-release
# re-split inside the verifier grader image (pinned duckdb)
docker build -f tests/Dockerfile --target grader -t ndp-grader .
docker run --rm --network none -v /secure/ndp-release:/source:ro -v "$PWD":/task:ro \
  -v /secure/ndp:/out ndp-grader \
  python /task/environment/data/build_heldout_split.py --source /source --output /out/runtime
export TASK_DATA_DIR=/secure/ndp/runtime/development TASK_PRIVATE_DIR=/secure/ndp/runtime/private
```

## What comes out

| tree | mounted where | files | bytes | `runtime-manifest.json` SHA-256 |
|---|---|---:|---:|---|
| `development/` | agent, read-only: `train`, `public/validation`, `trusted/validation`, `trusted/risk_normalization.parquet`, `manifest.json` | 14 | 894,287,109 | `758bc59e3dcff454fb98b41864661a61bd0a7046cd75ebdefd507502ddb9b4bb` |
| `private/` | verifier only: `public/test` to inference, `trusted/test` and normalization to the grader | 7 | 182,727,774 | `ae7275054829f6d13e215da93c7a0533d7eabd12a51c3c7e81b614b5e6828c50` |

The build is deterministic. Two independent builds produce identical inventories, and a rebuild
should reproduce both digests above. `development/split.json` records the counts:

| split | patients | index admission |
|---|---:|---|
| train | 68,175 | 45,346 ICD-9-era + 19 mixed/no diagnoses + 22,810 ICD-10-era |
| validation | 3,915 | ICD-10-era |
| test (hidden) | 11,424 | ICD-10-era |

Assignment is by SHA-256 bucket of `sample_id` with salt `ndp-heldout-v3`: of the ICD-10-era
patients, 30 % go to test, 10 % to validation and the rest to train. The script refuses to run if
any validation or test sample is not ICD-10-era, or if labels or boundaries do not cover the
sample grid.

## Notes

- **Era.** The era is the coding system of the index admission's diagnoses (`mapping_status`
  `direct` = native ICD-10). Event dates are shifted per patient, but this field is not.
- **Boundaries.** Test and validation `sample_boundaries.parquet` reuse the release's rows for
  patients who were in its validation split. For the rest they are derived from the label rows,
  which carry the same cutoff, death and observation fields; on the release's own validation rows
  the derivation matches all date and death columns exactly. Only the diagnostic stratum flag
  `any_future_disease_record_365` differs: when derived, it counts the 262 scored diseases rather
  than every disease.
- **Inherited aggregates.** The 262-disease vocabulary and `risk_normalization.parquet` were fit
  by stage 1 on its train split, which included some patients now in test. Both are population
  aggregates. The normalization feeds only an auxiliary Brier metric, and neither touches the
  reward.
- **Keep `private/` private.** It holds test truth. Do not publish it, and do not upload it next to
  the development release.
