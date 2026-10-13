# Model-visible data interface

This environment exposes only the active longitudinal task. Each sample is one patient at the second eligible live adult discharge. The cutoff is selected using information available by that discharge, without conditioning on whether the patient later returns. Patients are isolated across splits and appear once.

**Splits.** Validation (3,915) and the hidden test (11,424) hold only patients whose index (second) admission was coded in ICD-10, i.e. discharged after the October 2015 ICD-9 → ICD-10 switch. Training (68,175) holds the remaining ICD-10-era patients and every ICD-9-era patient. One-year outcomes after an ICD-9-era index admission are under-recorded: ICD-9 records whose category mapping is ambiguous are not counted, so for those patients many true first records are missing and appear as negatives (for example sepsis 0.15 % vs 3.3 % one-year incidence, heart failure 0.13 % vs 2.5 %). Unambiguous categories such as acute kidney injury are unaffected. An event's `mapping_status` (`direct` = native ICD-10) shows the coding system of each diagnosis.

The target is the first later inpatient record of a disease not recorded in that patient's visible history. It is not biological disease onset. Missing later records may reflect censoring or health-system capture; use the supplied validity masks rather than converting unknown outcomes to negatives.

## Visible inputs

`data/train/`, `data/public/validation/`, and the verifier's public test input share:

- `manifest.json`: split, ordered train-derived disease vocabulary, the 13 horizons `30,60,...,360,365`, and protocol/schema identifiers;
- `samples.parquet`: opaque `sample_id`, sex, approximate age, high-age flag, and cutoff; and
- `events.parquet`: cutoff-bounded diagnosis, lab, prescription, and administration events with code, raw value/unit where applicable, event time, availability time, opaque encounter/group keys, source/mapping metadata, and seconds before cutoff.

Every visible event satisfies both `event_time <= cutoff` and `available_time <= cutoff`. `sample_id` is an output-alignment key, not a feature. Opaque keys may group simultaneous or same-stay events but must not be treated as stable patient identity.

Training additionally contains `labels.parquet` and `label_windows.parquet`: direct disease/death labels, validity masks and observation boundaries for all horizons. They must never be joined to another split.

## Validation and hidden test

`data/trusted/validation/` contains validation labels for aggregate local scoring. It is not training data. The development container has no public test input, test labels, raw MIMIC tables, stable patient split identifiers, controller database, or `working.duckdb`.

The final verifier exposes only public test samples/events to `infer.sh`. A separate sealed grader receives test labels and the frozen train-only normalization configuration. The inference process cannot access the grader mount.

## Output and scoring

For every public sample, output 13 nondecreasing probabilities per scored disease and 13 for death. The reward, `panel_recall_at_5pct`, is case detection at a 5 % alert budget for a prespecified panel of serious new-onset endpoints at 365 days (`/workspace/contract.md`, `/workspace/evaluation/panel_scoring.py`). The earlier disease-first macro time-averaged AUROC over all diseases is still reported. Unknown labels are excluded and single-class cells are unavailable.

MIMIC-derived artifacts remain credentialed data. They may be accessed only by individually authorized users satisfying current PhysioNet training and DUA requirements, and must not be uploaded to a public release.
