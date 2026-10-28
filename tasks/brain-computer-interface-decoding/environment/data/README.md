# Model-visible data

Source acquisition, original CBraMod preprocessing, the task-side TUEV split
adaptation, and public dataset download/materialization code are documented in
[MATERIALIZE.md](MATERIALIZE.md). This directory describes the released
development data interface, not source dataset distribution.

Five tracks: faced (9 classes, 32x10x200), tuev (6,16x5x200),
ad65 (3,19x10x200), openbmi (2,62x4x200), longitudinal (2,58x2x200).
Each has train/ and val/ containing manifest.json and NumPy shards.
Only these development directories are mounted here; sources and test data are absent.
pretrained/ holds the three audited initialization checkpoints
(bci-cbramod/bci-biot/bci-codebrain.safetensors), pinned in
/workspace/baseline/braindecode/pretrained.lock.json.

A manifest sample gives id, path, index, label, subject, session and provenance.
The array is in microvolts; the supplied baseline divides by 100 before casting
to torch float32. Do not divide twice. FACED/TUEV source precision is retained.
Training labels may be used to fit models. Validation is model selection only.
Joint training on all five supplied training splits is required. External data
and arbitrary external weights are prohibited; the task-provided audited
checkpoint allowlist is the only permitted pretrained initialization.

AD labels are inherited patient diagnoses: 0 Alzheimer's, 1 frontotemporal
dementia, 2 healthy control. They are not separately annotated window findings.
OpenBMI: left=0/right=1. Longitudinal: closed=0/open=1.
FACED labels/splits follow the fixed official scripts. TUEV signals are imported
unchanged; its uploaded train/validation union is regrouped with the current
official sorted-subject rule (68445/15487), keeping the original test set.

### TUEV split source

TUEV uses the official test split unchanged. Training and validation are
partitioned by subject within the official development set.
The [provider paper, Section III](https://isip.piconepress.com/conferences/ieee_spmb/2015/papers/l01_04.pdf)
describes separate training and evaluation sets and states that each patient
appears only once in its annotated subset. Our task-side train/validation
partition and materialization are implemented in [materialize.py](materialize.py).

The test metadata uses a single partition group for sample-level scoring;
its group count is not the number of patients. Cross-subject separation follows
the provider-defined split.
