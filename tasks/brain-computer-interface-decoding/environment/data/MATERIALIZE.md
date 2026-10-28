# Controller-side materialization

## Default: install the prepared releases

```bash
pip install -r environment/requirements-data.txt
HF_TOKEN=... bash environment/data/materialize.sh release /large/bci
export TASK_DATA_DIR=/large/bci/runtime/development
export TASK_PRIVATE_DIR=/large/bci/runtime/private
```

`release.py install` takes three pinned sources and checks every file's size and SHA-256:

| tracks | release | pin |
| --- | --- | --- |
| AD65, OpenBMI, Longitudinal | public [sunweiwei/ai4sci-brain-computer-interface-decoding-data](https://huggingface.co/datasets/sunweiwei/ai4sci-brain-computer-interface-decoding-data) (9.7 GB), built with the rebuild commands below | `download-source.json` `public` |
| FACED, TUEV | `runtime/` of the controlled [AI4Sci-BCI/AI4BCI_private_dataset](https://huggingface.co/datasets/AI4Sci-BCI/AI4BCI_private_dataset) (19.8 GB); `HF_TOKEN` must belong to an approved account | `download-source.json` `controlled` |
| `pretrained/` | the three `braindecode/*` checkpoints (BSD-3-Clause) | `../baseline/braindecode/pretrained.lock.json` |

The rest of this guide documents the rebuild from the original sources and the author's releases.
The public release was rebuilt by the maintainer from the pinned sources with the commands below;
its test sets match the author's in sample and subject counts, and the author's packages score the
same on it (see `docs/acceptance-2026-10.md`). Its opaque sample IDs differ from the author's
build, which used a different private ID key.


Never expose source data roots, whole FACED LMDBs or private labels to the agent.
Run these commands from the task directory in an operator environment with
environment/requirements-data.txt and PyTorch 2.8.0. Braindecode training and
verifier images use the separate environment/requirements.txt with MNE 1.11.0;
the frozen data was materialized with MNE 1.10.2. The recorded preprocessing runtime is
Python 3.12, torch 2.8.0+cu128; the Docker runtime is defined separately.

## Source requirements and code

The two controlled-access sources must be obtained by each operator under the
provider's terms. Authorized maintainers can obtain the
fixed processed FACED/TUEV artifacts from the private dataset release:
https://huggingface.co/datasets/AI4Sci-BCI/AI4BCI_private_dataset .
[FACED on Synapse](https://www.synapse.org/Synapse:syn50614194/datasets/)
requires an account with access to the dataset; check its current terms before
downloading. The [TUH EEG access page](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml)
describes its access request and the TUEV release. Credentials stay outside this
repository. The original CBraMod scripts are included for audit:
[FACED preprocessing](cbramod_reference/preprocessing_faced.py)
and [TUEV preprocessing](cbramod_reference/preprocessing_tuev.py).
They are copies from the fixed [CBraMod source](cbramod_reference/README.md),
not rerun steps in this materialization. The author supplied their outputs.

| Track | Source files needed by the code | Source acquisition | Task-side processing |
| --- | --- | --- | --- |
| `faced` | CBraMod LMDB `FACED/data.mdb` with `__keys__` and train/val/test members. To generate it from source, the original script expects per-subject pickles in `Processed_data` (28 trials each). | Obtain FACED access from Synapse, run the linked official script after configuring its paths, or provide a matching authorized LMDB. | [`faced()` in `materialize.py`](materialize.py) imports official samples/labels without filtering or re-epoching, preserves split membership and creates isolated development/test stores. |
| `tuev` | Final CBraMod `TUEV/processed/{processed_train,processed_eval,processed_test}/*.pkl`. To generate it from source, the original script expects the TUEV `edf/train` and `edf/eval` trees and their annotations. | Obtain TUH access and run the linked official script after configuring its paths, or provide matching authorized final pickles. | [`tuev()` in `materialize.py`](materialize.py) preserves waveforms/test set and applies the sorted-subject regrouping described below. |
| `ad65` | ds004504 `participants.tsv` plus `derivatives/sub-*/eeg/*_task-eyesclosed_eeg.set` and EEGLAB companions. Raw EEG is not used by this recipe. | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504), pinned by [`download.py`](download.py). | [`ad65()` and `epochs()`](materialize.py): diagnosis labels, 19 EEG channels, 200 Hz, 10 s windows. |
| `openbmi` | 108 `sess??_subj??_EEG_MI.mat` objects: 54 subjects × two sessions; ERP, SSVEP and artifact files are not required. | [GigaDB 100542](https://gigadb.org/dataset/100542), with object metadata in [`openbmi.lock.json`](openbmi.lock.json) and downloads in [`download.py`](download.py). | [`openbmi()`](materialize.py): both offline and online runs, 62 channels, 4 s trials. |
| `longitudinal` | ds007176 `sub-*/ses-*/eeg/*_eeg.vhdr` with associated BrainVision signal files and `*_channels.tsv`. | [OpenNeuro ds007176](https://openneuro.org/datasets/ds007176), pinned by [`download.py`](download.py). | [`longitudinal()` and `epochs()`](materialize.py): 58 EEG channels, CE/OE conditions, 2 s windows. |

The downloaded OpenNeuro source trees are fixed to Git revisions
`63b2ec152bc3cd4487d131124f10f767ebe543bc` (AD65) and
`5578877aa48b7094df79e11d6058c5b0c072d149` (Longitudinal). The
downloader verifies source hashes/sizes and records local SHA-256 receipts.
The OpenBMI lock fixes the 108 MI objects' upstream MD5, sizes and timestamps.
Downloaded and derived bytes are never checked into Git.

## Acquisition

Transfer an authorized FACED LMDB and the final TUEV/processed directory.
Verify SHA-256 before/after transfer. Supported final TUEV layouts have
train/validation/test counts 66050/17882/29421 (author archive) or
68445/15487/29421 (current official sorted-subject split). A combined development pool is also accepted as
83932/0/29421, with every development pickle in processed_train and an empty
or absent processed_eval directory. All three layouts must contain the same 83,932 development windows and 290 development subjects;
the importer reapplies the frozen subject split. Outer processed_train/eval are
intermediate files, not the final training interface.
The importer checks the sorted development-filename union SHA-256
`96f5f65c05527be1626d858917e758e375debd90e5061082038c8e8066e0e7bb`
and subject-prefix-list SHA-256
`baf3ed5bd371dffbd57b2ab15541d6d4a88c27a99d007b6e3d37a091dcaafd81`
before materializing any supported layout. These hashes bind the author's frozen
processed source inventory; a new authorized source revision requires an
explicit protocol update rather than silent acceptance.
The uploaded 66050/17882 split predates the current deterministic subject split.
The author approved regrouping its union by sorted subject prefix: first 232 of
290 subjects train (68445 windows), last 58 validate (15487); test stays 29421.
Signals and labels are unchanged, and original paths remain in operator records.
An authorized maintainer should run `audit_controlled.py faced SOURCE OUTPUT`
and `audit_controlled.py tuev SOURCE OUTPUT` before materialization, then keep
the full inventories (which can include class-bearing filenames) in controlled
storage. The public receipt may disclose only file counts and inventory digest.
The private HF release provides materialized FACED/TUEV datasets;
see [fixed revisions and download commands](#pinned-hugging-face-releases).
Maintainers can use the published runtime arrays, manifests and sample IDs
directly. Processing scripts are maintained publicly in this GitHub repository.

TUEV training and validation are partitions of the same official development
pool. The importer combines both folders before assigning subjects, regardless
of whether the input uses the historical, current or combined partition. The official test
split is unchanged. A fresh import preserves signal/label values and subject
assignment; byte-identical sample IDs additionally require the original private
build key and source-path mapping. Use the published runtime for exact replay.

For each public source:
```bash
python environment/data/download.py ad65 /large/sources/ad65
python environment/data/download.py openbmi /large/sources/openbmi
python environment/data/download.py longitudinal /large/sources/longitudinal
```
The downloader records fixed OpenNeuro Git trees, original Git blob hashes,
annex SHA-256/size, and download receipts. OpenBMI uses the canonical GigaDB MI objects, with each object's upstream
x-amz-meta-md5chksum, size and modification timestamp fixed in openbmi.lock.json.
Downloads verify that upstream MD5 and record SHA-256 receipts. A changed object
fails verification; dataset updates require a deliberate lock revision.

## Build

```bash
bash environment/data/materialize.sh faced --source /large/sources/FACED --destination /large/runtime
bash environment/data/materialize.sh tuev --source /large/sources/TUEV --destination /large/runtime
bash environment/data/materialize.sh ad65 --source /large/sources/ad65 --destination /large/runtime
bash environment/data/materialize.sh openbmi --source /large/sources/openbmi --destination /large/runtime
bash environment/data/materialize.sh longitudinal --source /large/sources/longitudinal --destination /large/runtime
export TASK_DATA_DIR=/large/runtime/development
export TASK_PRIVATE_DIR=/large/runtime/private
export TASK_RESEARCH_DIR=/large/research/run-001
```

Materializers refuse to overwrite completed manifests. Use a new destination
for a protocol revision. A successful build writes records/TRACK/receipt.json;
exceptions write ERROR.txt and do not count as complete. Intermediate shards
are not usable until manifest.json exists. Keep private inputs and truth together
under the operator-controlled private root, outside the agent image.

New tracks split subjects with sklearn train_test_split, seed 3407: test_size=.2,
then val_size=.25 of the remaining subjects. AD is diagnosis-stratified.
The authoritative membership is records/TRACK/subjects.json. All sessions remain
with their subject. Longitudinal snapshot has 45 subjects (the initial proposal
said 43); all eligible subjects are included, explicitly approved by the author.

AD derivatives already include official 0.5–45 Hz, ASR and ICA processing.
We resample to 200 Hz and epoch 10 s without repeating that pipeline.
OpenBMI and Longitudinal filter continuous data 0.5–45 Hz and resample to 200 Hz,
retain source reference, and do not run ICA. New arrays are float32 microvolts.
OpenBMI includes offline and online runs in both sessions, onsets converted from
MATLAB's 1-based index; [0,4) s gives exactly 800 samples. Longitudinal labels
are CE=0/OE=1, no fixed session-count assumption, 2 s windows.
Boundary/bad annotations reject windows; tails are counted and discarded.

## Opaque IDs
records/opaque-id-key is a controller-only HMAC key, mode 0600. Keep it with the
private build records for byte-identical ID reconstruction; never copy it into
the agent image, development release or submission. IDs cannot be brute-forced
from class-bearing source filenames. Numeric arrays and subject splits are
deterministic independently of this key. Test input ordering is by opaque ID.

## Exact inherited FACED/TUEV operations
FACED sorts source subject files, uses the first 80 for training, next 20 for
validation and remaining subjects for test. scipy.signal.resample maps each
30-second trial to 6000 samples; reshape (28,32,30,200), then take three
non-overlapping 10-second windows. The fixed official 28-trial label vector is
retained. The loader divides sample by 100 before float32 collation.
TUEV first resamples EDF to 200 Hz, then applies 0.3–75 Hz filtering and 60 Hz
notch, reads microvolts, forms the official 16 bipolar channels, and extracts
5-second events including 2 seconds of context on each side using the script's
repeated-signal boundary behavior. It sorts training subject prefixes and
splits 80/20; original evaluation becomes final test. The loader reshapes to
(16,5,200), divides by 100 and subtracts one from labels. We import the user's
already processed output and preserve these values; no extra filtering occurs.
The supplied final `processed/` layout had 66,050 training and 17,882 validation
files, whereas the current official sorted-subject rule over their union yields
68,445 and 15,487. The task's `tuev()` imports their union and reassigns it to
232 training and 58 validation subjects. The 29,421 supplied test files remain
the test set. TUEV uses the official test split unchanged. Training and validation
are partitioned by subject within the official development set.
Both original preprocessing scripts are retained verbatim in
`environment/data/cbramod_reference/` for audit. The task's data loader
implements the inherited `/100` and TUEV label shift. Historical source-to-loader
checks are retained in the [evidence archive](../../docs/rsi-experiment.md#verification-and-evidence).

For optional source-to-runtime numerical spot checks, run
`python environment/data/audit_official.py TRACK SOURCE RUNTIME_ROOT` from the
task root for FACED or TUEV. It compares sampled materialized signals and labels
against the inherited source-loader semantics. Source-inventory hashing remains
in `audit_controlled.py`; the two checks serve different purposes.

## Pinned Hugging Face releases

FACED/TUEV require original-provider authorization and repository-owner approval
for the private releases. Private access grants no additional redistribution
rights. Credentials, datasets and weights stay outside Git and agent images.

| Resource | Repository | Immutable revision | Manifest footprint |
| --- | --- | --- | --- |
| Processed FACED/TUEV data | [AI4Sci-BCI/AI4BCI_private_dataset](https://huggingface.co/datasets/AI4Sci-BCI/AI4BCI_private_dataset) | `1672c0343a719761fea07b2590cfe4214000b957` | 989 files; 39,245,822,996 bytes |
| Model packages and initialization | [AI4Sci-BCI/AI4BCI_model_weights](https://huggingface.co/AI4Sci-BCI/AI4BCI_model_weights) | `0d18646feb67739f813012db408f8b9fedc67bba` | 1,267 files; 6,670,694,867 bytes |

With authenticated HF CLI access, run from the task root and choose local
destinations outside the repository:

```bash
hf download AI4Sci-BCI/AI4BCI_private_dataset --repo-type dataset \
  --revision 1672c0343a719761fea07b2590cfe4214000b957 \
  --local-dir "$BCI_CONTROLLED_DOWNLOAD"
python environment/data/verify_hf_release.py "$BCI_CONTROLLED_DOWNLOAD"
hf download AI4Sci-BCI/AI4BCI_model_weights \
  --revision 0d18646feb67739f813012db408f8b9fedc67bba \
  --local-dir "$BCI_MODEL_DOWNLOAD"
python environment/data/verify_hf_release.py "$BCI_MODEL_DOWNLOAD"
```

The per-release `release-manifest.json` checks every listed file's path, byte
size and SHA-256. Counts exclude the manifest itself and HF-managed metadata.
The dataset contains supplied processed sources and materialized FACED/TUEV
runtime trees. Stage `runtime/development/{faced,tuev}` under `TASK_DATA_DIR`
and `runtime/private/{faced,tuev}` under the separate `TASK_PRIVATE_DIR`.
Preserve manifests and sample IDs. Prepare the other three tracks with the
pinned public-source commands above; they are not supplied by this release.
Private test storage uses globally randomized layout v2.

The model release includes 125 current specialists, 25 shared baselines, the
selected RSI package, five RSI replications and three audited initialization
checkpoints. It also contains frozen candidate source and reproduction evidence;
optimizer/resume states are excluded. [RESULTS.json](../baseline/braindecode/RESULTS.json)
and [rsi-results.json](../../docs/rsi-results.json) identify the 146 reported packages by
`hf_package_path` and package hash.
Per-file bindings and historical receipts are linked from the
[RSI record](../../docs/rsi-experiment.md#verification-and-evidence).
Supplemental 100-epoch BIOT runs are under `models/shared-duration/biot/` and
`evidence/duration/`; they do not replace the primary 50-epoch reference.

Only the three allowlisted `pretrained/` initialization files may be exposed, at
`$TASK_DATA_DIR/pretrained/`. Do not mount the whole model download or its task-trained
baseline/RSI packages into an agent. Do not mount the controlled data download,
source archives or private subtree into training workers. Mount development
tracks individually, as specified in the compose file. Verify complete runtime
shards and the initialization allowlist using the [verification guide](../../tests/README.md).

Downloads resume verified HTTP Range offsets; rerun `download.py` with the same
destination to reuse complete files. Keep materialization receipts, inventories
and opaque-ID keys in operator storage. A completed transfer marker does not
replace full runtime integrity verification.
