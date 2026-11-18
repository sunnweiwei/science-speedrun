# BCI four-track shared submission contract (V4)

Keep one complete package under `/workspace/submission`, including `infer.sh`,
architecture configuration, code and weights. One trainable EEG backbone must
serve all four active tracks (`tuev`, `ad65`, `openbmi`, `longitudinal`); input
adapters and classification heads may be task-specific. The backbone must be
trained jointly on all four training splits. Initialization may be random or
only from the task-provided audited checkpoint allowlist
(`/workspace/baseline/braindecode/pretrained.lock.json`: `cbramod`, `codebrain`).
`BIOT` (`braindecode/biot-pretrained-six-datasets-18chs`) and any unapproved
external checkpoints are strictly prohibited due to `TUEV` pretraining overlap
and are rejected during verification. Package contents must not contain symlinks
and must run after relocation.

Invocation: `bash infer.sh INPUT_DIR OUTPUT_DIR`.
`INPUT_DIR/manifest.json` contains `schema="bci-input-v1"`, `track`, `config`, and `samples`.
Each sample is `{"id": "opaque", "path": "shard-000000.npy", "index": 0}`.
NumPy shards have shape `[N, C, S, 200]`, in microvolts. Track `C/S`:
`tuev` `16/5`, `ad65` `19/10`, `openbmi` `62/4`, `longitudinal` `58/2`.
The baseline scales input by `1/100`. No labels, subjects or source filenames are
part of test inputs. Support unseen IDs, arbitrary order and arbitrary sample counts
(including single-subject or single-session batches).

Per-subject / per-session isolated inference: the verifier invokes `infer.sh`
independently for each held-out `(subject, recording_session)` unit using a freshly
materialized single-unit shard (`shard-000000.npy`) and isolated temporary directory,
preventing transductive cohort-wide feature whitening or cross-subject BatchNorm
statistics across the held-out test cohort.

Write `OUTPUT_DIR/predictions.jsonl`, exactly one row per ID:
`{"id": "opaque", "probabilities": [0.2, 0.8]}`.
Class counts: `6/3/2/2` for `tuev/ad65/openbmi/longitudinal` respectively.
Values must be finite in `[0, 1]`, sum to `1` within `1e-5`, and use the documented
class order. Missing, duplicate, unknown IDs or invalid probabilities invalidate
that track (`reward = 0.0`).

Runtime: PyTorch 2.8.0, dependencies in `requirements.txt`, one GPU, 64 GiB RAM,
two hours maximum per track, no network, unprivileged user (`UID 10001`), package/input
read-only and output writable. Inference cannot use labels, external caches or
workspace files. No training/adaptation on evaluation inputs.

Primary metrics and V3/V4 composite score:
- Every native track score `BA_k` (`tracks[k].reward` / `primary_balanced_accuracy`) is in `[0, 1]`
  using subject/session-stratified balanced accuracy (NOT window-pooled):
  - `tuev` (`chance = 1/6`): 6-class balanced accuracy on the official patient-disjoint evaluation set (with weighted/macro F1 and Cohen kappa reported as diagnostics);
  - `ad65` (`chance = 1/3`): arithmetic mean of window probabilities per patient, then patient-level balanced accuracy (window-pooled metrics reported only as diagnostics);
  - `openbmi` (`chance = 0.5`): mean of per-subject balanced accuracies across held-out subjects;
  - `longitudinal` (`chance = 0.5`): balanced accuracy per `(subject, session)`, averaged across sessions within each subject, then averaged across subjects.
- Chance-adjusted skill per track: `skill_k = (BA_k - chance_k) / (1 - chance_k)`.
- Composite score: `signed_score = mean_{k in (tuev, ad65, openbmi, longitudinal)}(skill_k)`,
  `score = reward = clip(signed_score, 0.0, 1.0)`.
- On full 4-track evaluation, `reward.json` emits `{"score": score, "reward": score, "signed_score": signed_score, "tuev": BA_tuev, "ad65": BA_ad65, "openbmi": BA_openbmi, "longitudinal": BA_longitudinal}` (matching `harness/ssi_index_v3.py:bci()`). Infrastructure failures produce no reward.
