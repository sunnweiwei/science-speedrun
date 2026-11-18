# Portable encoder contract

## Package

Keep the current complete package at `/workspace/submission`. It contains
`encoder.pt`, `config.json`, `infer.py`, `provenance.json` and
`LICENSE-OlmoEarth`. The encoder uses its native feature dimension, which is
1,024 for the Large two-phase starter. The same selected package serves both
AWF and LFMC. Training-only heads are omitted; inference needs no WorldCover
maps, moisture targets, training data or labels.

Candidate submissions must be trained from random initialization. Public
pretrained weights cannot initialize or supervise candidate training. Separately
labeled public pretrained models may be scored as diagnostic references only;
they are ineligible submissions. WorldCover maps and NDMI targets derived from
the declared geographically filtered pretraining data may supervise training.
AWF/LFMC labels may not train the encoder or auxiliary heads. Package provenance
records architecture, data, objective, seeds and selected checkpoint.

The starter has 307,770,368 encoder parameters and 7,000 cumulative updates:
6,000 WorldCover updates followed by 1,000 moisture-adaptation updates. Its
provenance retains both phases and their verified scratch ancestry. See the
[starter recipe](baseline/starter/README.md) for commands and
[recorded results](baseline/starter/RESULTS.md) for validation evidence.

## Inference interface

Run `PYTHONHASHSEED=0 python infer.py --model MODEL --input INPUTS.h5 --output features.npz`.
The installed pinned runtime supplies model dependencies. The package must work
after relocation, without network access or dependencies on its training
directory. The verifier sets the hash seed inside the isolated interpreter to
fix upstream modality ordering. Keep GPU type and batch size fixed for paired
comparisons; the starter's inference batch size is 16. CPU inference can be
explicitly requested using `--device cpu` for portability checks.

Inputs contain string `ids` and Sentinel-2 L2A uint16 imagery
`sentinel2_l2a` with shape `[N,H,W,12,12]`, optionally Sentinel-1 RTC float32
`sentinel1` with shape `[N,H,W,12,2]`. The last two axes are time and bands;
band order is the official OlmoEarth order. Optional `<modality>_valid` arrays
`[N,12,num_band_sets]` mark available band groups. Unavailable groups use the
official missing-token mask.

The `task` HDF5 attribute selects AWF bilinear upsampling followed by sampling
the labeled center pixel (16x16 input), or LFMC center-patch pooling (32x32
input). Both first average observed temporal and band-group tokens into spatial
features. Patch size is 4. Retain the project-release legacy month-index
positional encoding. Support arbitrary new IDs and inputs of the declared
shape; IDs must not drive model predictions.

Output NPZ contains string `ids` and finite float32 `features` with shape
`[N,D]`. Each requested ID must occur exactly once, with no missing or extra
IDs; order may vary. If every supplied band group is unavailable for a sample,
return a fixed zero feature vector. Retain that sample and ID; all such rows
count toward metrics.

## Development and final scoring

Development evaluation uses the spatially isolated downstream train/validation
views (`/workspace/data/eval/awf`: 1,053 training and 201 validation windows,
with all 205 spatial blind-test windows and any spatially overlapping raw
windows excluded; `/workspace/data/eval/lfmc`: 31,175 training and 8,262
validation rows). Frozen probes fit only downstream training labels: during
development `probes.py` reports both standardized logistic regression (`C` in
`[0.001, 0.01, 0.1, 1, 10, 100]`) and cosine `20-NN` (`temperature = 0.07` on
unstandardized features) on AWF validation, and standardized ridge regression
(`alpha` in `[0.01, 0.1, 1, 10, 100, 1000, 10000]`) on LFMC validation. Keep
these searches, declared splits, preprocessing, missingness handling and spatial
pooling unchanged. Use AWF validation accuracy (`20-NN` and logistic) and LFMC
validation **MAE** to guide development, together with the reported secondary
metrics and controls.

Final evaluation replays the submitted package with only the installed runtime,
model and unlabeled HDF5 input mounted into a network-isolated process. Labels
never enter inference. Probe fitting and scoring run separately. AWF is scored
on the sealed 205-window spatially disjoint blind test split (`pairs_across_splits == 0`)
using the frozen cosine `20-NN` classifier (`temperature = 0.07`) fitted on the
1,053 spatial training windows (with the 201 spatial validation windows retained
as a diagnostic). LFMC is scored on the held-out test split by **Mean Absolute
Error (MAE, NOT RMSE)** using the validation-selected frozen ridge probe.

The final score (`reward.json["score"]` and `reward.json["reward"]`), with
higher values better, is:

```text
awf_skill = (awf_test_accuracy - 59/205) / (1 - 59/205)
lfmc_skill = 1.0 - lfmc_test_mae / 24.435180349813233
signed_score = 0.5 * awf_skill + 0.5 * lfmc_skill
score = clip(signed_score, 0.0, 1.0)
```

`awf_test_accuracy` is a fraction in `[0, 1]`, `59/205` is the majority-class
fraction on the 205-window AWF spatial test split, and `24.435180349813233` is
the frozen LFMC test **MAE** of the constant predictor equal to the downstream
training-label mean. Both signed skills are averaged before top-level `[0, 1]`
clipping. Validation metrics are development diagnostics and are never
substituted into the final score.


## Delivery and failure behavior

Use the baseline publisher to stage, check and atomically replace a complete
submission. The controller snapshots the package before scoring. Final scoring
has the four-hour verifier deadline configured in `task.toml`.

Missing or malformed required package files, explicitly ineligible provenance,
and missing or malformed predictions produce `reward: 0` with
`status: invalid_submission`. Dataset, runtime, namespace and scoring failures
produce a nonzero exit and no new reward. An unexplained inference-process exit
is conservatively treated as a runtime failure; inspect its traceback before
attributing it to the candidate. Existing rewards are never overwritten.
