# Evaluation protocol

The current baseline is OlmoEarth-v1 Large trained from scratch with WorldCover
supervision followed by WorldCover and moisture adaptation. Use the
[baseline recipe](../environment/baseline/starter/README.md),
[recorded results](../environment/baseline/starter/RESULTS.md), and
[evaluation commands](../environment/evaluation/README.md).
The [portable contract](../environment/contract.md) is the submission interface.

## Data populations and label access

AWF contains 1,459 released imagery windows: 1,115 training and 344 validation.
The project documentation describes 1,469 annotations; the track uses every
released imagery window and does not claim all original annotations. AWF has no
independent test split in this release.

LFMC contains 44,022 windows: 31,175 training, 8,262 validation and 4,585 test.
No exact rasterized point site occurs in multiple splits; this does not establish
a geographic buffer between neighboring split cells. All 218 windows with no
available Sentinel-1 or Sentinel-2 band group remain in the evaluation population.

Representation training uses only the declared geographically filtered
pretraining data and derived WorldCover/NDMI supervision. Downstream training
labels fit frozen probes; validation selects probes and candidate encoders.
LFMC test scoring follows candidate selection and runs in the separate verifier.
Public LFMC test was exposed during earlier development, so results must not be
presented as untouched-test generalization.

## Inputs and missing observations

Use the official released normalization statistics (two standard deviations,
no clipping), with Sentinel-1 intensity-to-dB conversion using a 1e-6 floor for
downstream inputs. Pretraining Sentinel-1 is already in dB. Fixed normalization
statistics are permitted and do not constitute pretrained model weights.
Retain the released observation masks and all samples, including missing dates.
Positional encoding follows the project default months 0–11, day 1, year 2024;
only month is used by v1.

The inference adapter preserves padding attention masks and pools over each
sample's observed modalities. Fully unobserved samples receive a fixed zero
feature vector. Batch independence, masked-pixel invariance and the all-missing
fallback are checked by `tests/test_inference.py`.

## Readout and frozen probes

AWF averages temporal/band tokens into a spatial grid, bilinearly upsamples by
four with `align_corners=False`, and reads the labeled center pixel. This follows
the spatial alignment of the [official project configuration](https://github.com/allenai/olmoearth_projects/blob/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a/olmoearth_run_data/awf/model.yaml).
LFMC uses center-patch pooling. One frozen encoder serves both tasks.

Fit standardization on training rows only. AWF uses logistic regression with
C in `[0.001, 0.01, 0.1, 1, 10, 100]`; LFMC uses ridge regression with alpha in
`[0.01, 0.1, 1, 10, 100, 1000, 10000]`. Select regularization on validation.
Primary metrics are AWF accuracy and LFMC MAE; report balanced accuracy,
RMSE and R² as secondary metrics. Compare the same encoder with matched random
initialization, raw seasonal spectral statistics and train-majority/train-mean
constants under the same splits and probe searches. The raw baseline fills
missing timesteps from the nearest present timestep, or uses zero normalized
features for a wholly absent modality.

These frozen probes do not reproduce the paper's trained downstream decoders.
Do not substitute the baseline's validation pair into a claimed final reward.
The contract specifies the final AWF-validation/LFMC-test reward formula; the
current results document reports uncertainty, selection limits and measured
costs separately.
