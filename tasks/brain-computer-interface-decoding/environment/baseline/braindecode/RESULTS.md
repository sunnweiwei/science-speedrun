# Braindecode baseline results

Updated 2026-10-21. All values below are **full private-test** primary metrics
in percent: seeds 0–4, mean ± sample standard deviation. There is no aggregate
test reward and no claim of statistical superiority. [RESULTS.json](RESULTS.json)
holds every seed's five scores, package hashes and the evaluated data identities.
The shared pretrained references follow the submission rules (one backbone for all
five tracks); the single-task specialists do not and serve as per-track references.

Single-task baseline results (five seeds):

| Model | Protocol | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | Single-task, scratch; FACED/TUEV standardized | 39.48 ± 0.78 | 56.90 ± 2.41 | 49.02 ± 4.64 | 77.30 ± 0.73 | 72.13 ± 1.27 |
| ContraWR | Single-task, scratch; FACED GroupNorm | 26.42 ± 0.36 | 43.44 ± 2.75 | 44.16 ± 7.71 | 66.48 ± 0.53 | 72.04 ± 1.18 |
| CBraMod | Single-task, scratch | 39.88 ± 0.76 | 57.89 ± 5.26 | 48.25 ± 3.99 | 71.35 ± 0.95 | 74.91 ± 1.41 |
| BIOT | Single-task, scratch | 18.48 ± 1.18 | 47.11 ± 1.27 | 39.49 ± 3.65 | 58.50 ± 1.94 | 76.03 ± 0.92 |
| CodeBrain | Single-task, scratch | 42.22 ± 17.05 | 61.36 ± 2.48 | 47.08 ± 3.77 | 71.00 ± 1.21 | 75.04 ± 0.91 |

Shared pretrained references (five seeds):

| Model | Protocol | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CBraMod | Shared, pretrained | 14.91 ± 0.60 | 40.71 ± 4.94 | 51.65 ± 4.13 | 57.41 ± 3.33 | 74.54 ± 0.56 |
| BIOT | Shared, pretrained | 16.75 ± 0.62 | 50.69 ± 0.92 | 59.71 ± 10.62 | 59.12 ± 2.90 | 74.22 ± 1.24 |
| CodeBrain | Shared, pretrained | 19.58 ± 1.26 | 55.77 ± 3.24 | 59.75 ± 9.45 | 63.31 ± 0.77 | 76.89 ± 1.07 |

FACED/TUEV: balanced accuracy. AD65: balanced accuracy after averaging window
probabilities per patient. OpenBMI: mean subject balanced accuracy.
Longitudinal: session balanced accuracy averaged within subjects, then subjects.

## Model-specific input settings

EEGNet uses per-sample, per-channel z-score normalization for FACED and TUEV, computed over the complete input window in both training and inference. TUEV additionally uses inverse-frequency replacement sampling. Other EEGNet tracks use /100 scaling without z-score normalization.

ContraWR uses one-group GroupNorm instead of BatchNorm for FACED, with /100 scaling and natural sampling. Other ContraWR tracks use the standard Braindecode normalization layers.

All specialist runs use seeds 0–4 and 50 training epochs; with replacement sampling, an epoch means as many draws as training examples. FACED/TUEV checkpoints are selected by validation kappa.

Exact specialist scores are under `single_task` in [RESULTS.json](RESULTS.json).

## Fixed controls on the randomized test layout

These controls use the same frozen v2 private data identities as the baseline evaluations. Random predictions use seed 3407; they are trivial references, not trained models.

| Predictor | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant class 0 | 11.11 | 16.67 | 33.33 | 50.00 | 50.00 |
| Random class, seed 3407 | 10.70 | 17.02 | 28.41 | 49.16 | 49.47 |


## Settings and comparability

- **All five single-task families:** independently trained specialist references. Random initialization; 50 complete epochs per task;
  batch 32; AdamW, learning rate 1e-4, weight decay .05, cosine schedule,
  label smoothing .1, gradient clipping 1. Select by validation kappa for
  FACED/TUEV and the task primary metric for the other three. Model-specific normalization and sampling are specified above.
- **Shared CBraMod, BIOT and CodeBrain references:** each family/seed has one shared backbone
  and five adapters/heads. Braindecode 1.8.1, audited pinned public pretrained
  core tensors, all parameters trainable; 50,000 updates (50 reporting periods
  × 1,000, not 50 complete passes per dataset), batch 8, AdamW 1e-4, decay .05,
  smoothing .1, cosine schedule, clipping 1. Equal task sampling; checkpoint
  selection by the equal mean of the five complete validation primary metrics.
  Input is /100-scaled EEG, adapted to shared channels and 2-second chunks;
  duration-weighted pooling retains all original window time. All 15 retained
  packages passed complete isolated testing and strict identity/coverage checks on the randomized v2 layout. The `shared_pretrained` index records each package and all five scores.

## Verification

The author trained and scored every package in this setup with the task's verifier
(full private test sets, inference isolated from labels). The maintainer rescored all 15 shared
references and 25 specialists through the verifier on the released data: every score matches
to within 0.0002 (one or two windows), and the five-seed means above are unchanged.
