# Build a zero-shot time-series foundation model from scratch

## Task and goal

Pretrain one forecasting model from scratch that, without any per-dataset training, turns the
history of an arbitrary real-world time series into probabilistic forecasts. Make it beat the
strongest released zero-shot checkpoints on the GIFT-Eval benchmark: 97 (dataset, frequency,
horizon) configurations over 23 datasets and 7 domains, frequencies from 10 seconds to a year,
horizons from 6 to 900 steps.

For every configuration, two metrics are computed and each is taken relative to Seasonal Naive:

- GIFT-Eval's MASE of the median forecast;
- the weighted quantile loss (the leaderboard's CRPS column).

Each is combined as a geometric mean over the 97 configurations. The reward is
`clip(1 − sqrt(rel MASE × rel WQL), 0, 1)`, the geometric skill over Seasonal Naive. Higher is
better.

| system | reward |
| --- | ---: |
| Seasonal Naive | 0.0075 |
| the 6.4M starter (minutes of training) | 0.161 |
| the TimesFM-3 architecture at 50.9M, trained from scratch here (`baseline/timesfm3-50m`) | 0.275 |
| Chronos-2 (119M, pretrained) | 0.418 |
| **TimesFM-3** (331M, the best released zero-shot model) | **0.447** |

The released models were pretrained for days on multi-GPU clusters, on corpora of about 10^11
observations. Matching them from scratch is the research problem. A small model trained on this
corpus is well ahead of Seasonal Naive on short horizons and behind it on MASE at the 480–900-step
ones, which dominate the reward.

## Workspace

- `/workspace/contract.md`: the package, `bash infer.sh INPUT_DIR OUTPUT_DIR`, its input and
  output format, the per-cohort calls and the inference limits.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `gifteval_pretrain/<directory>` holds the 95 allow-listed directories of GiftEvalPretrain
    (273 GB, 63 G points, Arrow), built to share no series with GIFT-Eval. Climate reanalysis
    (ERA5, CMIP6) and simulated building loads are most of the volume; `README.md` gives the
    composition, and how to mix the sources is up to you;
  - `valid/{input,truth}` holds one validation window per GIFT-Eval series (151,170 instances in
    97 configurations), staged in the contract's format with labels. Each window ends before
    any held-out region.
- `/workspace/baseline/`: verified reference implementations with the results we reproduced:
  - `timesfm3-50m/`: the main baseline. The TimesFM-3 architecture (its released Apache-2.0
    code, random init) at 50.9M parameters, data-parallel training with source-balanced sampling,
    and the package that runs the released TimesFM-3 forecaster on the trained weights; reward
    0.275;
  - `starter-41m/`: the starter's patched transformer at 41.8M parameters; reward 0.207;
  - `starter/`: the 6.4M starter, the quickest end-to-end pipeline; reward 0.161;
  - `seasonal_naive/`: the normalizer; reward 0.0075.
- `/workspace/references/`: the leaderboard and the replays of released models, not verified here.
- `/workspace/evaluation/`: the verifier's scorer (`score.py` reproduces gluonts to 1.9e-7).
- `/workspace/research/`: writable space for the packed corpus, checkpoints and notes.

## Rules

- **Build the model from scratch.** No pretrained weights of any kind: not TimesFM, Chronos,
  Moirai, TiRex, Toto or TTM, and no language or vision backbones. `model_manifest.json` must list
  `"pretrained_assets": []`. Re-implementing architectures, or using released model code without
  its weights (as `baseline/timesfm3-50m` does), is allowed.
- **Training data** is the allow-listed corpus directories, plus synthetic series generated here by
  your own code. Declare every corpus directory and generator in `sources_used`. No other data:
  not GIFT-Eval series, LOTSA, chronos_datasets, Time-300B, Monash or anything downloaded.
- **Validation** (`/workspace/data/valid`) is for model selection and diagnostics only, never for
  training or fine-tuning. The submission stays zero-shot in GIFT-Eval's sense.
- **Inference** uses only `INPUT_DIR`. Parameters are fixed: no test-time training or adaptation
  on the contexts.
  - An ensemble of models you trained is one package, if every member forecasts every instance.
  - Choices fixed before inference and tuned on validation are part of the package, including a
    fixed rule applied to each instance's own context (for example per frequency or term).
- Validation numbers are not on the reward's scale. The validation Seasonal Naive reference is the
  task's own, and the starter's default subset has 15 configurations. Validate across terms.
- There is no network.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission`,
starting early: `infer.sh`, `model_manifest.json`, `README.md`, model code, configuration,
weights, normalization statistics and every runtime asset. It must run relocated, without the
workspace or the trainer, deterministically, with the runtime pinned in
`/opt/runtime/requirements.txt`, and start cheaply, because the verifier calls it once per cohort.

Whenever you have a better model, replace the package in one atomic step: assemble it elsewhere,
then swap it in, as `baseline/starter/publish.py` does. The package may be read at any moment.

Score a package on the validation windows exactly as the verifier will:

```bash
bash /workspace/submission/infer.sh /workspace/data/valid/input /workspace/research/valid-pred
python3 /workspace/evaluation/score.py --truth /workspace/data/valid/truth --output /workspace/research/valid-pred \
    --baseline /workspace/evaluation/seasonal_naive_reference_valid.json --report /workspace/research/valid-report.json
```

Keep `EXPERIMENTS.md` in the package: each hypothesis, changed factor, seed, measured compute,
validation skill per configuration group, checkpoint hash and outcome, including negative results.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Analyze where the current model fails (by term, frequency and domain), form
a hypothesis, train and validate a candidate, and publish it when it is better. Then start the
next experiment. Do not stop to hand in a final answer. A published model or a reached score is
a milestone, not the end of the task.
