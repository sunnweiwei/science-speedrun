# Develop a PDE surrogate that generalizes across equations

## Task and goal

Develop, from scratch, one autoregressive surrogate model for two-dimensional, time-dependent
PDEs on regular grids. From the first four frames of a trajectory and its metadata, the same model
must predict the continuation of every held-out trajectory. The trajectories come from a corpus
of sixteen equation families and three transfer datasets (PDEArena Navier-Stokes, The Well's
active matter and turbulent radiative layer). Their dynamics, channel counts, grids and boundary
conditions differ. Any architecture is allowed.

Each held-out rollout is scored with The Well's variance-normalized RMSE (VRMSE), per channel and
step, clamped at 10. Rollouts are 26 steps for the corpus, 30 for The Well and 10 for PDEArena.
Datasets form four groups (the corpus as one group, each transfer dataset as its own). The reward is
the mean over groups of a log-band map of each group's mean VRMSE: 0.01 → 1.0, 0.1 → 0.667,
1.0 → 0.333, 10 or more → 0. Higher is better. For scale:

| predictor | reward |
| --- | ---: |
| persistence | 0.2827 |
| spatial mean of the last frame | 0.3105 |
| exact continuation | 1.0 |

The strongest released models (PDE-Transformer) reach corpus VRMSE 0.34 to 0.42 but transfer
poorly. Matching or beating them from scratch is the research problem.

## Workspace

- `/workspace/contract.md`: the submission package and the inference interface.
- `/workspace/data/`:
  - `README.md` and `AUDIT.md` describe the data;
  - `split.json` is the trajectory-level split into train and valid pools and held-out cases;
  - the four datasets are mounted read-only next to them.
- `/workspace/baseline/`: verified reference implementations with the results we reproduced:
  - `starter/` is small plumbing, reward 0.3127;
  - `cnext_unet/` is The Well's CNextU-net recipe from scratch, reward 0.3447.
- `/workspace/references/`: released-model replays and a literature audit, not verified here.
- `/workspace/evaluation/`: the verifier's scorer. Use it on the valid pools
  (`make_valid_cases.py`, `score.py`).
- `/workspace/research/`: writable space for derived data, checkpoints and notes.

## Rules

- Train on the `train` pools and select models on the `valid` pools of `split.json`. The
  held-out cases are replay-only. Do not train on them, tune on them, inspect their targets or
  hard-code anything about them.
- Build the model from scratch. No pretrained weights of any kind: not PDE-Transformer, DPOT or
  The Well's checkpoints, and no vision or language backbones. `model_manifest.json` must list
  `"pretrained_assets": []`. Re-implementing any architecture is allowed.
- Synthetic trajectories you generate here with your own solver code may be used for training.
  List them under `"sources_used"` as `"synthetic:<description>"`. Regenerating the held-out
  cases with the benchmark's own generator and seeds is not allowed.
- Inference uses only the supplied context frames and metadata: no other frames, data files or
  network. Parameters are fixed at inference, with no test-time training.
  - Allowed:
    - conditioning one model on the metadata;
    - an ensemble in which every member is applied to every case.
  - Not allowed: switching to a separately trained model by dataset name.
- There is no network.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission`,
starting early. Whenever you have a better model, replace the package with it in one atomic step:
assemble it elsewhere, then swap it in, as the baselines' `publish.py` and `package.py` do. The
package in `/workspace/submission` may be read at any moment. Keep `EXPERIMENTS.md` in the
package: each hypothesis, changed factor, seed, validation VRMSE per dataset, checkpoint hash
and outcome, including negative results.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Analyze where the current model fails, form a hypothesis, train and
validate a candidate, and publish it when it is better. Then start the next experiment. Do not
stop to hand in a final answer. A published model or a reached score is a milestone, not the
end of the task.
