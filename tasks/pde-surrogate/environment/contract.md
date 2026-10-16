# Model delivery and inference contract (`pde-surrogate-rollout/v1`)

This document defines the portable package the agent delivers and the exact
interface the independent verifier uses to run it. It is deliberately
independent of any architecture, training framework or data source.

## Deliverable

`/workspace/submission/` containing:

- `infer.sh`: the entry point, `bash infer.sh INPUT_DIR OUTPUT_DIR`.
- `model_manifest.json`: `{"contract": "pde-surrogate-rollout/v1", "name": ...,
  "description": ..., "parameters": <int>, "context_frames": <int>,
  "sources_used": [...], "pretrained_assets": [...]}`. `context_frames` is the
  number of initial frames the model consumes (1 to 4). The verifier always
  supplies four; a package that uses fewer must ignore the earlier ones.
- model code, configuration, weights, preprocessing statistics and every other
  runtime asset the package needs. No network, no development workspace, no
  undeclared cache. Code and weights must describe the same model.
- `README.md` describing what the model is and how it was selected.

The directory may be read at any moment, so replace it as a whole: assemble the new package
elsewhere and swap it in with one rename (the baselines' `publish.py` and `package.py` do this).

One package serves every dataset in the evaluation. Per-dataset specialist
checkpoints selected at inference time by dataset name are not permitted;
conditioning on the metadata supplied with each case is. Parameters are fixed
at inference: no test-time training or adaptation on the supplied frames. An
ensemble of models you trained counts as one package when every
member is applied to every case.

## Input directory

```text
INPUT_DIR/
  index.json
  cases/<case_id>.h5
```

`index.json`: `{"contract": "pde-surrogate-rollout/v1", "cases": [{"case_id":
..., "dataset": ..., "n_steps_out": K}, ...]}`. Case IDs are opaque strings; do
not infer anything from them.

Each `cases/<case_id>.h5` holds:

| object | shape / type | meaning |
| --- | --- | --- |
| `input` | `(4, C, X, Y)` float32 | the four initial frames, physical units |
| attribute `dataset` | string | source dataset name (e.g. `ape2d/burgers`, `well/active_matter`) |
| attribute `pde` | string | PDE name as stored upstream |
| attribute `fields` | list of strings, length `C` | channel names in order |
| attribute `field_scheme` | string | one letter per channel; consecutive identical letters form a vector field |
| attribute `dt` | float | physical time step between stored frames |
| attribute `domain_extent` | list of two floats | physical size along `X` and `Y` |
| attribute `boundary_conditions` | list of four strings | `x negative, x positive, y negative, y positive`, each `periodic`, `wall` or `open` |
| attribute `constants` | JSON string | name → value map of the trajectory's physical parameters; may be empty |
| attribute `n_steps_out` | int | number of frames to predict, `K` |

Every held-out case starts at the beginning of its trajectory: `input` holds
frames 0 to 3 and the prediction frames 4 to 3 + K. `X` and `Y` vary by dataset
(128 to 384 in the current split; any size may appear). `C` varies by dataset (1 to 11). A package must accept any case with
this schema, including datasets, shapes and constants it never saw; it may not
refuse a case.

A tiny synthetic example of both directories and a valid output is produced by
`python3 /workspace/evaluation/make_example.py OUT_DIR`; scoring it with `evaluation/score.py` returns a real number. Case IDs, order and count in
`index.json` may differ from the task's split; a package must not rely on them.

## Output directory

`OUTPUT_DIR/cases/<case_id>.h5` with one dataset `prediction` of shape
`(K, C, X, Y)` float32: the predicted frames for the `K` steps after the last
input frame, in the same physical units and channel order as `input`. Write
every case listed in `index.json`. Each file must be a regular HDF5 file and
`prediction` an ordinary dataset stored in it (as `h5py`'s `create_dataset`
writes it): a symlink, an external or soft link, a virtual dataset or external
raw storage is not read. A missing, malformed, non-finite or wrongly shaped
prediction, or one of those, scores as a failed case (see below); it does not
abort the evaluation.

`infer.sh` runs with the verifier image's runtime (Python 3.12, PyTorch 2.14, h5py 3.16,
NumPy 2.4; the same pins as `/opt/runtime/requirements.txt` in the workspace), which installs
nothing, on the verifier's single GPU with 4 hours of wall clock for
all cases (the determinism rerun of three cases shares that budget); it may run
on CPU. Predictions must be deterministic given the
package and the inputs (fix seeds; stochastic samplers must fix their seed).

## Scoring

The verifier holds the true continuation `target` of shape `(K, C, X, Y)` for
every case in a directory the package never sees.

Per case, per predicted step `t` and per **channel** `c`, The Well's
variance-normalized RMSE with the unbiased spatial variance:

```text
VRMSE_c(t) = sqrt( mean_{x,y} (pred - target)^2 / ( var_{x,y}(target) + eps ) ),   eps = 1e-7
```

clamped at `VRMSE_max = 10`. The case score is the mean over channels and
steps; the dataset score is the mean over its cases; datasets form groups (the
pretraining corpus `ape2d/*` is one group, every other dataset is its own
group); the reward is the unweighted mean over groups of

```text
R(v) = clip( (log10(10) - log10(max(v, 0.01))) / (log10(10) - log10(0.01)), 0, 1 )
```

so a group VRMSE of 0.01 gives 1.0, 0.1 gives 0.667, 1.0 (the spatial-mean
predictor) gives 0.333 and 10 or worse gives 0. The grader also reports, for
context and never for ranking, the joint-channel nRMSE
`sqrt(mean_{c,x,y}(pred - target)^2 / mean_{c,x,y} target^2)` of the
PDE-Transformer paper and The Well's rollout windows (steps 1–5, 6–12, 13–30)
where `K` allows. The reference implementation is `evaluation/score.py`.

A failed case (missing, malformed, non-finite, wrong shape) receives
`VRMSE_max` on every channel and step, which keeps it in every denominator.
If `infer.sh` exits non-zero or the time limit elapses, every unwritten case
fails. The verifier records failed cases and their reasons separately from the
score, reruns the package on three cases and records the maximum output
difference (the contract requires deterministic inference), so infrastructure
faults are distinguishable from model quality.

## Evaluation rules

- Inputs are the only observation; no other frames of the same trajectory,
  no dataset files, no network at inference.
- The package may use any resolution or preprocessing internally but must
  return predictions on the input grid.
- Model selection uses the validation trajectories named in the task's split
  manifest; held-out evaluation trajectories are replay-only.
- Synthetic training data generated inside the environment by the package
  author's own solver code is permitted and is declared in `sources_used` as
  `"synthetic:<description>"`; downloading data, or reconstructing held-out
  cases with the benchmark's generator and seeds, is not.
- No pretrained assets: the package's weights must originate from training
  started at random initialization inside the development session;
  `model_manifest.json` lists `"pretrained_assets": []`. Reference packages
  (released checkpoints replayed as the external target) are exempt and
  marked as such.
