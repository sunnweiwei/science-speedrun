# Training and evaluation resources

```text
/workspace/data/
  README.md                          this file
  release.json                       exact membership, runtime profiles and exclusions
  source-files.json                  Git blob/LFS hashes and byte sizes
  native-configs.json                pinned source configurations
  selection-research-v2.json, SELECTION-RESEARCH-V2.md   selection rationale and per-case costs
  *.py, other *.json, *.md           release code and selection history the evaluator imports; not data
  release/                           the prepared release, read-only (also reachable as /data)
    release.json  materialized.json
    environments/<CASE>/...          geometry, grids, configuration and initial flow states, 39 cases
```

Write generated experience, checkpoints and runs under `/workspace/research`.

## One frozen release, outside Git

`hydrogym-public-research-v2` contains exactly **39 cases (20 2D + 19 3D)**,
558 upstream files, **453,756,315,982 bytes**. The 500 GB ceiling covers one
shared copy of source assets, not images, simulation scratch, generated
experience or model checkpoints.

- Download repository/type: `sunweiwei/ai4sci-hydrogym-research-v2` / `dataset`.
- Download revision: `3ce0a7e1c394d52b1c8b2c201f01070a8698f07b`.
- Original authors' dataset: `dynamicslab/HydroGym-environments`, revision
  `cf5ef6748443ae72e9647f398487e98d421a7243` (byte-level scientific provenance).
- [release.json](release.json): exact membership, runtime profiles and exclusions.
- [source-files.json](source-files.json): Git blob/LFS hashes and byte sizes.
- [native-configs.json](native-configs.json): pinned source configurations.
- [selection/cost table](SELECTION-RESEARCH-V2.md) and [selection-research-v2.json](selection-research-v2.json): rationale and per-case costs.
- Canonical release SHA256:
  `cabc8b718cdf9b22fcd661b9b3c5f8e452498e1c299b0fc9c7ac1afc22df47d2`.

These files and their identities are unchanged by the baseline publication.
Historical release-500gb/speed/coverage/hour manifests explain selection history; they are **not**
additional data for this task.

## What is the training data?

This is interactive reinforcement learning. Prepared files contain geometry,
grids, boundary/configuration data and initial flow states. Training experiences
(observation, action, reward, next observation) are generated with the pinned
native simulator. There is no fixed offline-transition count or separate large
training corpus. Agent code may collect/reuse interactions from the 39 selected
environments, including reference-controller trajectories in those environments.

Training and evaluation intentionally use the same known configurations and
initial-state pools. All specifications, code and reference scores are visible.
This is not a private test or unseen-task transfer benchmark. During independent
evaluation, the candidate's weights and learned normalizers must stay frozen.
Recurrent episode state and non-learning planning are allowed.

## Physics, metrics and licensing

Training/evaluation use [case_spec.py](../evaluation/case_spec.py) and the same
native adapters. Preserve grids, dt, sensors, actuator bounds, control intervals
and horizons. The Channel actuator-clock repair and fine-nozzle restart binding
are explicit organizer profiles, not an assertion of unmodified paper physics.
The 39 [primary metrics](../evaluation/METRICS.md) are computed by the evaluator.

Access and redistribution follow upstream dataset, solver and asset terms.
HydroGym code being MIT does not license every third-party binary or resource.
This Git repository contains no large data, model weights, tokens or solver
binaries. The selected unchanged data bytes are mirrored on HF with source
attribution; the upstream data card at the pinned revision declares no dataset
license, and this mirror does not assert a new blanket license for those assets.
