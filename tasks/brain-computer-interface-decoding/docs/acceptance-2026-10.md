# Maintainer acceptance, 2026-10-28

Integrator: Weiwei Sun. Scope: the task standard (`TASK_STANDARD.md`), data, verifier and
baseline numbers, on one 8× H100 host. The author's packages were rescored, not retrained.

## Changes made during acceptance

- **Layout.** `RESULTS.md`/`RESULTS.json` moved from the task root into
  `environment/baseline/braindecode/`, so the agent can read them (before, the baseline README
  linked to files outside the image). The RSI record moved to `docs/rsi-experiment.md` and
  `docs/rsi-results.json`, outside the workspace, so agents start from the baselines only.
  Operator campaign scripts moved from `scripts/` to `tests/campaign/`, the FACED position control
  from `environment/controls/` to `tests/controls/`.
- **Instruction.** Rewritten in the five standard sections, with the baseline test scores and
  the standard continuous-research text; it no longer says that beating a baseline is not
  required.
- **Pretrained checkpoints.** Mounted at `/workspace/data/pretrained` from
  `$TASK_DATA_DIR/pretrained` instead of a separate `TASK_PRETRAINED_DIR`. They are fetched from
  their public `braindecode/*` repositories at the pinned revisions; all three hashes match the lock.
- **Publishing.** `train_shared.py` writes its package as a version symlink, which cannot replace
  the real `/workspace/submission` directory and must not be inside a package. The new
  `baseline/braindecode/publish.py` copies the resolved package, optionally runs `infer.sh` on 256
  validation windows of every track and checks the predictions with the verifier's reader, then
  swaps it in.
- **Data release.** AD65, OpenBMI and Longitudinal were rebuilt from the pinned public sources
  with `materialize.py` and published as
  [sunweiwei/ai4sci-brain-computer-interface-decoding-data](https://huggingface.co/datasets/sunweiwei/ai4sci-brain-computer-interface-decoding-data)
  (revision `6b42dfe`, 621 files, 9.65 GB). `materialize.sh release ROOT` (new `release.py`)
  installs that release, the FACED/TUEV runtime of the controlled release and the checkpoints,
  verifying every file. Both releases are pinned in `environment/data/download-source.json`.
- **Resources.** `task.toml` raised to the 1× H100 floors (64 CPUs, 128 GiB, 500 GiB) with
  `gpu_types = ["H100"]` on both sides.
- **Records.** `baseline/braindecode/reproduction/shared-pretrained-validation.json` holds the
  per-epoch validation rewards of the 15 shared references, copied from their packages.
- **README** rewritten in the eight standard sections; `TASK_STANDARD.md` lists this task among
  those with several rewards and no aggregate.

## Data

| track | source of the runtime used here | test windows / subjects | author's record |
| --- | --- | --- | --- |
| FACED | controlled release `1672c03` (`runtime/`, 983 files verified) | 1,932 / 23 | same files |
| TUEV | controlled release `1672c03` | 29,421 / official evaluation set | same files |
| AD65 | rebuilt from ds004504 `63b2ec1` | 1,376 / 18 | 1,376 / 18 |
| OpenBMI | rebuilt from GigaDB 100542 (108 MI objects, MD5-pinned) | 4,400 / 11 | 4,400 / 11 |
| Longitudinal | rebuilt from ds007176 `5578877` | 10,028 / 9 | 10,028 / 9 |

The rebuild took 30 s (AD65), 6 min (Longitudinal) and 30 min (OpenBMI). Sample IDs and the
randomized test order differ from the author's build (different private ID key); signals,
labels and subject splits are deterministic. Checks: no subject, source file or ID is shared
between train, validation and test in any track; test manifests contain only `id`, `path` and
`index`; the agent image contains no test data.

## Replay of the author's packages

Each package was scored by the verifier image (`tests/Dockerfile`) with the task's verifier
resources (one H100, 8 CPUs, 64 GiB, no network), on the data above. Scores in %; differences
are absolute, in reward units, against `RESULTS.json`. Differences of about 1e-4 are one or two
windows changing their argmax on a different GPU; the AD65 and OpenBMI scores on the rebuilt data
match exactly or within one trial.

| shared package | FACED | TUEV | AD65 | OpenBMI | Longitudinal | largest difference | verifier seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cbramod-s0 | 15.08 | 45.08 | 51.59 | 60.36 | 74.33 | 0.000000 | 79 |
| cbramod-s1 | 15.42 | 32.78 | 50.79 | 56.36 | 75.37 | 0.000000 | 80 |
| cbramod-s2 | 13.98 | 44.17 | 46.83 | 52.07 | 74.10 | 0.000237 | 79 |
| cbramod-s3 | 14.69 | 42.14 | 50.79 | 58.82 | 74.07 | 0.000098 | 80 |
| cbramod-s4 | 15.39 | 39.37 | 58.25 | 59.43 | 74.85 | 0.000089 | 80 |
| biot-s0 | 16.96 | 50.90 | 68.57 | 61.36 | 75.53 | 0.000180 | 72 |
| biot-s1 | 17.40 | 51.68 | 59.68 | 55.07 | 72.72 | 0.000000 | 71 |
| biot-s2 | 16.76 | 49.24 | 64.92 | 60.55 | 75.46 | 0.000000 | 72 |
| biot-s3 | 16.88 | 51.18 | 41.59 | 57.07 | 74.03 | 0.000089 | 67 |
| biot-s4 | 15.73 | 50.45 | 63.81 | 61.57 | 73.38 | 0.000092 | 67 |
| codebrain-s0 | 20.38 | 57.23 | 47.94 | 62.09 | 78.26 | 0.000059 | 236 |
| codebrain-s1 | 21.03 | 58.53 | 51.59 | 63.25 | 75.37 | 0.000227 | 239 |
| codebrain-s2 | 18.18 | 51.00 | 65.71 | 63.34 | 77.41 | 0.000227 | 235 |
| codebrain-s3 | 19.94 | 58.21 | 69.68 | 63.75 | 76.83 | 0.000227 | 236 |
| codebrain-s4 | 18.34 | 53.87 | 63.81 | 64.11 | 76.58 | 0.000000 | 237 |

| specialist package | track | replayed | author | difference |
| --- | --- | ---: | ---: | ---: |
| eegnet-ad65-s0 | ad65 | 51.59 | 51.59 | 0.000000 |
| eegnet-ad65-s1 | ad65 | 46.03 | 46.03 | 0.000000 |
| eegnet-ad65-s2 | ad65 | 53.49 | 53.49 | 0.000000 |
| eegnet-ad65-s3 | ad65 | 42.38 | 42.38 | 0.000000 |
| eegnet-ad65-s4 | ad65 | 51.59 | 51.59 | 0.000000 |
| eegnet-openbmi-s0 | openbmi | 77.84 | 77.84 | -0.000000 |
| eegnet-openbmi-s1 | openbmi | 76.27 | 76.27 | 0.000000 |
| eegnet-openbmi-s2 | openbmi | 77.52 | 77.52 | 0.000000 |
| eegnet-openbmi-s3 | openbmi | 78.02 | 78.02 | -0.000000 |
| eegnet-openbmi-s4 | openbmi | 76.84 | 76.84 | 0.000000 |
| biot-longitudinal-s0 | longitudinal | 76.02 | 76.02 | 0.000000 |
| biot-longitudinal-s1 | longitudinal | 75.35 | 75.35 | -0.000000 |
| biot-longitudinal-s2 | longitudinal | 77.55 | 77.55 | 0.000000 |
| biot-longitudinal-s3 | longitudinal | 75.22 | 75.22 | 0.000000 |
| biot-longitudinal-s4 | longitudinal | 76.01 | 76.01 | 0.000000 |
| codebrain-faced-s0 | faced | 51.03 | 51.03 | 0.000000 |
| codebrain-faced-s1 | faced | 52.71 | 52.71 | 0.000000 |
| codebrain-faced-s2 | faced | 44.35 | 44.35 | 0.000000 |
| codebrain-faced-s3 | faced | 12.27 | 12.27 | 0.000000 |
| codebrain-faced-s4 | faced | 50.76 | 50.76 | 0.000000 |
| codebrain-tuev-s0 | tuev | 64.39 | 64.39 | 0.000000 |
| codebrain-tuev-s1 | tuev | 61.85 | 61.85 | 0.000000 |
| codebrain-tuev-s2 | tuev | 61.73 | 61.73 | 0.000000 |
| codebrain-tuev-s3 | tuev | 57.49 | 57.49 | 0.000000 |
| codebrain-tuev-s4 | tuev | 61.35 | 61.35 | 0.000036 |

| RSI selected package | 24.24 | 65.62 | 65.24 | 77.84 | 77.49 | largest difference 0.000000 |

The five-seed means of the replayed shared references equal those in `RESULTS.md` to two
decimals.

## Harbor end to end

With `harbor-runner:0.23.0` and the `science-speedrun` runner (`run_verify.sh`, copy agent,
separate verifier built from `tests/`):

| trial | rewards (faced / tuev / ad65 / openbmi / longitudinal) | time |
| --- | --- | --- |
| author's shared CodeBrain s0 | 0.2038 / 0.5723 / 0.4794 / 0.6209 / 0.7826 (difference ≤ 6e-5) | 6 min including image builds |
| maintainer pilot (CodeBrain, 100 updates, published with `publish.py --check`) | 0.1111 / 0.2718 / 0.2778 / 0.5014 / 0.5035 | 4.6 min |
| `nop` agent, empty submission | 0 on every track, `invalid_submission`, no crash | 50 s |

The pilot ran in the agent image with the compose mounts (five tracks and `pretrained/`
read-only): EEGNet 100 updates in 5 s, CodeBrain in 81 s; the published submission is a plain
directory without symlinks.

## Fresh install

`bash environment/data/materialize.sh release ROOT` in a clean Python 3.12 container with
`requirements-data.txt` took 7 min 15 s: 611 public and 983 controlled runtime files verified,
three checkpoints matched the lock. The installed tree has the same 1,597 files (paths and sizes)
as the tree used above, and the shared CodeBrain s0 package scores the same on it (largest
difference 6e-5).

## Other checks

- 22 unit tests (`python -m unittest discover -s tests`) pass in the agent image.
- `tests/container_probe.py`: the valid probe package scores 1; missing, NaN, duplicate and
  wrong-class outputs score 0 as `invalid_submission`.
- `validate_pretrained.py` and `tests/test_shared_matrix.py`: all three checkpoints load into all
  five tracks.
- `layout_check.py` and `scripts/audit_tasks.py`: no findings.

## Agent behavior test

One Codex session (`gpt-6.1-sol`, reasoning medium, Codex 0.163.0, continuing-resume runner from
`science-speedrun`) ran for one hour on one H100 with the task's resources, then its last package
was scored by the Harbor verifier:

| | FACED | TUEV | AD65 | OpenBMI | Longitudinal |
| --- | ---: | ---: | ---: | ---: | ---: |
| agent, 1 h (test) | 35.13 | 59.51 | 71.59 | 73.02 | 75.25 |
| agent's last validation | 34.52 | 52.19 | 70.79 | 74.14 | 77.55 |
| shared CodeBrain, pretrained (five-seed test mean) | 19.58 | 55.77 | 59.75 | 63.31 | 76.89 |

The agent read the contract, baseline code and `reproduction/` records, published a pilot package
with `publish.py --check` two minutes in, and then tested several shared architectures (spatial
readout on CodeBrain, a scratch spectral covariance/power encoder, a temporal EEGNet branch and a
fused temporal–spectral model). It selected checkpoints on validation only, with per-track
regression guards, and published 24 versions. The final package is one jointly trained
temporal–spectral network with five heads. It normalizes each window by itself, runs in eval
mode and uses no test-time adaptation or subject information. Test scores track validation.
AD65 has 18 test patients, so one patient moves it by about 5 points.

The session found two environment defects, both fixed in this PR:

- `/dev/shm` was Docker's 64 MB default, so `--num-workers 2` failed; the compose file now sets
  `shm_size: "16gb"`.
- `shared_infer.sh` changed directory before passing its arguments on, so relative input/output
  paths broke; it now calls `shared_infer.py` by path (the verifier passes absolute paths, so
  scores were not affected).

## Not done

- Baselines were not retrained; the author's training records are in `reproduction/` and the
  author's evidence archive.
