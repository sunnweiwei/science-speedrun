# PR acceptance record — 2026-09-18

## Layout check after TASK_STANDARD.md (2026-09-27)

The task-root `baselines/` moved to `environment/references/specialists/` (only its curated subset
is copied into the image, as before), `VALIDATION.md` and `ROADMAP.md` moved to `docs/`, the
starter to `environment/baseline/starter/`, and the release mount to `/workspace/data/release`
(`/data` links to it) with `TASK_DATA_DIR`. Checked on one host with rebuilt images and the task's
own compose files, GPU 5:

- the development container mounted the 454 GB release read-only and an empty
  `/workspace/research`; the preflight (`evaluate_native.sh --case all --preflight`) passed;
- `init_random.py` published into the image-layer `/workspace/submission`; the development replay
  of a random package on Cylinder_2D_Re200 gave `drag_cd` 1.3518;
- the verifier (`tests/docker-compose.yaml`, policy isolation on) replayed the same package on the
  same case in 180 s with the identical `drag_cd` 1.3518;
- unit tests: `tests/` 156 passed (1 skipped), `environment/references/specialists` 45 passed,
  `public_release.py --check` reproduced the release SHA-256 `cabc8b71…`.


This records task-integration checks, not a full-paper reproduction or a
competitive jointly trained controller. The separately published
[specialist evidence](../environment/references/specialists/verified/README.md) contains the baseline scores.

## Format, data and regression checks

- Harbor **0.22.0** accepted `task.toml` through `TaskConfig.model_validate_toml`:
  task version 0.7.0, separate verifier, `/workspace/submission` artifact.
- Both native Compose definitions passed `docker compose config --quiet`.
- Development and verifier Docker targets built successfully from this task.
- Frozen release check: **39 cases, 558 files, 453,756,315,982 bytes**; release
  hash `cabc8b718cdf9b22fcd661b9b3c5f8e452498e1c299b0fc9c7ac1afc22df47d2`.
  No source data, membership or native physics was changed for publication.
- All **39 source/profile bindings** passed preflight in the clean verifier.
  This rechecks the existing materialization receipt, sizes and small-file
  hashes; it does not claim a new 454 GB full-byte rehash in this PR check.
- **145 task tests + 45 baseline tests = 190 passed, zero skips**, using the
  staged repository in the dependency-complete offline runtime. This includes
  shared training/resume, portable export/readability, physical metric extraction,
  invalid actions/regimes, immutable snapshots, exact case coverage, GPU binding
  and the Harbor bridge's all-39 default. Synthetic fixtures are not CFD scores.

```bash
python -m unittest discover -s hydrogym/tests -q
python -m unittest discover -s fluid-flow-control/environment/references/specialists -p 'test_*.py' -q
python fluid-flow-control/environment/data/public_release.py --check
```

## Real training from the development image

The curated agent-visible commands were run with network disabled, immutable
source data, one H100 per experiment, four host CPUs and 32 GiB host RAM.

1. `/workspace/references/specialists/research_train.py`: public-source PPO,
   Cylinder_2D_Re200, seed 42, **400 controls**, checkpoint exported successfully.
2. `/workspace/baseline/starter/train_native.py`: one shared PPO resumed across
   Cylinder_2D_Re200 (**400**) → Pinball_2D_Re100 (**400**) →
   Channel_3D_Retau180 (**200**). All three phases completed. The final zipapp
   records **1,000 training controls and 289,585 parameters**; it contains one
   parameter set with capacities for all 39 case interfaces, not three models.

The smoke-training development image was
`sha256:e4651e18b5e8af931335d9f0595cabc9b575a581129b01d1035e86390c62d9d6`.
Subsequent publication-permission/error-cleanup hardening is regression-tested;
neither changes the policy computation or the successful training path.
These short budgets exercise functionality, not useful convergence or a target
model size. No optimizer state or trained weights are committed to Git.

## Independent clean-verifier replay

Each experiment's package was mounted read-only into a **new, offline verifier
container**, with only `/data`, `/workspace/submission` and writable report output.
No training directory or task-code mount was supplied. The verifier replayed the
shared package sequentially on all three listed cases from the same snapshot.
The following are actual full-episode **integration-check results**, not the
published trained specialist baselines:

| Package | Case | Primary metric ↓ | Value | Complete controls | Replay wall time |
| --- | --- | --- | ---: | ---: | ---: |
| Shared 1,000-control starter | Cylinder_2D_Re200 | Mean CD, last half | 1.35365827 | 201 | 30.18 s |
| Same shared package | Pinball_2D_Re100 | Mean total CD, last half | 2.51191605 | 201 | 60.36 s |
| Same shared package | Channel_3D_Retau180 | Mean native wall shear, last half | 0.24766077 | 100 | 36.04 s |
| Reference PPO, 400 controls | Cylinder_2D_Re200 | Mean CD, last half | 1.33845974 | 201 | 28.28 s |

Both runs exited 0 and wrote numeric `reward.json`, complete `results.json` and
`grading.json` with `scope=development-subset`. These checks did not run the
costly full-39 default `tests/test.sh` or a formal Harbor agent session.

Evidence identities:

- Verifier image:
  `sha256:b8b0d09584bb34aa1ca37fc4079c340ed0a0386c5aa38ee8a512d0b967a46421`.
- Executed evaluator:
  `e2b3acf6dd8ae9ab73b8b810940ecb4aadc2ab52c424566417a3ff0411a5be9c`.
- Native profiles:
  `55d6cd1db2b6c3fe6c3dfb6be963e4c2ab69ba37cd4501b410867bdb2e6e48ce`.
- Shared submission:
  `53860d318cc8d9c53735c9d70e10f5893c353b657535d7040f2ef497db810250`.
- Reference submission:
  `c5d25411b53db86f5b9b105e1574744ffbee7779ab1a9a90de207b59e6143ea5`.

Use the [clean-container commands](environment/evaluation/README.md) to repeat
the check. Exact numerical equality across hardware/software is not promised.

## Published baseline scope and remaining work

The independent specialist snapshot contains **39/39** paired uncontrolled and
trained physical results (Cylinder_3D_Re3900 was appended on 2026-09-19, see the
integration review below). It includes seeds, actual budgets, configurations,
selected checkpoint identities, evaluation hashes and same-metric approximate
paper references.

These are per-case references, mostly 4,000 or 20,000 training controls, not
full-paper convergence, multiple-seed confidence intervals, or one shared
SOTA checkpoint. Historical longer paper-profile runs are separately labeled.
All-39 acceptance of a competitive jointly trained model remains future research.
The local campaign is not modified by these publication/integration checks.

## Actual data on Hugging Face — 2026-09-19

Corrected the distribution boundary: **Harbor code is on GitHub; actual large
CFD data is on Hugging Face**. The earlier code/manifest-only HF upload was a
packaging mistake. Its 182 Harbor files and `bundle.json` were removed from the
current Hub tree; Git and Hub history retain recoverable prior versions.

Data repository: `sunweiwei/ai4sci-hydrogym-research-v2`, complete immutable
revision **`3ce0a7e1c394d52b1c8b2c201f01070a8698f07b`**.
The `environments/` tree now contains all **558 real source files /
453,756,315,982 bytes** for the 39 selected tasks. The only other files are the
data card, original `release.json` and Hub `.gitattributes`; no Harbor code,
model weights, local receipts or credentials are included.

Checks used `huggingface_hub` / `hf` **1.32.0**:

- The resumable uploader processed all 560 candidate files (558 resources plus
  README/release), hashing the complete payload and committing every file.
- Anonymous Hub metadata verification at the final revision matched exact
  membership, all 558 sizes, **104 LFS SHA256 hashes and 454 Git blob SHA1 hashes**
  against the original frozen source index. The repository is public and ungated.
- The GitHub materializer now defaults to this actual-data mirror at the exact
  pinned commit, producing 39 commands for 453,756,315,982 source bytes.
- Using that new default and an empty local directory/cache, anonymously
  downloaded **Cylinder_2D_Re200** from **our mirror**: 12 files / 69,499,887 bytes.
  Every byte passed Git/LFS verification and a second offline check.
- Its receipt preserves original scientific source identity and separately
  records the mirror repository, revision and path prefix. Source release hash,
  physics, metrics and existing local data compatibility are unchanged.
- All **39** source/profile preflights still pass; profile hash remains
  `55d6cd1db2b6c3fe6c3dfb6be963e4c2ab69ba37cd4501b410867bdb2e6e48ce`.
- Updated committed task suite in the offline dependency-complete container:
  **151 task + 45 baseline = 196 tests passed, zero skips**, including six new
  mirror-selection, pinning, layout, checksum and receipt regression tests.

This verifies complete data publication and the new download path. It does not
claim downloading the entire 454 GB back to a second local copy or rerunning the
full CFD benchmark. No scientific data, metric or model contract was changed.
