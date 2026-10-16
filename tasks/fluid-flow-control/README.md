# Fluid Flow Control (HydroGym)

## 1. Overview

Develop **one jointly trained closed-loop controller for all 39 known flow-control
cases**. Training interactions with these environments are allowed; independent
evaluation freezes weights and learned normalization. This is known-task
multi-task RL, not a held-out-task or zero-shot generalization benchmark.

- Native Harbor task: `instruction.md`, `task.toml`, `environment/`, `tests/test.sh`.
- Data: `hydrogym-public-research-v2`, 20 two-dimensional and 19 three-dimensional
  cases; 558 files, **453,756,315,982 bytes** shared by training and evaluation.
- Public HF data mirror: `sunweiwei/ai4sci-hydrogym-research-v2`, revision
  `3ce0a7e1c394d52b1c8b2c201f01070a8698f07b`. No large data or checkpoints in Git.
- Delivery: [contract](environment/contract.md), one portable checkpoint/code
  package at `/workspace/submission`, resident JSONL `infer.sh`.
- Metrics: [one fixed physical primary metric per case](environment/evaluation/METRICS.md).
  Harbor emits 39 named physical values, not a mixed-unit total or training return.
- Reference results: [39/39 verified specialist baselines](environment/references/specialists/verified/README.md).
  Several trained specialists do not beat the uncontrolled flow (for example
  Cylinder_3D_Re3900: 1.2067 vs 1.1114); they are reported as measured.
- Specialists are comparison/reference implementations, **not** the required
  shared-model submission. Small-scale budgets are not full-paper/SOTA claims.

**Status.** 39 per-case specialist references verified by the integrator (values cross-checked against the campaign's raw
evaluation results; five packages replayed on GPU through the verifier). The shared starter is runnable; no shared-model result
is claimed. The workspace follows [TASK_STANDARD.md](../TASK_STANDARD.md).

## 2. Data

**Harbor code stays on GitHub; the actual large data files are on
[Hugging Face](https://huggingface.co/datasets/sunweiwei/ai4sci-hydrogym-research-v2).**
All 558 files / 453,756,315,982 bytes are present at the immutable revision in
`environment/data/download-source.json`. The materializer defaults to that
public, ungated data mirror and verifies the original upstream hashes. No HF
token is needed to download. See [data hosting](environment/data/README.md#hugging-face-data-hosting).

From the repository root, choose your own large-disk path:

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2
# The command above prints a no-download plan. To actually fetch and verify:
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2 --download
export TASK_DATA_DIR=/large-disk/hydrogym-public-research-v2
```

The operator installs `hf` for preparation; the task helper pins and checks the
upstream files. Harbor does not automatically run this helper. The native Compose
definitions bind the prepared root read-only to `/data` for both development and
verification. See [data documentation](environment/data/README.md) for offline
reuse, integrity, visibility and licensing. No account token enters the task.

The prepared release (`TASK_DATA_DIR`) is mounted read-only at `/workspace/data/release` in development (the image links `/data`
to it, the evaluator's default) and at `/data` in the verifier. There is no private label set: the verifier replays the same CFD
cases.

## 3. Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md`, `/workspace/contract.md` | the prompt; model package and inference contract |
| `/workspace/data/` | manifests, preparation docs and descriptors (image); `release/`, the prepared release (read-only mount) |
| `/workspace/baseline/starter/` | shared PPO: native training/resume of one file, atomic zipapp publication, inference |
| `/workspace/references/specialists`, `/workspace/references/results` | the verified per-case PPO/DDPG/TD3 trainer and recipes; 39/39 specialist results |
| `/workspace/evaluation/` | `METRICS.md`, `case_spec.py`, the self-service native evaluator |
| `/opt/references/hydrogym-author`, `/opt/hydrogym`, `/opt/references/tdmpc2` | authors' model/loss source, upstream physics, an optional architecture reference |
| `/workspace/submission/`, `/workspace/research/` | the current package, and writable scratch on a large disk (`TASK_RESEARCH_DIR`) |

## 4. Evaluation

Follow [starter instructions](environment/baseline/starter/README.md). Every training phase
resumes **the same** PPO file. The native entry supports MAIA, JAX and JAX-Fluids.
The starter is a research starting point, not a competitive shared-model result.

For a development check inside the runtime:

```bash
bash /workspace/evaluation/native_runtime.sh maia python /workspace/evaluation/harbor_verify.py \
  --data /workspace/data/release --submission /workspace/submission \
  --case Cylinder_2D_Re200 --output /workspace/research/check-001
```

The trusted separate verifier runs `tests/test.sh` on the **26-case verifier suite** (every 2D case and
one 3D case per family, side by side on eight GPUs: 2 h 21 min on 8×H100, set by Cube_3D_Re3700) and the
same snapshot. It writes numeric per-case metrics to `/logs/verifier/reward.json`
and detailed identities, directions, failures and timings under `replay/`.
There is no missing `suite.json` or reference-anchor preparation step.
See [evaluation instructions](environment/evaluation/README.md).


## 5. Running

```bash
export TASK_DATA_DIR=/large-disk/hydrogym-public-research-v2                        # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/hydrogym-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p fluid-flow-control -a <agent> -m <model> --allow-agent-host <model API host>
```

The external budget is set by the controller; `task.toml` carries the standard 48 h default.

The task requests 8 H100s for development. Three frozen solver profiles need
two GPUs per simulation; extra GPUs do not imply automatic linear speedup.
Large MAIA cases also require substantial host RAM. The separate verifier uses
sequential case replay, 512 GiB host RAM and a conservative seven-day watchdog;
this is not an agent-session duration or a measured completion-time promise.
Source data is ~454 GB; allow additional storage for images, checkpoints,
solver scratch and generated experience.

Current references cover all 39 independently evaluated specialist cases
using budgets of 4,000 or 20,000 controls. Historical long paper-profile runs are
recorded separately. Full-suite acceptance of one jointly trained competitive
model and a formal agent experiment are not claimed complete. See [validation](docs/VALIDATION.md) and [remaining research](docs/ROADMAP.md).

The immutable older data manifests, paper audits and local experiment scripts
remain in Git as historical provenance. They do not redefine this task's current
39-case data, evaluator, starter or metrics.

## 6. Results and validation

The 39 specialist results, with uncontrolled and paper references, are in `environment/references/specialists/verified` (copied to
`/workspace/references/results`); the integrator's cross-check and GPU replays are recorded in [docs/VALIDATION.md](docs/VALIDATION.md)
and the root README's Verified tasks table.

## 7. Limitations

- Known-task multi-task RL: training and evaluation use the same task configurations; this is not held-out generalization.
- The native solvers are not all run-to-run deterministic (Nozzle2D_coarse varies about 0.5 %).
- The selected public-resource suite is not the original full paper suite; see [docs/ROADMAP.md](docs/ROADMAP.md).

## 8. Citation and licenses

- HydroGym (upstream physics and environments), pinned in the image; attribution and licenses are recorded with the
  pinned sources in `environment/references/README.md`.
