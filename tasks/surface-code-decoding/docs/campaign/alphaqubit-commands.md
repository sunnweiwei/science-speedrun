# AlphaQubit reproduction commands

This is the active AlphaQubit procedure; `commands.md` records the retired
QRENND campaign. Run from the surface-code-decoding task directory.

## Environment and tests

```bash
uv sync --frozen --python 3.11 --project alphaqubit-runtime
export QEC_TASK_ROOT=/home/users/xy200/yinxunjian/mycode/ai4sci-tasks-qec/surface-code-decoding
export QEC_ASSET_ROOT=/home/users/xy200/yinxunjian/mycode/ai4sci-artifacts/surface-code-decoding/20260919
export QEC_DATA_ROOT="${QEC_ASSET_ROOT}/data/sycamore"
export PYTHONPATH="${QEC_TASK_ROOT}"
```

Use `alphaqubit-runtime/.venv/bin/python` explicitly. CPU tests need
`JAX_PLATFORMS=cpu`; on shared CPU allocations, constrain affinity to a small
set of allocated cores before importing JAX. GPU jobs set their own JAX/CUDA
runtime environment; no system CUDA module is required by the pinned runtime.

## Bounded representative training

The initial run selects groups 0 and 8 from `groups.txt`: d3/X/center(3,5) and
d5/X/center(5,5). The model is full size (approximately 8.45M parameters).

```bash
export QEC_STAGE=pretrain QEC_TRAIN_HOURS=2
sbatch --parsable --array=0,8%2 --signal=B:USR1@90 \
  --output="${QEC_ASSET_ROOT}/logs/alphaqubit-representative-%A-%a.log" \
  slurm/alphaqubit-train.sbatch
```

Register every returned job ID, array size and hard time cap in the external
`campaign.json` before further submission. Its conservative ceiling includes
settled retired allocations and the full caps of all unretired jobs. Accounting
must remain below 384 actual GPU-hours and eight concurrent GPUs.

The trainer streams independently seeded even-fitted pij simulations. Its
2-hour limit is cumulative across resumes; the Slurm hard limit is 2h15m per
allocation. It uses batch256, BF16 compute, FP32 master parameters, the literal
optimizer-step batch-switch counter, and 5120 independent full-noise validation
shots per duration. Validation uses EMA and the published fit rejection gate.
The cumulative limit normally stops15min before the Slurm deadline. New
allocations use `--signal=B:USR1@90` because the exec Python process is the
batch process ([Slurm signal semantics](https://slurm.schedmd.com/sbatch.html#OPT_signal)).
The original representative submission used the step-only warning option;
its own cumulative limit remains the planned stopping mechanism.
A best.npz is emitted only after an accepted validation LER fit. A poor early
fit does not establish that this architecture cannot learn.

`last.pkl` is a trusted task-owned resume artifact, never an inference format.
Source, model, data, schedule, and initializer identities must match on resume.
Do not edit semantic training sources during a campaign; a deliberate change
requires a separately identified run. `best.npz` is pickle-free portable FP32.

## Expansion, fine-tuning and held-out replay

Expansion is conditionally queued behind `slurm/alphaqubit-gate.sbatch`,
which runs only after both representatives complete. It reconstructs and hashes
the exact full-noise validation sets, independently scores standard MWPM, and
checks finalized checkpoint evidence. Both groups require at least10k updates,
4M samples and three complete validations; accepted selected LER must be no
more than2× MWPM, with at least10% shot-error improvement over no correction.
Main BCE must improve10% versus the first complete snapshot, or selected LER
must already be within1.25× MWPM. Failure cancels unstarted dependent jobs and
preserves evidence for diagnosis; it does not establish scientific failure.
Every evaluated, valid selection is frozen into a versioned representative
package with a validation-history sidecar before a full resume overwrites the
original stage. Package paths and hashes are included in the gate evidence.

The planned chain uses18h cumulative pretraining per group: eight new groups
have18h15m allocation caps (max6 concurrent), while the two representatives
resume with16h15m caps (max2 concurrent). Fine-tuning caps are2h15m each,
replay caps45m each. Including the earlier campaign and diagnostic probes,
all hard caps total359.036944 GPU-hours, with24.963056 uncommitted.
Actual job IDs, dependencies, environment and original caps live in campaign.json.
`QEC_TRAIN_HOURS` remains cumulative, whereas `sbatch --time` limits each new
allocation. Sum the original hard caps of all submitted allocations; do not
assume queued work or checkpoint resumes are free.

For hardware adaptation set `QEC_STAGE=finetune`. The script initializes each
group from its pretrained best.npz, resets optimizer moments, and anchors
weight decay to those exact weights. Hardware effective batch is1024 through
microbatch accumulation; the default maximum is30,000 updates. The allowed
19880 even training and5120 even validation shots per condition are fixed.

After all ten stages have stopped and accepted selections exist, the dependency
chain is:

1. `slurm/alphaqubit-freeze.sbatch`: freeze both stages, relocate a complete
   inference package, create a detector-only data copy, install its locked env.
2. `slurm/alphaqubit-replay.sbatch`: typed RTX PRO 6000 array, max8 concurrent;
   each group writes its own prediction directory for both stages.
3. `slurm/alphaqubit-score.sbatch`: trusted CPU scoring of all3.25M odd shots per
   stage with paired classical comparisons and2000 bootstrap draws; update HTML.
4. `slurm/alphaqubit-status.sbatch`: an `afterany` terminal CPU job refreshes
   accounting and HTML even when a prerequisite fails. Submit all dependency
   jobs with `--kill-on-invalid-dep=yes` so cancellations propagate.

Do not launch odd-test inference while using validation to select methods or
hyperparameters. Store complete raw probabilities, shot IDs and all provenance.
Fresh relocated inference and shot-identical replay must be checked before a
result is declared reproducible. This first campaign is one seed/fold; it does
not reproduce the paper's full ensemble or d7–11 Pauli+ study.

## Read status

The first complete run finished on 2026-09-22 at 02:16 UTC. Its immutable
scores and independently checked predictions are under `alphaqubit-v1/`.
Fine-tuned test LER is3.01265%/3.22889% (d3/d5); official-level accuracy has
not been reproduced across both distances.

The completed bounded follow-up uses `slurm/alphaqubit-d5-30k.sbatch` for groups8/9
and `slurm/alphaqubit-d5-30k-score.sbatch` for scoring. Before submission,
the task copied both finalized d5 fine-tuning directories, including exact
optimizer/RNG/EMA/anchor resume state, to `alphaqubit-d5-30k/models/` and
verified all file hashes under shared training locks. All eight original d3
prediction directories were copied into the separate follow-up prediction
tree. Copies and the original score identity are recorded in decision.json.
Do not resubmit these jobs or overwrite that directory while the chain runs.

Array12679372 resumes the original30k-step cap with a4.5h cumulative stage
limit, freezes each selected model and predicts using the previously installed
locked environment and detector-only inputs. Each task has a3h allocation cap
including inference, at most six extra GPU-hours total. Score12679373 requires
both array tasks to succeed, checks unchanged d3 artifacts and produces a
separate scores-finetune.json and comparison.json. Its first post-score check
incorrectly required stochastic bootstrap CIs to stay identical. CPU repair
12681093 passed after restricting exact equality to hashes, condition results
and deterministic macro fields; no GPU rerun was needed. Status12679374 refreshes
shared accounting/reporting after either scoring success or cancellation.
All sources, commands and IDs are retained in submission-audit.json and the
shared campaign.json ledger. Selection still uses even validation; the public
odd test was already inspected, so this is explicitly a post-test follow-up.

```bash
alphaqubit-runtime/.venv/bin/python -m baselines.accounting \
  --asset-root "${QEC_ASSET_ROOT}"
alphaqubit-runtime/.venv/bin/python -m baselines.alphaqubit_report \
  --asset-root "${QEC_ASSET_ROOT}"
```

The editable HTML snapshot is `alphaqubit-v1/report.html` under the asset root.
Training emits atomic progress.json, metadata.json, history.jsonl, resumable
last.pkl, and validation-selected best.npz/best.json. Generated artifacts and
large data never belong in Git.
