# The standard benchmark

The fixed protocol for comparing two checkpoints on this task. Operator-side: this
directory, like the rest of `benchmark/`, is excluded from both images.

There are **two** definitions. Both are frozen; pick one and say which in any table.

| definition | scope | cost |
| --- | --- | --- |
| **`targets-full.json`** | **51 targets** x 100 designs x 4 sequences = **5,100 designs** - every target the task defines | ~12 GPU-h generate, ~350 to fold |
| `targets.json` | 14 targets x 100 x 4 = 1,400 designs - enzyme sampled down to 4 | ~6 GPU-h generate, ~43 to fold |

| track | targets | length |
| --- | --- | ---: |
| unconditional | `uncond_L100` (synthetic - a length, no context) | 100 |
| sm_binder | FAD, IAI, OQO, SAM (all 4) | 150 |
| ppi_binder | rfd3_il2ra, rfd3_il7ra, rfd3_insulinr, rfd3_pdl1, rfd3_tie2 (all 5) | 80-92 |
| enzyme (**full**) | all 41 M-CSA targets | 180 |
| enzyme (sampled) | M0092_1dli, M0096_1chm, M0500_1e3i, M0907_1rbl | 180 |

**Prefer `targets-full.json`** - it is what the reference row in
`environment/evaluation/RESULTS.md` reports, and the enzyme track turned out to be
strongly bimodal (28 of 41 targets score exactly 0.00, two exceed 0.55), so a four-target
sample carries almost no information about the spread. Measured: the sampled four average
0.0475 against the other 37's 0.0543, so the draw was mildly pessimistic about the *mean*
and completely uninformative about the *distribution*.

`targets.json` is kept **unchanged** so the earlier 14-target row stays reproducible. Its
enzyme four were a fixed random draw with the seed recorded (`20260919`), because a
benchmark that re-randomises its targets cannot compare two checkpoints, which is the
entire point. Re-drawing is a deliberate act that starts a new benchmark version - bump
`name` and say so in any table that mixes the two.

**Numbers from the two definitions are different target sets, never a delta.** Only the
enzyme track differs between them (0.054 full vs 0.048 sampled); nothing improved.

This benchmark reports **per-track pass rates and no composite**. Averaging the tracks
would mix RFdiffusion3's three published tracks with this task's synthetic
`unconditional` one, which has no published criterion, no target in the evaluation view,
and a pass rate an order of magnitude higher than the rest - so the mean would be
dominated by the one track nobody publishes. It would also be read as the graded reward,
which `tests/grader.py` computes differently and writes to `reward.json`.

## Running it

All four stages live here. `generate.sh` and `score.sh` are the sbatch drivers that
produced the reference row in `environment/evaluation/RESULTS.md`.

```bash
export STANDARD_BENCH_WORK=<a FRESH work dir>
B=$PWD                                   # this directory

# 1. specs: upstream's published input.yaml, lengths pinned
python $B/make_specs.py --out $STANDARD_BENCH_WORK/inputs --definition $B/targets-full.json

# 2 + 3. generate then score, ONE ARRAY TASK PER TARGET, per track
mkdir -p $STANDARD_BENCH_WORK/logs && cd $STANDARD_BENCH_WORK/logs
for s in unconditional:0-0 sm_binder:0-3 ppi_binder:0-4 enzyme:0-40; do
  TRACK=${s%%:*}; RANGE=${s##*:}
  gen=$(TRACK=$TRACK CKPT=<ckpt> sbatch --parsable --array=$RANGE $B/generate.sh)
  TRACK=$TRACK sbatch --array=$RANGE --dependency=afterany:$gen $B/score.sh
done

# 4. report
python $B/report.py --scores $STANDARD_BENCH_WORK/scores \
       --out $STANDARD_BENCH_WORK/report.json
```

`--array` ranges are 0-(N-1) for the track's target count: **1 / 4 / 5 / 41** for
unconditional / sm_binder / ppi_binder / enzyme under `targets-full.json` (enzyme is
`0-3` under the sampled `targets.json`). Both scripts print N and refuse an index past
the end, so a wrong range fails loudly instead of silently skipping targets. With no
`SLURM_ARRAY_TASK_ID` they run every target sequentially, which is what you want for the
one-target unconditional track or a run without slurm.

Host paths are environment overrides, defaulting to this cluster: `RFD3_VENV`,
`DESIGNABILITY_ASSETS`, `LIGANDMPNN_DIR`, `RFD3_TMPDIR`, and
`PROTEIN_DESIGN_BENCH_STANDARD` if a script cannot self-locate (sbatch copies the
submitted script into the job spool directory, so `BASH_SOURCE` does not find the
checkout; the scripts recover the path from `scontrol` first).

Two environment settings in the drivers are load-bearing and easy to lose if you rewrite
them: `TMPDIR` must be a **short** path, because dataloader workers bind an `AF_UNIX`
socket in it and a long one blows the 108-char limit; and `score.sh` runs from the work
dir and **not** `/tmp`, because a phantom `matplotlib` directory there makes torchmetrics
believe matplotlib is installed and then fail on `matplotlib.axes`.

Cost, measured 2026-09-19 and again 2026-09-23: **~110 s/design at 4 sequences**; every
design is folded, since nothing is gated. Generation is ~20 min per target. Scoring is
the long pole at **7-11 h per target** (~200 s/design x 100 on the slower targets).

Four operational lessons from the 51-target run, each of which cost something:

- **One array task per *target*, not per track.** 41 enzyme targets in one job exceeds
  any walltime. Select with `$SLURM_ARRAY_TASK_ID` and pass `json_keys_subset=[$TARGET]`
  - a bare `json_keys_subset=a,b` is rejected by hydra as an ambiguous value.
- **Give the scorer `-t 24:00:00`, not 12.** `score_standard.py` writes its single JSON
  only at the end, so a timeout loses the entire target - there is no per-design
  checkpoint. Three targets timed out at exactly 12 h. `scontrol update jobid=... 
  TimeLimit=` is **admin-only** on this cluster, so a running job cannot be rescued.
- **`--dependency=afterok` on an array needs ALL tasks to succeed.** One failed
  generation task leaves the scoring job unstartable and eventually killed as
  `DependencyNeverSatisfied`. For a rerun-after-timeout use **`afterany`**; `afterok`
  would never be satisfied.
- **Guard a rerun** with `if [ -s <score file> ]; then exit 0; fi` so targets that did
  finish are not re-scored. That guard saved a full GPU-target on the rescore pass.

Merging is a file copy - `report.py` globs `*.json` from one directory and there is one
file per target (`<track>-<target>.json`), so a partial rerun never needs re-scoring what
already landed.

## What decides a pass

**`success.txt` alone.** There is no pre-fold gate: every design is inverse-folded with
LigandMPNN/ProteinMPNN, refolded with RoseTTAFold3, and judged on the track's published
criterion. Nothing is rejected before folding.

| track | passes when |
| --- | --- |
| unconditional | backbone RMSD <= 1.5 A |
| sm_binder | backbone RMSD <= 1.5 A AND ligand RMSD <= 5 A AND interface PAE <= 1.5 AND ipTM >= 0.8 |
| ppi_binder | binder backbone RMSD <= 2.0 A AND interface min PAE <= 1.5 |
| enzyme | motif all-atom RMSD <= 1.5 A after motif-backbone alignment (any of 5 models, any of 4 sequences) **AND** no ligand clash |

4 sequences per backbone on every track; the design passes if ANY sequence (and for
enzyme, any of its 5 predictor models) satisfies the criterion.

**enzyme's ligand-clash check is the one criterion success.txt measures on the RAW
design** - "any non-motif backbone atom (N, CA or C) within 1.5 A of the closest ligand
atom" - so it is part of the verdict, not a filter. It uses the grader's own
`minimum_distance` and `LIGAND_CLASH_MIN`.

### The task's own extra criteria, and why they are no longer recorded

This harness used to compute the task's own additions per design and store them beside the
verdict - backbone continuity (CA-CA), self-clash under both rules, radius of gyration,
ligand burial, and motif RMSD on the raw design - as `gate_all_atom` and `gate_backbone`.
**success.txt contains none of them**, and as of 2026-09-26 neither does this harness:
`score_standard.py` records no gate columns and `report.py` prints and emits no
`passed_gate`, `gate_rate_*` or `gate_mode`. Its JSON key for the verdict is `criterion`.

They were removed because they were severe rather than cheap. Measured on the 161 ep200
enzyme designs, backbone + motif sidechains admitted 15 (9%) and the all-atom rule 0 (0%),
against success.txt's ligand-clash term at 70 (43%). The same comparison over the whole
5,100-design reference run is tabulated in `environment/evaluation/RESULTS.md`, which is
where the decision to delete them is argued.

The sm_binder case shows why the all-atom rule went first: `FAD-0` trips the all-atom
self-clash at 0.49 A yet **passes the real criterion** (sc_RMSD 1.148, ipTM 0.885, interface PAE
1.38, ligand RMSD 2.90).

Every verdict-making function is the grader's own, imported not copied:
`pipeline.DesignabilityScorer`, `geometry.minimum_distance`, and `collect.py`'s parsing.

## This is now the same quantity as an agent's own score

It was not, until 2026-09-26. An agent self-scoring with
`environment/evaluation/score.py` went through the submission path, which capped a target
at 32 designs (`common.py::MAX_DESIGNS_PER_TARGET`) and gated on geometry and conditioning
before folding, while this benchmark read raw RFD3 output at 100 designs/target and gated
on nothing. Both sides moved: the cap is **100**, matching the protocol, and the gate is
deleted, so the two paths apply the same criteria to the same number of designs.

One difference is left, and it is about the inputs rather than the verdict: this benchmark
builds its specs from upstream's published `input.yaml` over the evaluation view, while an
agent's designs come from `starter/rfd3_inputs.py` rebuilding conditioning out of a `.npz`
over `fit`/`tune`. Same measurement, different target sets - so the two can share a table
only if the column says which targets it is over.

## Known confounders, as of 2026-09-23

- **ppi_binder — FIXED, but it makes ppi a two-change comparison.** `make_specs.py` now
  sets `infer_ori_strategy: "hotspots"` (`make_specs.py:80`). Before that, neither our
  spec builders nor upstream's published `input.yaml` set it, so the origin fell back to
  the target's centre of mass and binders were generated INSIDE the target (measured: COM
  separation 5.72 A, 500/500 interpenetrating; after the fix, 23.70 A and 0/20).
  **Consequence:** any ppi comparison spanning that fix reflects *two* changes — the spec
  and whatever changed in training. The 0.004 vs 0.000 ppi move is not attributable to
  either on its own. To get a clean ppi delta, re-run the older checkpoint's ppi track
  with the fixed specs so the origin strategy is held constant on both sides.

- **enzyme — contamination is near-total.** All 41 enzyme targets were deposited
  1987-2014, i.e. every one falls inside the corpus's 2021-09-30 cut. Together with
  sm_binder FAD (7bkc) and SAM (7c7m) that is **43 of the 50 conditioned targets in the
  training corpus**; only the 5 ppi targets (excluded by upstream's own dataset filters)
  and sm_binder IAI/OQO (post-cut) are clean. Contamination does not appear to drive the
  scores — 28 of the 41 contaminated enzyme targets score 0.00 — but the enzyme headline
  is measured almost entirely on structures the model trained on. Target selection is
  open work.
- **enzyme**: `insertion_rmsd` is **2.66 A** against upstream's released 168M model at
  **0.70 A** (upstream's own metric, from its shipped tutorial outputs). Guideposts are
  inserted by OVERWRITING the diffused chain's coordinates
  (`trainer_utils.py:441`), so a seam is exactly the gap between where the model put the
  chain and where it put the motif. This is a capacity/compute gap, not a bug: every
  enzyme-side training switch (tipatom 5.0, `p_unindex_motif_tokens` 0.5,
  `alpha_unindexed_diffused` 1.0) matches upstream exactly. `insertion_rmsd` is a cheap
  continuous proxy for a size sweep - readable from a design's .json, no MPNN or RF3.
