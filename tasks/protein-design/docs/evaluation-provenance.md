# Evaluation provenance (maintainers)

Material that belongs to building and checking this task rather than to developing a
model in its workspace, so it is kept out of both images. `TASK_STANDARD.md` keeps
budgets, GPU counts, the verifier's schedule and operator detail out of what the agent
sees; the agent-facing numbers stay in `environment/evaluation/RESULTS.md`.

### Cross-check: the graded path reproduces this row

Everything above comes from the **operator** harness, which scores upstream's published
specs. `tests/main_verifier.py` → `tests/grader.py` is a **different code path** - it
rebuilds every request's conditioning from the task's own `.npz` contexts via
`environment/baseline/rfd3-small/rfd3_inputs.py`. The two are supposed to pose the same problem, and
until 2026-09-28 they did not; see "What the cross-check cost" below.

Graded pass, same `epoch-0100.ckpt`, 51 targets × 100 designs, 9 h 33 min on 8× L40S:

| track | operator (above) | graded | graded diversity |
| --- | ---: | ---: | ---: |
| sm_binder | 0.030 | **0.0275** | 0.750 |
| ppi_binder | 0.004 | **0.002** | 0.200 |
| enzyme | 0.054 | **0.0624** | 0.341 |
| *unconditional* | *0.410* | ***0.420*** | *1.000* |

**Every track agrees within sampling noise**, on counts of 11 vs 12, 1 vs 2, 256 vs 220
and 42 vs 41 passing designs. `conditioning_satisfied` is 100% on all four. This is the
replayable known answer: an integrator who builds the images and runs the verifier on this
checkpoint should land on the graded column, and a build that does not is misconfigured
rather than unlucky.

Two honest qualifications:

- **ppi_binder is measurable, not good.** 1 of 500 is not distinguishable from zero —
  P(0 of 500 | true rate 0.004) is ~13%. What the cross-check establishes is that the
  number is now *produced by the model* rather than by a spec bug, which was not true of
  any earlier graded run.
- **5 designs of 5,100 never reached the grader** (4 enzyme, 1 ppi_binder). Each carries a
  single `UNK` residue, which `rfd3_collect.py::design_chain` rejects, discarding the whole
  design. They are counted as **failed designs**, per `contract.md` — "per-sample failure is
  the measurement" — which shifts the pass rates by ≤ 0.1%. Discarding a sound backbone over
  one residue label nonetheless contradicts the contract's own statement that inverse folding
  replaces the sequence before anything is measured, so the collector is wrong here and is
  filed to be fixed.

Artifacts: `/net/tukwila/yanjing/verifier-ref-fixed/results/{reward,report}.json`
(slurm 27395799). `report.json` carries no per-target breakdown by design — the graded
view must not name evaluation targets.

### What the cross-check cost: five spec divergences

Running the two paths against each other is what found them, and **diffing the two spec
sets directly is what found them cheaply** — minutes against the days that re-running the
benchmark had already spent reproducing the disagreement rather than explaining it.

The first graded pass read enzyme **0.1037** against 0.054 and ppi_binder a flat
**0.0000**, on this same checkpoint. Neither was sampling:

1. **`ingest_targets.py` dropped the ligand half of `select_fixed_atoms`**, so the graded
   path pinned **1.90×** the ligand atoms upstream pins (1,517 against 800; 0 of 41 enzyme
   targets matched). The motif half of the same dict was honoured — the asymmetry was the
   bug. Contexts now carry `ligand_atom_names` and `ligand_fixed_mask`.
2. **Cross-ligand atom-name bleed** in that new mask: one `wanted` set unioned over every
   ligand, and PDB ligand atom names collide constantly (`C1'`, `PA`, `O2B` are
   nucleotide-wide). 16 of 41 enzyme targets affected. Now built per ligand.
3. **A non-standard amino acid used as a ligand vanished entirely** (2 targets). The
   branch was `if is_protein and name in ATOM14 / elif not is_protein`, and `DAL` / `ORN`
   match neither. Now `elif not residue.is_water()`.
4. **`infer_ori_strategy` was never set on ppi specs**, so `set_com` fell through to "COM
   of the fixed motif" — and on a ppi spec the fixed motif *is the whole target chain*.
   **Every binder was generated inside its target**, median minimum interatomic distance
   0.25–0.45 Å. Graded ppi 0/500 was never a measurement of the model.
5. **ppi hotspots were token-level** where `input.yaml` names exactly two sidechain atoms
   per hotspot, and **contiguous renumbering flattened real numbering gaps** on the one
   target that has them, compressing a hotspot by 43 residues.

The rule that settles which side is right, and the one to apply to any future divergence:
**`environment/data/targets/*/input.yaml` is the source of truth**, not `make_specs.py`.
The graded path's only licence to differ is residue **renumbering** — deliberate, so
evaluation target ids cannot leak through a spec — and every selection *value* must come
through verbatim. Auditing against `make_specs.py` was a proxy that happened to work.

After all five, the audit over every published key/value across all 50 conditioned targets
reports **0 problems**, and the remaining differences are documented and intentional.

### Designability is the only metric, 2026-09-26

**The row above is exactly what `tests/grader.py` reports** - not a looser cousin of it.
Both fold every design that could still satisfy its criterion, both apply the published
criteria, and neither adds a requirement the criteria do not contain. There is no gate
column any more because there is no gate: `score_standard.py` and `report.py` stopped
recording one, and `score.py`'s `--geometry-only` shortcut was removed with it.

Until 2026-09-26 the grader gated on geometry and conditioning *before* folding and only
folded the survivors. What that cost, measured on the same 5,100 designs:

| track | designs | old gate, all-atom | old gate, backbone + motif | **designability** |
| --- | ---: | ---: | ---: | ---: |
| unconditional | 100 | 0.020 | 0.590 | **0.410** |
| sm_binder | 400 | 0.013 | 0.583 | **0.030** |
| ppi_binder | 500 | 0.006 | 0.006 | **0.004** |
| enzyme | 4,100 | 0.001 | 0.056 | **0.054** |

The all-atom rule took a real trained checkpoint to **0 of 200 designs** on the graded
path, because it judged the generator's own free rotamers - which the reference protocol
discards before inverse folding and the training loss never penalises (`lp_weight: 0.0`).
Excluding those rotamers fixes the worst of it but is still not enough: intersected with
the criterion it admits 0.022 of enzyme designs against 0.054 designable, and **0.000** of
ppi against 0.004. A filter that discards 59% of the designable enzyme designs and every
designable ppi binder is not cost control; it is a second criterion, one no paper
publishes and no agent can read off `success.txt`.

What still runs before folding is only this, and neither is a criterion the task invented:

- **The enzyme ligand-clash term** - `success.txt` criterion 2, the only published term
  measured on the raw design rather than the refold. Checking it first skips a fold it
  would have wasted (52% of enzyme designs on the reference run) and cannot change a
  verdict, being a conjunct either way.
- **Request validation** - the motif map must name one distinct in-range residue per motif
  position carrying the requested identity, or motif recapitulation has nothing to measure
  against. Contract validation, not scoring.

Deleted outright: CA-CA continuity, self-clash, radius of gyration, motif RMSD on the raw
design, ligand burial, ppi target clash, ppi hotspot contact. Two consequences worth
stating plainly rather than burying:

- **Ligand burial's rationale was already obsolete.** It existed because "without ipTM and
  PAE nothing else forces a design to engage its ligand" - and RoseTTAFold3 enforces both.
- **Epitope specificity is no longer checked.** Hotspot contact is gone, and interface PAE
  proves only that *an* interface exists, not that it is the requested one. RFdiffusion3's
  ppi criterion has no hotspot term either, so the task now inherits upstream's looseness
  instead of inventing a stricter rule of its own.

ppi_binder's 0.004 is unaffected by any of this: its gate rates were 0.006 under *both*
rules, so those designs failed with sidechains already removed - the
binders-inside-the-target problem noted above, not a rotamer artifact.

### Reproducing it

All four stages are in `environment/evaluation/benchmark/standard/`. The complete sequence that produced the
row above:

    export STANDARD_BENCH_WORK=<a FRESH work dir>
    B=<checkout>/protein-design/environment/evaluation/benchmark/standard

    # 1. specs - all 51 targets
    python $B/make_specs.py --out $STANDARD_BENCH_WORK/inputs \
           --definition $B/targets-full.json

    # 2 + 3. generate, then score - ONE ARRAY TASK PER TARGET, per track
    mkdir -p $STANDARD_BENCH_WORK/logs && cd $STANDARD_BENCH_WORK/logs
    for s in unconditional:0-0 sm_binder:0-3 ppi_binder:0-4 enzyme:0-40; do
      TRACK=${s%%:*}; RANGE=${s##*:}
      gen=$(TRACK=$TRACK CKPT=<ckpt> sbatch --parsable --array=$RANGE $B/generate.sh)
      TRACK=$TRACK sbatch --array=$RANGE --dependency=afterany:$gen $B/score.sh
    done

    # 4. report over every score file
    python $B/report.py --scores $STANDARD_BENCH_WORK/scores \
           --out $STANDARD_BENCH_WORK/report.json

Host paths are overridable: `RFD3_VENV`, `DESIGNABILITY_ASSETS`, `LIGANDMPNN_DIR`,
`RFD3_TMPDIR`, and `PROTEIN_DESIGN_BENCH_STANDARD` if the scripts cannot self-locate.

Four things this run cost. The first three are now **encoded in the scripts**, not left
to the operator, but they explain why those scripts look the way they do:

- **One array task per *target*, not per track.** 41 enzyme targets in one job exceeds any
  walltime. The target is selected by `$SLURM_ARRAY_TASK_ID` and passed as
  `json_keys_subset=[$TARGET]` — a bare value is rejected by hydra as ambiguous. An index
  past the end is refused, so a wrong `--array` fails loudly rather than skipping targets.
- **The scorer gets 24 h, not 12.** Scoring runs **7–11 h per target** at ~200 s/design ×
  100, and `score_standard.py` writes its single JSON only at the end, so a timeout loses
  the whole target — there is no per-design checkpoint. Three targets timed out at exactly
  12 h here. `scontrol update TimeLimit=` is admin-only on this cluster, so a running job
  cannot be rescued, only resubmitted.
- **`--dependency=afterany`, never `afterok`.** `afterok` on an array needs *all* tasks to
  succeed, so one failed generation leaves scoring unstartable and eventually killed as
  `DependencyNeverSatisfied`; and for a rerun after a timeout it can never be satisfied at
  all. `score.sh` carries an `if [ -s <score file> ]` guard, which is what makes `afterany`
  safe and lets the same script serve as both the first pass and the rerun — it saved a
  full GPU-target here.
- **A fresh work dir every run.** `generate.sh` passes `skip_existing=True`, so reusing a
  populated `designs/` silently skips generation and scores the previous checkpoint under
  a new label. This one is *not* enforced — it cannot be, since resuming a partial run is
  the same operation as corrupting a new one.

`targets.json` holds the frozen 14-target subset used by the earlier reference row and is
deliberately **left unchanged** so that row stays reproducible; `targets-full.json` is the
complete 51-target definition used here.

Artifacts: `/net/tukwila/yanjing/standard-bench-enzyme-full/report-full.json` (51 targets),
`/net/tukwila/yanjing/standard-bench-ep100/report-ep100.json` (the frozen 14).

### The earlier 14-target row, for continuity

Before the full enzyme track was run, this section reported the same checkpoint and the
same protocol with enzyme **sampled down to 4 of 41** by `random.Random(20260919)`. Only
the enzyme track differed: it read **0.048** there against **0.054** here.

**Those are different target sets, not a delta.** Nothing improved between them; 37
enzyme targets were added. The sampled four averaged 0.0475 against the other 37's
0.0543, so the frozen draw was mildly pessimistic — close enough that the four-target
sample was not badly misleading about the *mean*, and completely misleading about the
*spread*, which is the part that matters.
