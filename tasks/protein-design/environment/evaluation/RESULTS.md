# Published criteria and results

This page holds two kinds of number, and they must not be mixed.

**"Measured by this task"** is the one section produced by this task's own grader, on
this task's own targets. It is the reference row for what the starter achieves.

Everything under **"Reported results"** is a **claim from the cited source**. None of
those systems has been run through this task's grader, so none of those numbers is
directly comparable to a reward produced here. They are recorded so an agent knows what
the field currently achieves and by what criteria.

## Success criteria this task follows

Taken from RFdiffusion3's benchmark definitions, which are also the definitions used by
this task's evaluation targets.

### enzyme

> 100 backbones, 8 sequences per backbone
> 1. Chai-1 motif sidechain recapitulation: the backbone passes if any of 5 Chai models
>    (one trunk seed) for any of the 8 LigandMPNN sequences has a motif backbone
>    (N, Cα, C) aligned, motif all-atom RMSD ≤ 1.5 Å.
> 2. No ligand clashes: any non-motif backbone atom (N, CA or C) within ≤ 1.5 Å of the
>    closest ligand atom.

### sm_binder

> 400 backbones, 8 sequences per backbone
> backbone RMSD ≤ 1.5 Å & backbone-aligned ligand RMSD ≤ 5 Å & interface min PAE ≤ 1.5
> & ipTM ≥ 0.8

### ppi_binder

> 400 backbones, 4 sequences per backbone
> 1. evaluate min inter-chain PAE
> 2. designability: RMSD

**What this task enforces: these criteria, and nothing besides them.** The motif
recapitulation, backbone and ligand RMSD, **interface PAE and ipTM** terms are enforced by
RoseTTAFold3, with sequences from LigandMPNN (or ProteinMPNN where there is no ligand);
the enzyme ligand-clash term is measured on the raw design, as `success.txt` specifies. The
task added gates of its own until 2026-09-26 — geometry, ligand burial, hotspot contact —
and they are gone; see "The pre-fold gate was removed" below for what they cost.

Two deliberate deviations from the criteria quoted above:

- **4 sequences per backbone on every track**, not 8 for enzyme and sm_binder. Uniform
  and affordable; since the criterion is "pass if ANY sequence passes", this can only
  LOWER a pass rate, so results here are conservative against the published ones.
- **RoseTTAFold3 is the predictor**, where RFdiffusion3 used Chai-1 for the enzyme
  criterion and AlphaFold for the rest. RF3's parameters are public and ship in the
  image, which is what lets an agent optimise the criterion it is graded on; AF3's
  cannot be redistributed. The thresholds were calibrated with AlphaFold, so **numbers
  here are internally comparable but not comparable with the RFdiffusion3 paper.**

A third deviation was removed rather than kept: the **ligand-burial gate**, which existed
because "without ipTM and PAE nothing else forces a design to engage its ligand". Those
terms are enforced now, so its own rationale no longer held.

## Measured by this task

The starter's performance under this task's own grader, on this task's own targets.
Produced by `environment/evaluation/benchmark/standard/` (operator-side; excluded from both images).

### What was measured

The starter is upstream RFdiffusion3, vendored at `environment/baseline/rfd3-small/rfd3/`, reduced
in layer widths only to **12,445,554 parameters** (upstream `pretrain` is 168,038,994).
The task ships **no checkpoint** — `instruction.md` gives components, not finished
generators — so this row is a checkpoint trained with the starter's own code and recipe,
not a shipped artifact.

| | |
| --- | --- |
| checkpoint | `epoch-0100.ckpt`, job 25880709 |
| training | cold start, 2026-09-20 10:07 → 09-22 10:07 (48 h wall, 8× L40S), ended on walltime at ~epoch 108 |
| examples seen | ~264k, i.e. **~4.5% of one pass** over the 5,872,063-example corpus |
| recipe | upstream `pretrain_small` + `ppi.frequency: 0.0 → 5.0`; every other condition frequency identical to upstream |
| protocol | **every target the task defines** — 51 targets × 100 designs × 4 sequences = 5,100 designs, all folded, nothing gated pre-fold |
| verdict | `success.txt` alone (see criteria above) |

This row covers the **complete** target set, not a sample: all 41 enzyme targets, all 5
ppi, all 4 sm_binder, and the 1 synthetic unconditional target.

### Result — pass rate per track

| track | targets | designs | pass rate |
| --- | ---: | ---: | ---: |
| sm_binder | 4 | 400 | **0.030** |
| ppi_binder | 5 | 500 | **0.004** |
| enzyme | 41 | 4,100 | **0.054** |
| *unconditional* | *1* | *100* | *0.410* |

A track's pass rate is the **mean of its per-target rates**, not pooled over designs.
A target's rate is passing designs / 100 **requested** — designs lost to a parse error
count as failures, which is the conservative convention.

**No composite number is quoted here, and the grader no longer computes one either.**
The three tracks above the line are RFdiffusion3's benchmark, each judged by its own
published `success.txt`; averaging them is not a choice this task makes.
*unconditional* is italicised because it is **not part of the benchmark** — there is no
published unconditional criterion to compare a number against. The `uncond_L100` row is
a synthetic length invented by this task, so mixing it into a mean with the other three
would inflate the result with the one track nobody publishes. It is nonetheless a
**graded** track: since 2026-09-28 `ingest_targets.py` synthesises the same target into
the evaluation view (51 targets: 41 enzyme, 4 sm_binder, 5 ppi_binder, 1 unconditional),
so this row and the graded one measure the same thing.

`tests/grader.py` reports the same quantity - per-track `pass_rate`, with `diversity`
beside it, and no composite - into `reward.json`, at the same sample size: the graded
evaluation view requests 100 designs per target too, as of 2026-09-26. Both definitions
agree exactly, not merely in spirit: a mean over per-target rates and a pooled count over
requested designs are the same number when every target requests the same 100.

The enzyme distribution, over all 41 targets — 220 passing designs of 4,100:

| per-target pass rate | targets |
| --- | ---: |
| 0.00 | **28** |
| 0.01 – 0.09 | 6 |
| 0.10 – 0.19 | 3 |
| 0.20 – 0.49 | 0 |
| 0.50 – 0.68 | **2** |

Target ids are deliberately **not** listed here. This file ships to the agent image at
`/workspace/references`, and naming which evaluation targets score well would hand an
agent a shortlist to overfit — 43 of the 50 conditioned targets are in the pre-2021
corpus an agent can assemble, so the named ones would be retrievable. The operator's
per-target breakdown is in the run's own `report-full.json`.

### How to read it

- **The enzyme track is a wide spread, not a uniform low rate.** 28 of 41 targets score
  exactly 0.00 while two exceed 0.55. Roughly a third of targets produce designs that
  survive the `success.txt` ligand-clash criterion; the rest lose essentially every
  design to non-motif backbone threading through the ligand. **A track mean of 0.054 is
  not the rate you should expect on an arbitrary enzyme target** — the per-target
  distribution is the finding, and it is bimodal.
- **ppi remains carried by a single target**, at 2/100, with the other four at zero.
  Which one is deliberately not named, for the reason given above; the operator's
  `report-full.json` has it.
- **ppi is non-zero for the first time, and that is not yet a result.** 2/500 against a
  previous 0/500 is Fisher one-sided **p = 0.25**. It also reflects **two** changes at
  once — PPI conditioning enabled in training, and `infer_ori_strategy: "hotspots"` added
  to the ppi specs, without which binders are generated inside the target. The two are
  not separated by this run.
- **The checkpoint is early.** ~4.5% of one data pass, stopped by walltime rather than
  convergence. This is a floor for the architecture, not its ceiling.
- **Benchmark contamination is extensive and unfixed.** **43 of the 50 conditioned
  targets are in the training corpus**: all 41 enzyme targets (deposited 1987–2014, every
  one before the corpus's 2021-09-30 cut) plus sm_binder FAD (7bkc) and SAM (7c7m). Only
  7 are clean — the 5 ppi targets, excluded by upstream's own dataset filters, and
  sm_binder IAI (5sdv) and OQO (7v11), both deposited post-cut. Verified against the
  public atomworks metadata, not assumed.
  Contamination does **not** explain the scores: 28 of the 41 contaminated enzyme targets
  score 0.00, and the two contaminated sm_binder targets total 4 passes against the clean
  pair's 8. But the enzyme track's headline number is measured almost entirely on
  structures the model trained on, and **target selection is open work.**
- **Not comparable to the RFdiffusion3 paper**, for the two reasons given above: RF3 as
  predictor in place of Chai-1/AlphaFold, and 4 sequences per backbone instead of 8.

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

## Reported results

### RFdiffusion3
[bioRxiv 2025.09.18.676967](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)

- All-atom diffusion over a 14-atom-per-residue representation, with atom-level
  conditioning for enzyme design and other constrained problems.
- Reported ~10x faster inference than RFdiffusion2 at improved in-silico benchmark
  performance.
- Experimental: a DNA-binder screen of 5 designs yielded 1 with EC50 5.89 ± 2.15 µM.
  An enzyme screen of 190 cysteine hydrolase designs yielded 35 multi-turnover
  catalysts, best kcat/Km 3557 M⁻¹s⁻¹.
- Trained on the PDB with `deposition_date < 2024-01-01`, which is the cutoff this
  task's development split uses.

### BoltzGen
[bioRxiv 2025.11.20.689494](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1)

- All-atom generative model for designing proteins and peptides against a wide range
  of biomolecular targets.
- Weights, data, training code and inference code released under MIT.

### DISCO
[General Multimodal Protein Design](https://github.com/DISCO-design/DISCO)

- Generates sequence and 3D structure jointly rather than backbone-then-inverse-folding,
  using cross-modal recycling to keep the two consistent.
- Conditions on small-molecule ligands, DNA and RNA, and multi-entity complexes.
- Reported best in-silico performance on 178/179 evaluated ligands, plus DNA and RNA.
- Applied to catalysis conditioned solely on reactive intermediates, without
  pre-specifying catalytic residues or template scaffolds.
- Does **not** currently support multi-chain protein complexes, so it does not cover
  this task's `ppi_binder` track directly.

### BoltzDesign1
[bioRxiv 2025.04.06.647261](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1)

- Inverts a pretrained Boltz-1 with **no model finetuning**, using only the Pairformer
  and confidence modules and optimising the distogram rather than sampled structures.
- Redesigns sequences with ProteinMPNN / LigandMPNN and validates with AlphaFold3 on
  iPTM and pLDDT.
- The authors describe it as experimental software that has not been experimentally
  validated.

### Proteina-Complexa
[NVIDIA-BioNeMo/Proteina-Complexa](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa)

- Fully atomistic binder and enzyme generation combining partially-latent flow matching
  with inference-time optimisation and search.
- Pretrained on curated structures from over 1M experimental and predicted structures,
  including a synthetic binder-target set built from domain-domain interactions.
- Reported experimental validation including multiplexed phage display and kinetic
  measurements.
- Multi-license: code Apache-2.0, datasets CC BY 4.0, weights under the NVIDIA Open
  Model License, which permits commercial use and derivative models. Vendored at
  `/opt/references/proteina_complexa`. Do not confuse it with the original Proteina
  repository, which *is* research / non-commercial only.

## Reproduction status

No **published** entry above has been reproduced here. The starter is still the only
system this task has measured, and that measurement is now in "Measured by this task" —
which supersedes this section's former claim that the starter "is expected to score near
zero on designability". It does not: on the full target set every track is non-zero —
enzyme **0.054**, sm_binder **0.030**, ppi_binder **0.004** — with two individual enzyme
targets above 0.55. The near-zero expectation came from geometry-only scoring under the
all-atom clash rule, which measured rotamer quality rather than designability; both rules
are tabulated above.

Still outstanding, and still the honest way to calibrate a score: a **null system** row,
and **one published method** run under identical conditions. Without the null row it is
not established how much of these pass rates a trivial generator would collect — and with
28 of 41 enzyme targets at exactly 0.00, the null row matters more on this track than any
average suggests.
