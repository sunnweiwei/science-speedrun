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
Produced outside this workspace; the benchmark code is in neither image.

### What was measured

The starter is upstream RFdiffusion3, vendored at `environment/baseline/rfd3-small/rfd3/`, reduced
in layer widths only to **12,445,554 parameters** (upstream `pretrain` is 168,038,994).
The task ships **no checkpoint** — `instruction.md` gives components, not finished
generators — so this row is a checkpoint trained with the starter's own code and recipe,
not a shipped artifact.

| | |
| --- | --- |
| checkpoint | `epoch-0100.ckpt` |
| training | cold start with the starter's own recipe, stopped early at ~epoch 108 |
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
- **The checkpoint is early.** ~4.5% of one data pass, stopped early rather than at
  convergence. This is a floor for the architecture, not its ceiling.
- **Benchmark contamination is extensive.** **43 of the 50 conditioned
  targets are in the training corpus**: all 41 enzyme targets (deposited 1987–2014, every
  one before the corpus's 2021-09-30 cut) plus sm_binder FAD (7bkc) and SAM (7c7m). Only
  7 are clean — the 5 ppi targets, excluded by upstream's own dataset filters, and
  sm_binder IAI (5sdv) and OQO (7v11), both deposited post-cut. Verified against the
  public atomworks metadata, not assumed.
  Contamination does **not** explain the scores: 28 of the 41 contaminated enzyme targets
  score 0.00, and the two contaminated sm_binder targets total 4 passes against the clean
  pair's 8. But the enzyme track's headline number is measured almost entirely on
  structures the model trained on.
- **Not comparable to the RFdiffusion3 paper**, for the two reasons given above: RF3 as
  predictor in place of Chai-1/AlphaFold, and 4 sequences per backbone instead of 8.

Provenance of this row, the reconciliation with the graded path, the frozen-sample
history and the recipe that reproduces it are in `docs/evaluation-provenance.md`,
which is kept out of both images.

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
