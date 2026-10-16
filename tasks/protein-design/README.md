# Protein design: one all-atom design system, four tracks

## 1. Overview

An agent develops **one conditional all-atom protein design system** and is scored on
**designability**: whether the structures it invents are ones a sequence actually folds
into. Grading runs the benchmark's published pipeline - pinned LigandMPNN designs
sequences per structure, pinned RoseTTAFold3 refolds the complex, and each track applies
its own success criterion from the benchmark's `success.txt`.

| Track | Conditioning | Analogue |
| --- | --- | --- |
| `unconditional` | length only | RFdiffusion3 monomer generation |
| `sm_binder` | a bare ligand, no pocket template | RFdiffusion3 / RFdiffusionAA |
| `ppi_binder` | target chain + hotspot residues | RFdiffusion3 binder design, BoltzDesign1 |
| `enzyme` | catalytic residue constellation | RFdiffusion3 motif scaffolding |

Read [instruction.md](instruction.md) for the rules and
[contract.md](environment/contract.md) for delivery and scoring.

### Why the task is shaped this way

**The artifact is a design system, not necessarily a trained model.** The two current
reference points sit in different paradigms: [RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)
trains an all-atom diffusion model, while [BoltzDesign1](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1)
trains nothing at all - it inverts a pretrained Boltz-1 at inference time. A
from-scratch parameter band would have silently excluded the second. `harness.json`
declares which paradigm a submission is (`trained_generator`, `predictor_inversion`, or
`hybrid`) and the verifier enforces that the declaration matches reality. This uses
CONTRIBUTING's existing allowance for a harness as the graded artifact.

**All-atom, because catalytic geometry lives in sidechains.** Structures use the
atom14 layout RFdiffusion3 uses: up to 14 heavy atoms per residue, backbone in slots
0-3. Motif scoring compares sidechain atoms and requires the correct residue
identities, so scaffolding a Ser-His-Asp triad means placing that triad, not three
residues in roughly the right place.

**Designability is intrinsic, so there are no hidden labels.** Nothing is withheld and
development scoring is unrestricted. The verifier's two-container split exists so an
untrusted system cannot tamper with the instrument measuring it, not to hide answers.
That is a different boundary from the clinical tasks here, and it is documented rather
than assumed.

**The mechanical part is satisfiable; the metric is not.** Correct bond lengths,
threading a catalytic motif and producing a parsable all-atom structure are solved
problems, and the starter clears them essentially always. Refolding is not: the starter's
measured pass rates are a few percent on the conditioned tracks. The headroom sits in the
scientific question rather than in format compliance, which is why none of that
mechanical work is scored any more.

The graded metric is the published pass rate and nothing else. Until 2026-09-26 the
grader also gated on geometry and conditioning before folding, and that gate admitted 0.1%
of enzyme designs where 5.4% were designable - it took a real trained checkpoint to 0 of
the 200 designs the graded view held at the time, because it judged sidechain rotamers the
reference protocol discards before inverse folding. It was removed rather than corrected:
even a rotamer-blind version still discarded 59% of designable enzyme designs. All that
still runs before folding is `success.txt`'s own enzyme ligand-clash term, a conjunct of
the published criterion rather than an addition to it, and the motif-map contract check
that motif recapitulation needs in order to have anything to measure against. What the
gate cost is tabulated in [RESULTS.md](environment/evaluation/RESULTS.md).

**Degenerate systems cannot win.** Extended or collapsed chains do not refold, so
designability rejects them without a geometry rule of its own, and three of the four
tracks carry published criteria - motif recapitulation, ligand RMSD, ipTM and interface
PAE - that a fixed output cannot satisfy. The fourth, `unconditional`, is where a single
designable fold emitted repeatedly would otherwise look perfect, so every track reports
**diversity** beside its pass rate - the fraction of passing designs in distinct 2.0 A
clusters, with rotated duplicates not counting as distinct because clustering
superimposes first. Diversity is reported rather than blended into the pass rate, so mode
collapse is visible as itself instead of being averaged away.

#### Known limitation: evaluation targets are contamination-exposed

The evaluation targets are RFdiffusion3's benchmark, which is built from old PDB
entries (`1nzy`, `13pk`, `1dbt`). They are in the training set of every current design
model. This is inherent to the benchmark and is **not** solved here - RFdiffusion3
accepts the same exposure, since scaffolding a known catalytic site is the task. The
gain is comparability to published numbers.

The temporal holdout in `materialize.py` therefore protects only the development split:
`deposition_date < 2024-01-01`, RFdiffusion3's published training cutoff, with
homologous clusters dropped from fit and tune. An earlier BioLiP2-based pipeline could
not support even that - its catalytic annotations stop in 2021, leaving zero enzyme
targets after that date - which is why conditioning is now derived from coordinates on
a current PDB snapshot instead.

#### Known limitation: the metric is optimisable

Permitting pretrained predictors means a system can optimise directly against a
structure predictor, which is what BoltzDesign does by design. In-silico designability
measured by LigandMPNN plus RoseTTAFold3 is a proxy, and a system tuned against
correlated predictors can inflate it without producing better proteins. The task
mitigates but does not solve this: the geometry and conditioning gates are
predictor-independent, and the verifier records when a submission carries the grading
checkpoints. **This got harder, not easier, when the scorer became redistributable** -
RoseTTAFold3's parameters ship inside the agent's own image, so inverting the instrument
no longer requires the agent to supply it. The declaration in `harness.json` is the only
signal; nothing scans bytes. Wet-lab validation
is the only real answer and is out of scope here.

### Status

This is a task definition with executable preparation, training, inference, and
grading interfaces. It is **not** a trained solution, and it is not yet fully
validated. See [Validation](#validation-2026-09-03) for exactly what was and was not run.

## 2. Data

Nothing is redistributed. Two halves - see [data/README.md](environment/data/README.md):

- **Evaluation** is RFdiffusion3's own benchmark, **vendored in this repository** at
  `environment/data/targets/` (1.2 MB) and converted by `ingest_targets.py`.
  51 targets: 41 `enzyme`, 4 `sm_binder`, 5 `ppi_binder` and the synthesised
  `uncond_L100`, each requesting **20 designs** (RFdiffusion3's protocol requests 100), so
  1,020 designs in all. No external input needed.
- **Development** (`fit`, `tune`) is built from a dated RCSB PDB snapshot with all
  conditioning derived from coordinates, following RFdiffusion3's dataset
  construction. AlphaFold DB is deliberately not used.

```bash
export TASK_DATA_DIR=/secure/protein-design

bash environment/data/materialize.sh /secure/pinned-sources "$TASK_DATA_DIR"
```

To run the task, the operator sets the three standard variables (`TASK_STANDARD.md`). The
development container reads `development/` from `TASK_DATA_DIR`; the verifier reads
`evaluation/` from `TASK_PRIVATE_DIR`, which is the same materialized root:

```bash
export TASK_DATA_DIR=/secure/protein-design TASK_PRIVATE_DIR=/secure/protein-design
export TASK_RESEARCH_DIR=/large-disk/runs/protein-design-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p protein-design -a <agent> -m <model> --allow-agent-host <model API host>
```

The development half also stages an **RFdiffusion3 training corpus** - the assemblies
themselves plus two index parquets - because upstream trains from structures, not from
the per-track design requests. `materialize.py --corpus-targets` writes the structures and
`build_training_index.py` indexes them; the schema, the cluster conventions and the
offline gotchas are in
[starter/TRAINING_DATA.md](environment/baseline/rfd3-small/TRAINING_DATA.md), whose §0 also documents
the networked alternative - upstream publishes the real metadata tables, and
`atomworks setup metadata` downloads them.

Requires `gemmi` and `pyyaml` on the host. A default full run downloads **~3.8 GiB** -
76 MiB of bulk indexes plus ~3.7 GiB of assemblies - in **roughly an hour**, serially,
latency-bound. `--corpus-targets` scales that linearly; see
[data/README.md](environment/data/README.md#download-size-and-time).

**The scoring models need no operator mount.** LigandMPNN (MIT, the original
`dauparas/LigandMPNN` repository) and RoseTTAFold3 (BSD-3, IPD/UW) are both built into
the images, and **both sets of parameters are public downloads pinned by sha256**,
fetched at build time by `environment/fetch_designability_weights.sh`. Run that script
to reproduce the same `/assets` tree on any cluster.

This replaces AlphaFold 3, whose parameters must be received directly from Google under
terms that forbid sharing them. That restriction kept the real metric out of the agent's
image entirely, leaving a geometry-only upper bound as the only score an agent could
compute for itself; that shortcut was deleted on 2026-09-26, since the real metric now
runs wherever the agent does. The trade is that the per-track thresholds come from
RFdiffusion3's `success.txt`, calibrated with AlphaFold, so scores are internally
comparable but no longer directly comparable with the published table.

## 3. Workspace

```text
/workspace/
  instruction.md, contract.md
  data/README.md, data/fit/, data/tune/     read-only mounts of development/; /data links here
  baseline/rfd3-small/   upstream RFdiffusion3 reduced in layer widths: README, RESULTS, publish.py
  references/            pinned reference systems, published results, the weights boundary
  evaluation/            score.py and the scoring library (the verifier's own code)
  submission/            empty at start
  research/              empty, writable, large disk
  /assets/               LigandMPNN and RoseTTAFold3 parameters, baked into the image (2.9 GB)
  /pretrained/           declared pretrained assets, same path here and in verification
```

| baseline | pass rate: enzyme / sm_binder / ppi_binder / unconditional | cost |
|---|---|---|
| `rfd3-small` (RFdiffusion3, 12.4M params, `epoch-0100.ckpt`) | 0.0624 / 0.0275 / 0.002 / 0.420 | graded pass 9 h 33 min on 8× L40S |
| trivial generator (extended backbone, reduced view) | 0.000 / 0.000 / 0.000 / 0.000 | minutes |

Per-baseline detail: [`environment/baseline/rfd3-small/RESULTS.md`](environment/baseline/rfd3-small/RESULTS.md).

## 4. Evaluation

The main verifier validates `harness.json`, runs `infer.sh` unprivileged and
networkless, then seals the design index and structures into a hash-manifested
read-only snapshot. The trusted grader scores that snapshot and never imports the
submission. Per-track pass rates and diversity land in `/logs/verifier/reward.json`.
There is **no composite score**: the four tracks apply different published criteria and
`unconditional` is synthetic to this task, so they are reported and compared
individually. See [the verifier README](tests/README.md).

Sizing: 51 evaluation targets x **20 designs** = **1,020 designs** at 80-180 residues
(RFdiffusion3's published protocol is 100 per target; the pass rate is a per-target fraction, so 20
estimates the same quantity at a fifth of the grading cost). Generation is the cheap half - the operator
harness measured ~20 min per target at 100 designs. The table below is the 100-design sizing; every row
scales by 1/5 (~8,600 folds). Refolding dominates, and it is
dominated in turn by `enzyme`, which is 41 of the 51 targets and the only track folding 5
predictor models per sequence (`common.py:38`); every track inverse-folds 4 sequences per
backbone:

| track | designs | folds/design | folds |
| --- | ---: | ---: | ---: |
| unconditional | 100 | 4 | 400 |
| sm_binder | 400 | 4 | 1,600 |
| ppi_binder | 500 | 4 | 2,000 |
| **enzyme** | **4,100** | **20** | **~39,000 of a nominal 82,000** |
| | **5,100** | | **~43,000** |

`enzyme` is the only row where folds are not just designs x folds/design: `success.txt`'s
ligand-clash term is checked on the raw design first, and on the reference run 52% of
enzyme designs fail it and are never folded. Even so ~91% of grading cost is that one
track.

Budget accordingly, and grade in parallel. Folding was measured at **8.9 s/fold on an
RTX A4000** - ~106 h for a full pass on one such card. Designs are independent and every
fold is already its own subprocess, so `grader.py` folds one design per visible GPU
(`--workers`, defaulting to all of them) and `task.toml` asks the verifier for **8**. That
is ~13 h across 8 A4000-class cards; the H100 it declares has **not** been calibrated, so
any speedup beyond that is an estimate rather than a measurement. `[verifier]
timeout_sec` is **172,800 s (48 h)** to keep a wide margin over an uncalibrated figure,
with `main_verifier.py --grader-timeout` and `grader.py --wait-seconds` at 129,600 s so a
correct grading run is not killed for being slow before the verifier's own budget expires.
Inference keeps 43,200 s. Worker count cannot change a score - the verdicts are replayed
in submission order, because `cluster_designs` is order-sensitive and `diversity` would
otherwise drift with scheduling.

`environment/evaluation/benchmark/` is an operator-side harness for scoring a trained
upstream-RFD3 checkpoint on the same tracks using upstream's **published** benchmark
specs, so a number can be compared with RFdiffusion3's own. It sits under `evaluation/`
with the rest of the scoring code, but is **excluded from both images** by
`.dockerignore` - its `targets*.json` enumerate the evaluation target ids, and the agent
sees only `fit` and `tune`. The agent-facing equivalent is `baseline/rfd3-small/rfd3_infer.sh`,
which rebuilds each request's conditioning from its `.npz` context and therefore works
on `fit` and `tune`. See
[environment/evaluation/benchmark/README.md](environment/evaluation/benchmark/README.md)
for why the two cannot be one script.

## 5. Running

```bash
export TASK_DATA_DIR=/secure/protein-design TASK_PRIVATE_DIR=/secure/protein-design   # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/protein-design-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p protein-design -a <agent> -m <model> --allow-agent-host <model API host>
```

**Resources (external budget, in `task.toml`)**: development 8× H100, 64 CPUs, 512 GB RAM, 48-hour
agent timeout; verifier 8 GPUs, 48-hour limit. The measured graded pass of the old 100-design view
was 9 h 33 min on 8× L40S, dominated by RoseTTAFold3 reloading a 2.8 GB checkpoint once per design
rather than by folding. The 20-design view (1,020 designs, novelty gate included) took 6 h 51 min
end to end on 4× L40S (inference, refolding and Foldseek), about 1.4 h on 8× H100 by the usual
L40S-to-H100 ratio; the 48-hour request keeps wide headroom.

Unit tests (standard library plus numpy): `bash protein-design/tests/run_unit_tests.sh`. To
replay a package through the isolated verifier outside Harbor, see
[`tests/README.md`](tests/README.md).

## 6. Results and validation

Verified by running it:

- **Metric and contract**: 23 unit tests pass deterministically across repeated runs,
  covering the input contract, tamper detection, request validation, motif
  residue-identity enforcement, the published enzyme ligand-clash term, the harness
  manifest across all three paradigms, the failure policy, path escapes, mode collapse,
  and the trivial-generator check. *(Reworded 2026-09-26: the gate this bullet used to
  call out is deleted, and the conditioning criteria it covered are now asserted through
  the stubbed designability scorer, where the published criteria live.)*
- **atom14 table**: validated against six real PDB entries - 9,938 of 9,948 heavy atoms
  map into the layout across all 20 residue types, with no residue exceeding 14 slots.
  The only unmapped atoms are `OXT`, the C-terminal oxygen, which AlphaFold's atom14
  also omits.
- **Geometry gate calibrated on real proteins** *(history: the gate was deleted
  2026-09-26, and `geometry.validate_geometry` and its thresholds were removed with it
  rather than left advertising limits nothing enforces. Kept because the calibration is
  what established that the thresholds were the problem, not the implementation.)*: 1ubq,
  2trx, 4lzt, 3eiy and 1a3n pass; 1lyz is correctly rejected
  for a 4.53 A crystallographic chain break. An earlier version wrongly rejected 2trx and
  3eiy because their **cis-prolines** put consecutive CA atoms 2.89 A apart, so the gate
  accepted trans (3.80 A) and cis (2.90 A).
- **Geometry core**: Kabsch recovers an exact rigid motion to 2e-15 and refuses to fit
  a reflection. All-atom PDB round-trips exactly, preserving mask and residue types.
  Symmetric motif RMSD returns 0.0000 A under ASP/GLU/PHE relabeling instead of a
  spurious penalty. These are all still live - the refold criteria are measured with
  them. The clash-gate check in the original bullet went with the gate; what remains of
  `minimum_distance` in the grader is `success.txt`'s enzyme ligand-clash term.
- **Starter** *(rewritten 2026-09-24; the original bullet described a hand-written
  reimplementation that has since been deleted)*: the starter is now **upstream
  RFdiffusion3 itself**, vendored at `environment/baseline/rfd3-small/rfd3` from
  [RosettaCommons/foundry](https://github.com/RosettaCommons/foundry) `b02eed6a` and
  reduced in **layer widths only**. Measured by composing both configs and instantiating
  the network: **12,445,554** parameters against upstream `pretrain`'s **168,038,994**,
  i.e. 7.4%. Every depth is upstream's, untouched - 18 diffusion transformer blocks,
  2 + 2 pairformer, 3 + 3 atom encoder/decoder, `n_recycle` 2 - as are the crop, the
  losses, the AF-3 optimizer/schedule and EMA. The eight width leaves that change are
  `c_s` 384->128, `c_z` 128->32, `c_atom` 128->64, `c_token` 768->192, `c_t_embed`
  256->128 and up/downcast `c_model` 128->64; `c_atompair` stays 16 deliberately, being
  the pair channel that carries reference-conformer geometry. For what a trained
  checkpoint actually scores, see `environment/evaluation/RESULTS.md` - those are real
  LigandMPNN + RoseTTAFold3 designability numbers, not stubs.
- **End to end**: `infer.sh` runs on all 50 real benchmark targets and every design
  parses. Two real data bugs surfaced this way: one target's ligand carries four atoms
  with no coordinates, and one constrains only two motif atoms, which is fewer than a
  rigid superposition needs. Both are handled.
- **Data pipeline** *(rewritten 2026-09-26; the original bullet described a BioLiP2 /
  M-CSA routing step that `materialize.py` no longer performs - see
  [data/README.md](environment/data/README.md), "Curated annotation files are
  deliberately not used")*: conditioning is now derived from coordinates on a dated RCSB
  snapshot. Measured against the real `entries.idx` at the default `2024-01-01` cutoff:
  **223,883 eligible entries, 29,677 past the cutoff, of which 10,670 carry an enzyme
  classification.** The staging and indexing halves both run end to end -
  `materialize.py --corpus-targets` writes the assemblies and
  `build_training_index.py` writes the two parquets - and real optimizer steps have been
  taken against the result.
- **Pinned revisions**: LigandMPNN `26ec57ac`, RoseTTAFold3 (foundry) `b02eed6a`, and
  the four reference systems, all resolved against their repositories; the reference
  clones were verified to fetch at those exact SHAs. Both scoring checkpoints are
  additionally pinned by sha256 in `environment/fetch_designability_weights.sh`.

**Not yet done** *(this block was rewritten 2026-09-26; most of the original list has
since been done, and keeping it as written would have misdescribed the task)*:

- **The verifier's grading budget has been measured, but not on the declared hardware.**
  A full graded pass over the 51-target view at 100 designs per target - 5,100 designs, ~43,000
  folds - took **9 h 33 min on 8× L40S** (the view now requests 20 per target, ~1.9 h there) (2026-09-28, slurm 27395799), against the 48 h `task.toml`
  requests. The earlier 8.9 s/fold single-card figure understates by ~2× when divided by
  GPU count: RoseTTAFold3 reloads a ~2.8 GB checkpoint once per *design*, so wall clock is
  dominated by weight loading rather than folding, and a second GPU bought nothing on a
  400-fold run. No pass has been timed on the **H100** the task declares, so the headroom
  is real but its size is still arithmetic. 8 H100s per graded submission also remains a
  heavy provisioning ask that Harbor may refuse.
- ~~**Docker images are unverified.**~~ *Closed 2026-09-30 by the integrator: both images
  now build (development 32/32, verifier 16/16) and their contents are checked. Building
  them is what found the missing compiler; see [docs/VALIDATION.md](docs/VALIDATION.md).*
- **The thresholds are uncalibrated for RoseTTAFold3.** They come from RFdiffusion3's
  `success.txt`, which was calibrated with AlphaFold. Numbers here are internally
  comparable and *not* comparable with the published table.
- **No null-system row.** With 28 of 41 enzyme targets at exactly 0.00, it is not yet
  established how much of the reported pass rates a trivial generator would collect.
- **No `predictor_inversion` reference.** The starter is a `trained_generator`; that
  path is specified and unit-tested but has not been exercised by a real submission.
- **The entity-1 cluster approximation stands**, and the temporal holdout protects only
  the development split - see the contamination limitation above.

Done since the original list, and no longer open: the dataset is produced and real
`.npz` contexts are written from coordinates; designs are scored with real LigandMPNN +
RoseTTAFold3 weights; and a system has been trained (the RESULTS.md row).

**Closed 2026-09-29: the graded path now reproduces the measured row.** The verifier has
been run on `epoch-0100.ckpt` at the real protocol - 51 targets × 100 designs - and every
track agrees with `environment/evaluation/RESULTS.md` within sampling noise (enzyme 0.0624
vs 0.054, sm_binder 0.0275 vs 0.030, ppi_binder 0.002 vs 0.004, unconditional 0.420 vs
0.410). That cross-check is the replayable known answer an integrator can check a build
against, and getting it required fixing **five** divergences in which the graded path was
posing a different problem from the published specs; both the row and the five are written
up in [RESULTS.md](environment/evaluation/RESULTS.md). The earlier 4-design pass that read
enzyme 0.1037 with ppi_binder and sm_binder at zero is superseded and should not be quoted.
**One thing remains unexercised anywhere:** `run_submission`'s `os.setuid` privilege drop,
which no session has had an `agent` user or root to run.

## 7. Limitations

- **Thresholds are uncalibrated for RoseTTAFold3.** They come from RFdiffusion3's
  `success.txt`, which was calibrated with AlphaFold. Numbers here are internally comparable
  and **not** comparable with the published table.
- **The null system is only partly established.** A trivial generator scores 0 on all four
  tracks on a reduced view, but the full row over 51 targets is open, and it matters most on
  `enzyme`, where 28 of 41 targets score exactly 0.
- **`ppi_binder` is measurable, not good.** 1 of 500 passing is not distinguishable from zero.
- **Evaluation targets are contamination-exposed** and **the metric is optimisable** — both are
  set out under section 1, with what the task does and does not do about them.
- **The entity-1 cluster approximation stands**, and the temporal holdout protects only the
  development split.
- **No `predictor_inversion` reference system** has been run, though the paradigm is specified
  and unit-tested.
- **The reference row was measured by the task author, not reproduced by the integrator**,
  because the checkpoint is not distributable; see section 6.

## 8. Citation and licenses

- RFdiffusion3, Baker lab, [bioRxiv 2025.09.18.676967](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2);
  code and weights in [RosettaCommons/foundry](https://github.com/RosettaCommons/foundry).
- BoltzDesign1, [bioRxiv 2025.04.06.647261](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1),
  [GitHub](https://github.com/yehlincho/BoltzDesign1).
- BioLiP2, Zhang C. et al., *Nucleic Acids Research* (2024). Free for academic use.
- M-CSA, Ribeiro A.J.M. et al., *Nucleic Acids Research* (2018). CC BY 4.0.
- RCSB PDB structures and sequence clusters. CC0.
- LigandMPNN, Dauparas J. et al. MIT.
- RoseTTAFold3 / AtomWorks, Institute for Protein Design. BSD-3. Code and model
  parameters are both public; the checkpoint is a pinned download from
  files.ipd.uw.edu.
- AlphaFold 3, Abramson J. et al., *Nature* (2024). No longer used for scoring - its
  model parameters are not redistributable - but its algorithms (Algorithm 18 sampling,
  the optimizer schedule, atom14) are what RFdiffusion3 and RoseTTAFold3 are built on.

Digests, revisions, and licenses are recorded in
[environment/data/sources.json](environment/data/sources.json).
