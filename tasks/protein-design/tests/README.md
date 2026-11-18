# Independent verification

`test.sh` is Harbor's verifier entry point. In a fresh environment it validates the
evaluation view, the submission, and its `harness.json`, runs `infer.sh` as an
unprivileged user with no network, and seals the design index plus every referenced
structure into a read-only, hash-manifested snapshot. The trusted `grader` service then scores that snapshot and
writes `/logs/verifier/reward.json`.

## Two services, one boundary

The submission generates structures. It never sees LigandMPNN, RoseTTAFold3, or the
scoring code: those live in the `grader` service, which reads only sealed structures
and the public conditioning contexts and never imports the submission.

This task has **no hidden labels** - designability is intrinsic - so the boundary is
not there to hide answers. It is there so an untrusted system cannot tamper with the
instrument that measures it.

Pretrained predictors are permitted, so carrying the grading checkpoints is recorded
rather than rejected: `harness.json` in the verifier output lists any that are present,
because a system optimising against the checkpoints that score it inflates in-silico
designability. See the task README's note on this limitation.

## Scores

`reward.json` carries the primary V3 target-weighted combined `score` (and `reward` alias)
alongside every per-track metric flattened with underscore-separated names (`<track>_pass_rate`,
`<track>_diversity`, `<track>_backbone_sc_rmsd_mean`, and the design counts behind them):

```text
score = unconditional / 51 + (50 / 153) * (sm_binder + ppi_binder + enzyme)
```

| Track | Targets | Designs (`20`/target) | Weight in `score` | `pass_rate` | `diversity` |
| --- | ---: | ---: | ---: | --- | --- |
| `unconditional` | 1 | 20 | `1 / 51` | mean per-target fraction of requested designs satisfying `sc_rmsd <= 1.5 A` and novelty `TM < 0.8` | fraction of passing designs in distinct 2.0 A clusters |
| `sm_binder` | 4 | 80 | `50 / 153` | mean per-target fraction satisfying `sc_rmsd <= 1.5 A`, `ligand_rmsd <= 5.0 A`, `iptm >= 0.8`, `interface_pae <= 1.5 A`, and novelty `TM < 0.8` | fraction of passing designs in distinct 2.0 A clusters |
| `ppi_binder` | 5 | 100 | `50 / 153` | mean per-target fraction satisfying `sc_rmsd <= 2.0 A`, `interface_pae <= 1.5 A` under `proteinmpnn-fixed-target/v2`, and novelty `TM < 0.8` | fraction of passing designs in distinct 2.0 A clusters |
| `enzyme` | 41 | 820 | `50 / 153` | mean per-target fraction satisfying non-motif backbone ligand distance `>= 1.5 A`, refolded motif all-atom `RMSD <= 1.5 A`, and novelty `TM < 0.8` | fraction of passing designs in distinct 2.0 A clusters |

Every target requests **20 designs** (RFdiffusion3's published protocol requests 100; the
per-target pass rate estimates the same quantity), so the evaluation view is 51 targets x 20 =
**1,020 designs** (`20 + 80 + 100 + 820`), and grading them folds ~8,600
structures. Grading folds one design per visible GPU (`grader.py --workers`, default
all of them) and `task.toml` asks the verifier for 8, since designs are independent and
each fold is already its own subprocess. Worker count cannot move a score: verdicts are
replayed in submission order, because `cluster_designs` is a greedy walk and `diversity`
would otherwise depend on which fold finished first. The 48 h `[verifier] timeout_sec`
covers the rest, with `main_verifier.py --grader-timeout` and `grader.py --wait-seconds`
at 129,600 s so a correct run is not killed before the verifier's own budget expires. The
rate behind those numbers is 8.9 s/fold measured on an RTX A4000 - ~106 h on one card,
~13 h across 8; the declared H100 has not been calibrated, so the margin is an estimate.

**Target-weighted combined score (`1/51 + 3 x 50/153 = 1.0`).** Because the evaluation suite
consists of `1` synthetic `unconditional` monomer target (`uncond_L100`) and `50` conditioned
RFdiffusion3 benchmark targets (`4` `sm_binder`, `5` `ppi_binder`, `41` `enzyme`), a naive
equal `1/4` track mean would assign `25%` of the score to a single synthetic length-100 monomer
row whose pass rate is an order of magnitude higher than conditioned design. Assigning weight
`1/51` to `unconditional` and splitting the `50/51` conditioned share equally (`50/153` each)
across `sm_binder`, `ppi_binder`, and `enzyme` matches `harness/ssi_index_v3.py::design()`
exactly while preserving balanced sensitivity across all three conditioned modalities.

**`unconditional` is the one track whose target is synthesised, not ingested.**
RFdiffusion3's benchmark has no unconditional target, so a request there is a design
LENGTH and nothing else: `ingest_targets.py::UNCONDITIONAL_TARGETS` emits the single
`uncond_L100` row (length 100, a context holding `design_length` alone), mirroring the
frozen operator definition in
`environment/evaluation/benchmark/standard/targets.json`. Both paths build the same
`{"length": "100-100"}` spec, so the graded number is comparable with RESULTS.md's
operator row on this track as well as the other three.

It was an empty file until 2026-09-28, on the reasoning that a synthesised length does
not belong in a graded view. That was withdrawn because it contradicted the task as
delivered: `instruction.md` tells the agent to build a system serving **all four tracks
in the mounted data**, `contract.md` publishes a passing criterion for the track, and
`materialize.py`'s practice view has always populated it - so an agent developed against
four tracks and was graded on three. Populating the graded view costs ~400 of ~43,000
folds, under 1%, and receives `1/51` of the combined score.

Diversity is reported beside the pass rate, not blended into it. Mode collapse does not
show up in a pass rate, since one designable fold emitted repeatedly passes every time;
on `unconditional` diversity is what separates a generator from a lookup table.

For PPI the grader supplies the binder (chain A) and the fixed target (chain B) to ProteinMPNN; only A is redesigned. RF3 always receives the binder sequence and the target template. The report identifies this protocol as `proteinmpnn-fixed-target/v2`. Archived outcomes from binder-only inverse folding have their own provenance.

A design passes when it is **designable**: LigandMPNN (or ProteinMPNN) sequences refolded
by RoseTTAFold3 satisfy the track's published criterion, which for enzyme includes motif
sidechain recapitulation on the refolded model. The grader adds nothing to that criterion.
Two things run before folding and neither is an addition: `success.txt`'s enzyme
ligand-clash term, which the published criterion measures on the raw design anyway, and
contract validation of the request - the `motif_map` must name one distinct in-range
residue per motif position carrying the requested identity, or recapitulation has nothing
to measure against. The geometry and conditioning gate this section used to describe was
deleted on 2026-09-26; `environment/evaluation/RESULTS.md` tabulates what it cost.

Designs that were never emitted stay in the denominator. A scorer that cannot run
exits nonzero and writes no `reward.json` at all, so infrastructure failure is never
reported as a measured zero.

## Local interface checks

```bash
bash protein-design/tests/run_unit_tests.sh
```

23 tests over synthetic all-atom structures and a stubbed designability scorer. They
cover the input contract, tamper detection, request validation, motif
residue-identity enforcement, the published enzyme ligand-clash term, the harness manifest
across all three paradigms, the failure policy, path escapes, mode collapse, and the
trivial-generator check. The conditioning criteria are asserted through the stubbed
scorer, since that is where they now live. They need only NumPy.

They deliberately do **not** load LigandMPNN or RoseTTAFold3 weights, so they establish
no benchmark result. A real score needs the evaluation view, the fetched `/assets`, and a
trained submission.

They also call `grader.score_track` directly rather than `grader.main()`, so the entry
point that writes `reward.json` is not covered by them. Exercise it explicitly when
changing the grader's output.

## Running the grader outside the image

The defaults (`/assets`, `/opt/ligandmpnn`) are what both Dockerfiles set, so inside a
container nothing needs passing. Reproducing a known answer on another cluster does:

```bash
export PYTHONPATH=<task>/environment/baseline/rfd3-small/rfd3/src:<task>/environment/evaluation/rf3/src
python tests/grader.py --sealed SEALED --contexts VIEW --output OUT \
    --assets /path/to/fetched/assets \
    --ligandmpnn <task>/environment/evaluation/ligandmpnn
```

`--assets` and `--ligandmpnn` also read `PROTEIN_DESIGN_ASSETS` and `LIGANDMPNN_ROOT`,
matching `environment/evaluation/score.py`. Do not run it from `/tmp`: a phantom
`matplotlib` directory there makes torchmetrics believe matplotlib is installed and then
fail on `matplotlib.axes`.

**Off-image scoring needs one patch the image applies and `--ligandmpnn` does not.**
LigandMPNN vendors a copy of openfold predating NumPy 2 which still uses `np.int`,
`np.object` and `np.bool`, and `run.py` imports it at module scope — so pointing
`--ligandmpnn` at a bare checkout dies at import on *every* track with
`AttributeError: module 'numpy' has no attribute 'int'`. The fix lives only in
`environment/Dockerfile`, deliberately, so `environment/evaluation/ligandmpnn` stays
diffable against upstream. Apply it to a **copy**, not to the checkout:

```bash
cp -r <task>/environment/evaluation/ligandmpnn /somewhere/ligandmpnn-patched
sed -i 's/\bnp\.int\b/int/g; s/\bnp\.object\b/object/g; s/\bnp\.bool\b/bool/g' \
    /somewhere/ligandmpnn-patched/openfold/np/{residue_constants.py,relax/utils.py} \
    /somewhere/ligandmpnn-patched/openfold/data/templates.py
```

`fetch_designability_weights.sh` covers the weights but not this, so it is not by itself
the whole portability story.

### The known answer to reproduce

A full graded pass on the reference checkpoint, at the real protocol of 51 targets × 100
designs. A build that lands here is scoring the same quantity as the operator harness; one
that does not is misconfigured rather than unlucky.

| track | `pass_rate` | `diversity` |
| --- | ---: | ---: |
| enzyme | 0.0624 | 0.341 |
| sm_binder | 0.0275 | 0.750 |
| ppi_binder | 0.002 | 0.200 |
| unconditional | 0.420 | 1.000 |

`conditioning_satisfied` is 100% on all four tracks, and the pass takes **9 h 33 min on
8× L40S**. The checkpoint, the packaged submission and both JSON artifacts are outside the
repo at `/net/tukwila/yanjing/verifier-ref-fixed/`; ask before assuming a local path. The
operator-side comparison and the five spec bugs that had to be fixed before the two paths
agreed are in `environment/evaluation/RESULTS.md`.

Packaging a submission to replay this also means writing `harness.json` by hand; nothing
in the starter emits it. See `environment/baseline/rfd3-small/README.md`, "You must write
`harness.json` yourself".

### Full 51-target × 20-design (`1,020`-slot) straight-helix negative control

`tests/generate_trivial_helix_1020.py` builds a deterministic, contract-valid `1,020`-backbone
null submission across all `51` evaluation targets (`1` `unconditional` + `4` `sm_binder` +
`5` `ppi_binder` + `41` `enzyme`, `20` designs/target = `1,020` slots). Each slot places an
idealized canonical straight right-handed alpha-helix (`3.80 A` `CA-CA` spacing, `100 deg` turn,
`1.50 A` axial rise) of the exact requested `design_length` near the conditioning centroid (with
valid `motif_map` and `motif_restypes` on `enzyme`) without clash optimization, motif geometry
grafting, or inverse-folding tricks, and packages a self-contained submission ready for sealed
GPU verification via `main_verifier.py` or `grader.py`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

1. **Upstream all-atom designability definitions (*RFdiffusion3* & *ProteinMPNN*).** Per-track
   acceptance criteria follow the published benchmark specifications (`results/*/success.txt`)
   of *RFdiffusion3* (Butcher et al., *bioRxiv* 2025, `10.1101/2025.09.18.676967`) paired with
   all-atom / backbone inverse folding from *ProteinMPNN* (Dauparas et al., *Science* 2022,
   `10.1126/science.add2187`) and *LigandMPNN*, refolded in complex by *RoseTTAFold3* (`RF3`)
   and gated for structural novelty against the training split (`Foldseek` TM-align
   `max_TM < 0.8` against `/reference/fit`):
   - `unconditional` (`1` synthetic monomer target `uncond_L100` × `20` designs = `20` slots):
     ProteinMPNN (4 sequences) + RF3 refold with `sc_rmsd <= 1.5 A` and novelty `TM < 0.8`.
   - `sm_binder` (`4` bare-ligand targets `FAD, IAI, OQO, SAM` × `20` designs = `80` slots):
     LigandMPNN (4 sequences) + RF3 holo refold requiring `sc_rmsd <= 1.5 A`, backbone-aligned
     `ligand_rmsd <= 5.0 A`, `iptm >= 0.8`, `interface_pae <= 1.5 A`, and novelty `TM < 0.8`.
   - `ppi_binder` (`5` canonical binder targets `rfd3_insulinr, rfd3_pdl1, rfd3_il7ra,
     rfd3_il2ra, rfd3_tie2` × `20` designs = `100` slots): target-conditioned binder-only
     ProteinMPNN (`proteinmpnn-fixed-target/v2`, supplying fixed target chain `B` while
     redesigning only binder chain `A`, then templating chain `B` in RF3; calibrated on crystal
     complexes `1BRS` and `7JZU` where positive controls pass `4/4` and wrong-target negative
     controls fail `interface_pae` at `7.75-15.63 A`) requiring `sc_rmsd <= 2.0 A`,
     `interface_pae <= 1.5 A`, and novelty `TM < 0.8`.
   - `enzyme` (`41` M-CSA catalytic active-site targets `M0024_1nzy` .. `M0907_1rbl` × `20`
     designs = `820` slots): exact upstream Atomic Motif Enzyme (`AME`) `success.txt` criterion
     — non-motif backbone `N/CA/C` to ligand atom distance `>= 1.5 A` on the submitted design
     plus motif all-atom `RMSD <= 1.5 A` (aligned on motif `N/CA/C`, falling back to shared
     constrained atoms when fewer than 3 motif backbone atoms are fixed) across 4 LigandMPNN
     sequences × 5 RF3 diffusion samples (`20` refolds/design), with fixed motif residue
     identities and novelty `TM < 0.8`.
2. **Target-count weights (`1/51 + 3 × 50/153 = 1.0`).** Across the `51` evaluation targets
   (`1,020` total design slots), `score` (and `reward`) is locked to `harness/ssi_index_v3.py::design()`:
   `score = unconditional / 51 + (50 / 153) * (sm_binder + ppi_binder + enzyme)`.
   Assigning weight `1/51` to the single synthetic `uncond_L100` target prevents an
   unconstrained monomer task (whose pass rate is ~10× higher than conditioned all-atom design)
   from dominating the headline metric, while splitting the `50/51` conditioned weight equally
   (`50/153` each) across `sm_binder`, `ppi_binder`, and `enzyme` ensures balanced sensitivity
   across small-molecule binding, protein-protein interface design, and catalytic scaffolding
   despite their unequal target counts (`4`, `5`, and `41`).
3. **Public in-silico oracle scope boundary.** Modern de novo binder and scaffold design
   pipelines — including *AlphaProteo* (Zambaldi et al., 2024), *BindCraft* (Pacesa et al.,
   *Nature* 2025, `10.1038/s41586-025-09429-6`), and *BoltzDesign1* (Cho et al., *bioRxiv* 2025,
   `10.1101/2025.04.06.647261`) — legitimately use structure-prediction and inverse-folding
   networks inside inference-time `Best-of-N` filtering, hallucination, or `predictor_inversion`
   loops. Shipping public `RF3 + LigandMPNN` checkpoints in `/assets` and permitting
   `trained_generator`, `predictor_inversion`, and `hybrid` paradigms reflects real scientific
   practice; this benchmark explicitly measures **in-silico designability under a public
   all-atom structure-prediction oracle** (requiring wet-lab validation for physical affinity or
   catalysis), and `main_verifier.py` audits `/pretrained` and `grading_instruments_present` in
   `harness.json` so oracle-guided entries are transparently identified.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **No ad-hoc global `sc_rmsd` gate on `enzyme`.** Upstream *RFdiffusion3* AME `success.txt`
   evaluates enzyme scaffolds strictly on catalytic active-site recapitulation (`motif_rmsd <= 1.5 A`)
   and absence of non-motif ligand clashes (`>= 1.5 A`), deliberately omitting a global scaffold
   `sc_rmsd <= 1.5 A` cutoff. In `25` of the `41` M-CSA enzyme targets (`61%`), `select_fixed_atoms`
   constrains fewer than `3` motif backbone atoms (fixing only catalytic sidechain tip functional
   groups), allowing distal loop rearrangements between the raw diffusion trace and the refolded
   holo structure while the catalytic triad/oxyanion hole folds to `< 1.5 A` all-atom RMSD.
   Adding task-invented pre-fold geometry or global `sc_rmsd` gates previously discarded `59%`
   of genuinely designable enzyme scaffolds on the reference `rfd3-small` evaluation (`0.022` vs
   `0.054` pass rate; see `environment/evaluation/RESULTS.md`).
2. **No ad-hoc pre-fold hotspot-distance or ligand-burial heuristics.** Earlier ad-hoc pre-fold
   gates on radius of gyration, raw ligand burial, and raw hotspot contact rejected `100%` of
   designable `ppi_binder` structures before refolding. Complex refolding metrics (`iptm >= 0.8`,
   `interface_pae <= 1.5 A`, `ligand_rmsd <= 5.0 A`, `sc_rmsd <= 1.5/2.0 A`) directly test
   whether the designed sequence forms a confident holo interface in RF3 without brittle
   hand-tuned geometric proxies.
3. **No blending of structural diversity into `pass_rate`.** Folding `diversity` into `pass_rate`
   would conflate sample foldability with mode coverage and make per-target pass probabilities
   depend on sample-set clustering heuristics.

### 3. Diagnostic Signals Retained in report.json

While excluded from ad-hoc pass/fail gating, `report.json` (and flattened numeric fields in
`reward.json` plus verifier `harness.json`) retains rich diagnostics for post-hoc scientific
audit:
- `diversity`: within-target structural cluster fraction at `2.0 A` CA RMSD over passing designs
  (directly exposing mode collapse, especially on `unconditional`).
- `backbone_sc_rmsd_mean` and `designable_backbone_sc_rmsd_mean`: global designed-chain CA
  self-consistency RMSD across all folded designs and across passing designs on every track
  (including `enzyme`, quantifying global scaffold rigidity alongside active-site motif RMSD).
- `novelty_max_tm_mean`, `designs_requested`, `designs_returned`, `designs_valid`,
  `conditioning_satisfied`, `designs_designable`, `designs_novel`, and structured `failures`
  histograms recording exact per-criterion rejection reasons.
- `grading_instruments_present`, `kind`, `declared_parameters`, and `pretrained_assets` in
  verifier `harness.json` to distinguish pure learned generators from oracle-guided
  `Best-of-N` / `predictor_inversion` systems.
