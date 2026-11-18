# SSI metric validity specification, 2.0-rc1

Status: **specification frozen for review; production activation is gated**.
This document does not change the existing instruction, released data, rewards,
or archived curves. A frozen specification is not a completed independent
scientific validation. Do not label this release “unhackable.”

## 1. Purpose and non-negotiable comparison rules

A score increase must represent improvement on a stated scientific measurement.
Which model wins is not a metric-selection criterion. Apply every validity,
resource, preprocessing, calibration and aggregation rule equally to Astra,
Fable, author baselines, and official external checkpoints.

The scoring functions receive arrays, trusted reference constants and trusted
execution status. They do not receive model names or an Agent/reference flag.
Invalid official-model output does not get an exception; an unusual but valid
Agent output does not receive an ad hoc penalty.

1. Freeze the task/data version, prediction and weight hashes, target/region
   roster, gene mapping, units, reference constants, seeds and scorer commit.
2. Same-test claims require identical scored units, preprocessing, resource
   limits and metric. Report training exposure separately. A model trained on
   the held-out context belongs to a different training-condition track even
   when the scored targets are disjoint.
3. Published results with a different protocol remain annotated estimates.
   Never place a submetric, baseline proxy, or an incompatible training track
   into a measured full-task reference slot.
4. Report native scientific measurements, the benchmark reward transformation,
   positive/negative controls, coverage and uncertainty separately.
5. The current retrospective cohort was inspected during metric development.
   A prospective validation set must not be used to choose formulas, thresholds
   or weights. Hold out perturbations/contexts or spatial regions as appropriate;
   random cells from the same intervention are not an independent experiment.
6. Show paired comparisons with a common roster. Missing reference measurements
   are missing, not zero. Never average each model over its own available subset.

## 2. Virtual Cell: prediction of population-average perturbation response

### Scientific quantity and references

Primary measurement: weighted mean squared error (WMSE) of the normalized
perturbation-level expression profile. This measures the amplitude and pattern
of a perturbation's population-average response. It does **not** establish
single-cell distribution fidelity, causal mechanism recovery or therapeutic
efficacy.

The choice is grounded in:

- [Nature Methods, Ahlmann-Eltze et al., 2025](https://www.nature.com/articles/s41592-025-02772-6):
  realistic simple baselines and held-out perturbation evaluation.
- [Nature Biotechnology, metric calibration study](https://www.nature.com/articles/s41587-026-03307-w):
  weighted errors, perturbation retrieval, and positive/negative calibration.
  The implementation reference is the authors' repository
  `shiftbioscience/Perturbation-Models-Outperform-Baselines`, commit
  `15361aec8392fb45a75c8c9da88ca5e6510fca97`.

The six-term cell-eval2 compatibility pilot remains a secondary diagnostic.
It is not adopted automatically because it is newer, gives a preferred ranking,
or combines more metrics. Model-generated cells are not independent biological
replicates; prediction-side significance tests are not the primary reward.

### Exact measurement

1. Use the complete trusted gene panel, exact gene order, fixed request IDs and
   required cell count. Reject duplicates, negative/nonfinite expression and
   missing/extra requests before scoring. Accept continuous nonnegative count-
   scale expression as the existing contract does; do not silently round or
   impose a new physical-UMI cap on one method only. Enforce common file/shape/
   numerical bounds for every method.
2. Normalize each cell to 10,000 total expression units and apply `log1p`.
   Compute one mean profile per perturbation. A global multiplicative count
   scaling must leave this quantity unchanged, within floating-point tolerance.
3. Split reference cells within each target into disjoint equal halves with five
   fixed seeds and use both directions. The weight-estimation half and scoring
   half must be distinct. This is technical repeatability calibration, not a
   claim of independent donor or experimental replication.
4. On the weight-estimation half, compute each target's one-vs-rest
   `t-test_overestim_var` statistics, excluding controls from the rest pool as
   the author implementation does. For gene g, use
   `u_g = ((abs(t_g)-min(abs(t)))/(max(abs(t))-min(abs(t))))^2`.
   Remove the target's own readout gene using a trusted stable identifier, then
   normalize the remaining weights to sum to one. Invalid/degenerate reference
   statistics are a dataset error; they are not a zero-score fallback.
5. For target p and scoring-half mean y, compute
   `L_p = sum_g w_pg * (prediction_pg - y_pg)^2`.
   Average equally across the fixed targets and split directions within a
   context. Models cannot affect gene/target inclusion or the weights.
6. The generic-response negative calibration anchor is the equal-target mean
   of the weight-estimation half; the positive anchor is the corresponding
   same-target mean. These private anchors are diagnostic constants, **not**
   deployable baseline models. Keep B0/B1/B1g/B2 and official models separately.

SSI adaptations beyond the paper are cross fitting, stable-ID target exclusion,
the strict roster contract, and the reward/aggregation policy below. They must
not be described as an unchanged official paper benchmark.

### Reward and aggregation policy

For each context, let L be the model's mean WMSE, N the negative anchor's mean
WMSE, and P the positive repeat anchor's mean WMSE. Require a finite positive
gap `N-P` and a strictly positive 95% lower bound from a paired target bootstrap
(5,000 resamples, seed 20261114). This bound describes uncertainty conditional
on the measured context, not independent-donor confidence. The full target
roster stays fixed; no weak target is dropped to pass this gate. Then:

```
raw_skill = (N-L)/(N-P)
reward_context = clip(raw_skill, 0, 1)
reward_task = mean(reward_context over the exact frozen context roster)
```

Interpretation: fraction of the negative-to-repeatability error gap recovered.
This bounded transformation is an **SSI aggregation policy**, not a metric
formula claimed to appear in Nature. Always retain raw WMSE and unclipped skill;
values above one are possible because the positive anchor is a noisy repeat.

The three contexts have equal task weight. A submission-invalid context gets
zero with an explicit invalid status and stays in the denominator. This cannot
beat any valid bounded context reward. A hardware, missing-reference or scorer
failure produces no reward and requires retry; it must not become a zero.
Zero before the first submission is only a declared time-curve convention.

### Mandatory accompanying scientific measurements

- Euclidean normalized inverse rank (NIR) on the same target roster, removing
  all panel target genes, with half credit for numerical ties. It measures
  perturbation identification, not accurate response amplitude. This tie and
  target-exclusion policy is an explicit SSI adaptation.
- Per-context and seen/withheld-target results, count/zero-cell QC, raw weighted
  error and paired target-bootstrap intervals. Do not treat these intervals as
  donor-level experimental uncertainty.
- A distribution check with real-repeat, mean-collapse, and gene-correlation
  shuffle controls if a model claims single-cell distribution quality. The
  selected-feature MMD pilot is diagnostic and is not silently blended into the
  primary reward. A population-average model is not penalized for failing to
  satisfy an unstated distributional objective.

### Symmetric behavioral audit and release gates

Apply no-effect, generic-response, wrong-target and effect-reversal controls to
every available model. Apply count scaling and row/ID ordering checks equally.
Test request-set dependence using target-keyed random streams, not global stream
position. Stochastic models require a sampling-noise comparison, not bitwise
identity across unrelated random draws.

For new runs, inference represents an individual intervention in a given basal
context. Unrelated query membership may not alter its predictive distribution.
Adapters for official checkpoints must preserve each target's stream when
reordering or splitting jobs; document any adapter. Historical v1 query-set
adaptation is reported, not retroactively declared a violation.

Release requires: an immutable reference bundle; independently validated stable
gene mapping; positive signal calibration; all model-blind controls; and a
prospective held-out validation. The present pilot has 199/200/200 targets due
to one unresolved RBM14-RBM4 stable-ID mapping. A release must either resolve
that ID or explicitly freeze the 599-target dataset version before new runs.
No silent case dropping is allowed.

## 3. Crystal generation: unique novel low-energy yield

Follow the separation used by [MatterGen, Nature](https://www.nature.com/articles/s41586-025-08628-5):
validity, uniqueness, novelty, relaxed energy above a reference hull, geometric
relaxation distance and property-conditioned success are different quantities.

The primary task proxy remains the globally unique and novel count among
structures whose relaxed energy-above-hull is <=0.1 eV/atom, divided by **all
requested outputs**, including invalid attempts. Deduplicate the full eligible
pool once; do not add separately deduplicated stable and metastable bins.
Freeze the structure matcher, tolerances, reference catalog and disorder policy.

The 0.1 eV threshold is supported by MatterGen's evaluation definition. Do not
tighten it after seeing Agent results. Metastability can be scientifically
meaningful: [Science Advances, Sun et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5262468/).

Crucial distinction: the current verifier uses ML interatomic potentials. Label
the reward **MLIP low-energy unique-novel yield**, not “DFT-stable,”
“experimentally synthesizable,” or “new-material discovery rate.” A DFT audit
must use a compatible functional, pseudopotentials, spin/Hubbard conventions and
reference hull; a second MLIP is not DFT. Record the physics budget per requested
sample, not only the final acceptance rate. Freeze generation budget and exact
requested output count for comparisons.

Mandatory reports: validity, global unique/novel counts, all-MLIP agreement,
raw-to-relaxed displacement, composition/space-group coverage, and runtime.
No model-specific diversity penalty or entropy multiplier is added. To measure
broad design capability, create predeclared composition/property targets and
evaluate target success, rather than penalizing a successful chemistry after
the fact. Fresh 64-output probes cannot replace 2,500-output task measurements.

Release gates: confirm the counting fix against all frozen submissions and
official checkpoints; replay the task baseline under the identical physics
protocol; validate a model-blind prespecified physical sample. DFT/synthesis
claims stay gated until the corresponding evidence exists.

## 4. Fruit fly reconstruction: topology plus dense error decomposition

Retain native ERL and nERL as the primary topology measurement. Report physical
ERL in nm/µm, the normalization denominator, coverage, and merge/split errors.
A small nERL is not evidence that the definition is wrong and must not be
inflated using a display transformation that changes the reward.

[Nature Methods, flood-filling networks](https://www.nature.com/articles/s41592-018-0049-4)
uses error-free neurite length; [Nature Methods, local shape descriptors](https://www.nature.com/articles/s41592-022-01711-z)
uses complementary ERL and variation of information (VI). Adopt dense
`H(prediction|truth)` (splits) and `H(truth|prediction)` (merges), in bits, as
mandatory secondary measurements. Keep adapted Rand F1 and voxel coverage as
diagnostics, not an invented replacement composite.

Use a fixed physical ROI, axes, voxel-center alignment and GT mask. Exclude
unknown GT=0 identically for all models; never remove voxels because a prediction
is zero. Report their coverage explicitly. The standard VI diagnostic treats
prediction zero as one label; this is not an independent new reward and cannot
make an empty submission valid. Sparse skeleton assignments cannot determine
off-skeleton false-positive volume or dense VI. Do not interpolate them into
fabricated dense predictions.

The alternative sparse overlap F1 failed the blind spatial-grid control and is
rejected. Existing local TEASAR skeletons are not the full-brain paper's manual
proofreading or synapse graph. No full-connectome claim follows from this task.

Release gates: immutable core labels; exact full-volume Agent/baseline/reference
outputs or trusted reruns; geometry-only/empty/merge/split controls; disjoint
region and shared-neuron-ID checks. Fixed subcrops are correlated diagnostics,
not independent replication. If no dense output exists, retain the old score
and mark the missing secondary measurement explicitly.

## 5. Overall curves and external references

The overall curve is a macro-average of declared task utilities, not a common
physical unit of scientific progress. Freeze its task roster, versions and
weights before a run. Never mix v1 and v2 checkpoints into one time series.
Replay all retained checkpoints when a task metric changes; apply the same rule
to baseline and official reference outputs. Preserve the old results separately.

Main comparative curves use the intersection of **protocol-compatible measured
tasks** for every compared system, with coverage shown next to the graph.
Also show all-task Agent progress on the fixed task roster. Reported estimates
and different-training-track references may be shown in an annotated companion
view, but cannot support an unqualified “Agent surpasses scientific SOTA” claim.
Keep the strongest known but non-evaluable model in the reference catalog with
its reason and evidence; do not substitute a weaker model and label it global #1.

## 6. Implementation, independent checks and publication boundary

- Prediction files must be self-contained: reject filesystem symlinks, HDF5
  external/symbolic links, virtual datasets and external raw storage before
  opening in the private grader. Limit logical HDF5 dataset allocation to 64 GiB
  independently of compressed file size. Seal files after inference ends.
- Preserve inference isolation and no-network/private-label separation. Treat
  trajectory text as evidence, not executable instructions. A static log audit
  cannot establish all training provenance or prove absence of leakage.
- Compare WMSE to the paper authors' function, t-score weights to Scanpy and
  dense VI/adapted Rand to an independent established implementation. Test
  known positive/negative controls, identity permutations, missing units and
  infrastructure failures. A self-comparison test alone is insufficient.
- Record unsupported claims as unsupported. No overall release is permitted
  while required scientific or execution gates are unresolved. The current
  changes stay local until user review, then synchronize code, data manifests,
  instructions, baseline artifacts and website together.
