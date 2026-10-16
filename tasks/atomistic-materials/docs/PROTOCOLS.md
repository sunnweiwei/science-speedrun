> Historical **v1 pilot** protocol, retained for the ongoing 500k/5k experiment and Score v1.
> The expanded eight-suite task is defined in [the current contract](../environment/contract.md);
> its pool, panels, failure handling and Score v2 differ. Do not mix these scores.

# Fixed physical evaluation protocol

The implementation adapts [FAIRChem calculation recipes](https://github.com/facebookresearch/fairchem/tree/a1605489fc8ffe875108721b2c25d08b70375f5b/src/fairchem/core/components/calculate/recipes).
The model supplies a conservative ASE calculator; the evaluator controls all
relaxation, displacement and strain settings. Runtime dependencies are locked.

These protocols are unchanged for the from-scratch pretraining track. They assess
the transferred physical behavior of one learned potential, without fitting
separate discovery, phonon or elasticity models. The 1,200 validation / 6,000 test
memberships and scoring targets stay frozen. Training-fitted elemental offsets
must be restored to raw total energy before prediction; the DFT reference energies
and MP2020 corrections below are evaluator resources, not training statistics.
The old pretrained trials remain comparisons with possible inherited data exposure;
they do not establish a random-initialization baseline or convergence budget.

## Stability: Matbench Discovery / WBM

Start from unrelaxed WBM geometry. Use FIRE + FrechetCellFilter, `fmax=0.02` eV/Å,
maximum 500 steps, then return raw total energy and relaxed geometry. Trusted
scoring substitutes these in the corresponding ComputedStructureEntry, reapplies
MP2020 corrections once, subtracts published elemental reference energies, and
divides by atom count:

```
predicted hull distance = published DFT hull distance
                        + predicted formation energy - published DFT formation energy
```

Signed distances are retained; stability means distance ≤ 0 eV/atom. Report F1,
precision, recall, confusion counts and hull-distance MAE (meV/atom), both overall
and on the selected unique-prototype subset. Undefined metrics are JSON null.
Most-stable-10k and geometric leaderboard metrics are not implemented.

Corrections require the separate Python 3.11 / `pymatgen==2023.5.10` scoring lock;
newer versions are rejected. This preserves the historical MP2020 behavior,
including [upstream issue 358](https://github.com/janosh/matbench-discovery/issues/358).
A 128-case reference audit passed with maximum reconstruction residual 0.045
meV/atom, below the 0.1 meV/atom reference-precision tolerance. Published references
and model outputs are not rounded. Full-population identity is not claimed.
MP scoring requires a checkpoint-bound, operator-reviewed raw-energy training
audit. That receipt is not an attestation for hostile submissions.

## MDR PBE phonons

Prediction receives only geometry, masses, supercell and primitive matrices;
DFT forces, force constants and target properties stay outside model inputs.
Relax the primitive cell at `fmax=0.005`, maximum 500 steps. Reconstruct the unit
cell, use 0.01 Å displacements and `symprec=1e-4`, predict supercell forces, and
fit force constants with Phonopy's traditional solver. No post-fit force-constant
symmetrization or nonanalytic correction is applied.

Maximum/mean frequencies use commensurate q-points, signed frequencies, and ASE's
`h × 10^12 / kB` conversion to K. Thermal quantities use a 20×20×20 mesh at 300 K:
entropy and heat capacity in J/(mol primitive cell K), free energy in kJ/mol
primitive cell. These are not per-atom quantities. Default imaginary-mode handling
is retained; these five MAEs do not define a complete dynamical-stability classifier.

The published labels used 47.992 K/THz and Phonopy's default density mesh `100.0`,
per the [reference-generation code](https://github.com/hyllios/utils/tree/4e86e82ae2eec3762f66e6228a70f67b69570789/benchmark_ph).
Near-zero acoustic modes also affect entropy. An 18-reference diagnostic explained
the discrepancies: recipe-versus-label residuals reach 0.035 K, 0.500 J/(mol cell K)
entropy, 0.067 J/(mol cell K) heat capacity, and 0.134 kJ/mol cell free energy.
The task retains UMA's recipe and published targets unchanged; any recomputed
reference variant needs a new protocol version.

## MP PBE elasticity

Use geometry and the published bulk/shear VRH targets from the pinned
`mp-pbe-elasticity-2025.3.json.gz`. Remove non-geometric site fields from model
inputs. Relax with FIRE/FrechetCellFilter at `fmax=0.02`, maximum 500 steps.
Normal strains are −0.01, −0.005, 0.005, 0.01; shear strains are −0.06, −0.03,
0.03, 0.06. Each strained structure relaxes internal positions at fixed cell.
Pymatgen fits the elastic tensor using equilibrium stress. Convert tensile-positive
ASE stress from eV/Å³ to GPa once; report bulk and shear MAEs separately.
Identity with UMA's internal ASE database has not been independently established.

## Coverage and separation

Missing, duplicate or unknown IDs, failed cases, nonfinite values and incompatible
WBM entries invalidate the strict score. No successful-only MAE or outlier removal
is allowed. Finite step-capped relaxations remain included and are counted.
The compact 1,200/6,000 panels are frozen subsets with no atom cap; the complete
original split contains 56,733 validation and 222,310 test cases. Materialize the
HF `protocol` component for the exact assignments.
Prediction/scoring separation currently serves development; restricted mounts,
network isolation and resource limits still need a native final verifier.

## Aggregate task score

`crystal-potential-score-v1` gives the three suites equal weight, independent of
their case counts or number of reported diagnostics. Its four primary metrics are
discovery F1 (`f`), mean phonon-frequency MAE in K (`p`), and bulk/shear modulus
MAEs in GPa (`b`, `g`). Define normalized errors:

```
d = (1 - f) / (1 - 0.8563218390804598)
v = p / 19.767337214745005
e = 0.5 * (b / 4.852471740287598 + g / 9.989963350875858)
score = 100 / ((1 + d) * (1 + v) * (1 + e)) ** (1/3)
```

Equivalently, take the geometric mean of the three suite scores `100/(1+error)`.
Higher is better; 100 is ideal, and 50 matches the reference error profile. This
is a calibrated index, not percent accuracy or an official source-benchmark score.
The calibration is the measured `standard-500k-20260923-01` validation report at
HF revision `1177b47d5c221602ec13216bf1dc3955ca9dc247`,
`records/physical/3ep/score.json`. Constants and metric choices were fixed before
the 30-epoch scratch experiment and must not be refitted to validation/test results.
The same constants apply to both panels; 50 is not a promise about the reference
model's as-yet unmeasured test score.

Invalid coverage, failures and nonfinite outputs yield aggregate zero and
`valid: false`. Partial diagnostic panels receive no aggregate. Finite step-capped
relaxations remain included, following the existing policy. All raw metrics and
convergence counts remain visible: the score does not capture the full phonon
spectrum, dynamical stability, high-temperature MD, surfaces or defects. In
particular, additional correlated phonon thermodynamic errors are diagnostics,
so that suite does not receive extra weight merely for reporting more metrics.
For a future verifier requiring reward in [0,1], use `aggregate.score / 100`.
