# Submission contract

## Package

`/workspace/submission/` is one self-contained directory:

```text
submission/
  calculator.py     defines make_calculator(submission_dir, device)
  README.md         model, training data and mix, compute, validation results, limitations
  ...               weights and any code calculator.py imports (relative to submission_dir)
```

```python
def make_calculator(submission_dir: pathlib.Path, device: str) -> ase.calculators.calculator.Calculator:
    """Return an ASE calculator that implements energy, forces and stress."""
```

- Units: total energy in eV (not per atom, not a formation energy), forces in eV/Å, stress in
  eV/Å³ with ASE's sign convention (positive = tensile). Periodic cells only.
- `device` is `"cuda"`. Each evaluation worker owns one GPU (`CUDA_VISIBLE_DEVICES` is set to it)
  and calls `make_calculator` once, then reuses the calculator for many structures, calling
  `calculator.reset()` between cases.
- The runtime is the image's `/opt/venv` (Python 3.12, FAIRChem 2.16.0, PyTorch, ASE, phonopy
  3.5.1, phono3py 3.30.0). Anything else must be vendored in the package. There is no network and
  no checkpoint download at inference.
- `baseline/*/calculator.py` is a complete example for a UMA checkpoint (`model.pt` next to it,
  FAIRChem `omat` task).

## Pretraining requirement

- Random initialization only; UMA-S is the starter architecture, not a pretrained checkpoint.
- Train on the released pool or its filtered MP starter sample. Fit normalization and elemental
  offsets on training frames only; reserve labeled dev and physical validation for selection.
- Record initialization, selected data hashes, mixtures, fitted statistics and compute in the
  submission README. The starter export includes a preparation receipt; it is audit evidence,
  not a cryptographic proof of scratch training.
- Historical pretrained models/results are comparison material. Their weights are not mounted.

## Energy convention

Stability screening applies the Materials Project 2020 corrections to your raw total energies and
compares them with the MP convex hull. The energies must therefore be on the MP PBE/PBE+U scale
(raw, uncorrected), as in MPtrj. The other suites use energy differences, forces and stress only.
How you reconcile sources computed with different DFT settings is up to you; `data/README.md`
lists each source's settings.

## What the evaluator runs

Every protocol is fixed in `/workspace/evaluation` and is the verifier's own code. The verifier runs
the test split; you can run the validation split with the same commands.

| Suite | Driver | Protocol |
| --- | --- | --- |
| discovery, phonons, elasticity | `run_local.py`, scored by `score.py` (MP2020 runtime `/opt/scoring-venv`) | relaxation, finite-displacement phonons and strain-stress elasticity of the original task |
| kappa | `extra.py --suite kappa` | FIRE + FrechetCellFilter with FixSymmetry, fmax 1e-4 eV/Å, 300 steps; FC2/FC3 with 0.01 Å displacements; imaginary-mode and symmetry gate; Wigner RTA at 300 K |
| mof | `extra.py --suite mof` | your calculator + D3(BJ, PBE, 40 Bohr cutoff); LBFGS cell relaxation (fmax 1e-3, 1,000 steps); 11 volumes within ±4 %, FIRE at fixed cell; Birch–Murnaghan |
| amorphous | `extra.py --suite amorphous` | single-point energy per atom and forces |
| cleavage | `extra.py --suite cleavage` | single points on slab and bulk; E = (E_slab − n_slab/n_bulk · E_bulk) / 2A |
| neb | `extra.py --suite neb` | relax end points; climbing-image NEB (improved tangent, k = 5, FIRE, fmax 0.1, 300 steps); barrier = max − min |

Commands (validation; `--gpus 0,1,...` spreads the work, one worker per listed GPU):

```bash
cd /workspace/evaluation
python run_local.py --inputs /workspace/data/val/physical/inputs/cases.jsonl --factory submission \
  --artifact /workspace/submission --output /workspace/research/val/physical
/opt/scoring-venv/bin/python score.py --references /workspace/data/val/physical/references \
  --predictions /workspace/research/val/physical --output /workspace/research/val/physical-score.json
for s in kappa mof amorphous cleavage neb; do
  python extra.py run --suite $s --inputs /workspace/data/val/$s/inputs --factory submission \
    --artifact /workspace/submission --output /workspace/research/val/$s
  python score_extra.py --suite $s --run /workspace/research/val/$s \
    --inputs /workspace/data/val/$s/inputs --references /workspace/data/val/$s/references
done
```

## Scoring

The primary task score (`score` and `reward` in `reward.json`, in `[0, 1]`, higher is better) is the
**SSI V4 eight-suite equal arithmetic mean** (`1/8` weight per suite), calibrated against frozen
zero-force error anchors stored in `/workspace/evaluation/atom_zero_force_anchors.json`:

1. Let `base = 2 * 0.1648 / (1 + 0.1648)` (`≈ 0.28296703296703296`, the F1 of predicting all WBM
   candidates stable at positive prevalence `0.1648`), and for each error metric `k` with anchor `d_k`
   define `v[k] = 1 - metric[k] / d_k`.
2. Compute the eight unclipped suite skill terms (`1/8` weight each):

| Suite (`1/8` weight) | Metric(s) | Zero-force anchor(s) | Suite skill term |
| --- | --- | --- | --- |
| `Discovery` | `discovery_F1` | `base = 2 × 0.1648 / (1 + 0.1648)` | `(discovery_F1 − base) / (1 − base)` |
| `Phonons` | `phonon_avg_freq_MAE_K` | `419.2277237301322` K | `v["phonon_avg_freq_MAE_K"]` |
| `Elasticity` | `bulk_modulus_MAE_GPa`, `shear_modulus_MAE_GPa` | `107.020216` GPa, `54.186936` GPa | `0.5 × (v["bulk_modulus_MAE_GPa"] + v["shear_modulus_MAE_GPa"])` |
| `Thermal conductivity` | `kappa_SRME` | `2.0` | `v["kappa_SRME"]` |
| `Metal–organic frameworks` | `mof_bulk_modulus_MAPE_pct` | `atom_zero_force_anchors["mof_bulk_modulus_MAPE_pct"]` (ZeroForce + PBE-D3(BJ) anchor) | `v["mof_bulk_modulus_MAPE_pct"]` |
| `Amorphous materials` | `amorphous_energy_RelRMSE_pct`, `amorphous_force_RelRMSE_pct` | `100.0%`, `100.0%` | `0.5 × (v["amorphous_energy_RelRMSE_pct"] + v["amorphous_force_RelRMSE_pct"])` |
| `Cleavage` | `cleavage_relMAE_pct` | `100.0%` | `v["cleavage_relMAE_pct"]` |
| `Migration barriers` | `neb_barrier_MAE_eV` | `0.6803895877435844` eV | `v["neb_barrier_MAE_eV"]` |

3. Average the eight suite terms with equal `1/8` weight (`signed_score = mean(terms)`), and clip once
   at the top level: `score = reward = clip(signed_score, 0.0, 1.0)`.
   Note that because the MOF driver adds `TorchDFTD3Calculator(damping="bj", xc="pbe", cutoff=40.0 Bohr)`
   via `SumCalculator`, the MOF anchor in `/workspace/evaluation/atom_zero_force_anchors.json` is
   parameterized by the unrounded `ZeroForce + D3(BJ)` bulk-modulus MAPE so a zero-force calculator
   scores exact `0.0` on the MOF component.

Missing or invalid primary metrics give `score = reward = 0.0`; per-case failures retain the caps below.
`reward.json` contains `score`, `reward` (`= score`), `signed_score`, the retained historical diagnostic
`crystal_potential_score` (Crystal Potential Score v2, 0–100 geometric mean), and all 15 raw metrics.
`report.json` records `ssi_v4` (version, score, signed_score, per-suite components and anchors) alongside
`aggregate` (historical v2 breakdown). For flat validation metrics, run:
`python /workspace/evaluation/aggregate.py --metrics /workspace/research/val-metrics.json` (pass
`--version v2` to inspect the historical 0–100 v2 diagnostic).

All raw metrics remain available:

- discovery: F1 (stable = hull distance ≤ 0) and hull-distance MAE;
- phonons: MAE of maximum and mean frequency, vibrational entropy, heat capacity and free energy;
- elasticity: bulk and shear modulus (VRH) MAE;
- kappa: SRME (Matbench Discovery);
- mof: bulk-modulus MAPE;
- amorphous: energy and force RelRMSE, averaged over the five systems;
- cleavage: relative MAE;
- neb: barrier MAE over all paths, also reported over converged paths with the converged fraction.

Failures are scored, not dropped. A case that raises, returns non-finite values or is not finished
in time scores as a failure:
- discovery: +1 eV/atom formation-energy error;
- phonons and elasticity: prediction 0;
- kappa: SRME 2;
- amorphous: about twice the error of predicting the mean (energy ±2 standard deviations, forces −F);
- mof: 200 % error;
- cleavage: twice the reference;
- neb: 2 eV error.

For mof, cleavage and neb, per-case errors are also capped at these values. A suite whose driver
crashes before writing anything scores as all failures; the physical suites are then reported
invalid.

## Limits

The verifier runs all eight test suites in one session on the GPUs it exposes (eight H100s in the
reference setup), all suites at the same time with several workers per GPU, within **3 hours in
total**. At the limit every suite stops: the cases it finished are scored, and the rest score as
failed cases (the per-suite failure values above). The most expensive cases start first.

- The mixed-replay baseline finished in **2 h 5 min** (MOF 2 h 4 min, physical 1 h 59 min, NEB 1 h 51 min,
  κ 1 h 9 min, cleavage 5 min, amorphous 2 min). The largest MOF alone took 2.0 h.
- An undertrained calculator is much slower: relaxations and NEB bands do not converge and run to their
  step limits. A 300-update scratch package took 39 min per NEB path (baseline 4.5 min, 0 converged)
  and about 2 h per MOF, so it reaches the limit with failed cases.
- A calculator much slower than UMA-S per step also loses cases to the limit; check the timing on the
  validation subset before publishing.

A CUDA out-of-memory error is retried, not counted as a failure, unless it repeats four times on
the same case.
