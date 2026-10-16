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

**Crystal Potential Score v2** is the headline objective (0–100, higher is better).
For each primary error below, divide by its fixed scale; discovery uses `(1 − F1) / 0.144`.
Average the two normalized errors within elasticity and amorphous, then compute
`100 × exp(−mean(log(1 + suite_error)))` over the eight suites.

| Suite | Primary error scale(s) |
| --- | --- |
| Discovery | 1 − F1: 0.144 |
| Phonons | Mean-frequency MAE: 18.2 K |
| Elasticity | Bulk: 5.560 GPa; shear: 7.277 GPa |
| κ | SRME: 0.455 |
| MOF | Bulk-modulus MAPE: 26.6% |
| Amorphous | Energy RelRMSE: 8.780%; force RelRMSE: 14.8% |
| Cleavage | Relative MAE: 6.598% |
| NEB | Barrier MAE: 0.147 eV |

Scales are frozen from the MP fine-tune's rounded 3-epoch **validation** table, before expanded scratch
results, and reused on test. 50 matches that reference profile; 100 is ideal. This is a custom
index, not percent accuracy or an official leaderboard metric. It is not comparable to the old
three-suite Score v1. Missing/invalid primary metrics give 0; per-case failures retain the caps
below. `reward.json` contains `reward = score / 100`, `crystal_potential_score`, and all 15 raw
metrics. `report.json` records the score version and suite breakdown. For flat validation metrics,
run `python /workspace/evaluation/aggregate.py --metrics /workspace/research/val-metrics.json`.

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
