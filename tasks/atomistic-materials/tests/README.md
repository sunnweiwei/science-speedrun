# Sealed Verifier: `atomistic-materials`

This directory holds the sealed evaluation harness (`main_verifier.py` and `grader.py`) and unit tests for `ai4sci/atomistic-materials`. Nothing in `tests/` is copied or mounted into the Agent development image (`environment/Dockerfile` and `environment/docker-compose.yaml`).

## Two-Container Label-Isolated Architecture

1. **`main` (`main_verifier.py`)**:
   - Mounts only label-free test inputs (`/benchmark/test/<suite>/inputs`) and validates that zero `references*` or `targets*` files exist in the input tree.
   - Copies `/workspace/submission` to a private working directory owned by unprivileged `uid 10001` (`agent`) with `network_mode: none`, `PIP_NO_INDEX=1`, and `HF_HUB_OFFLINE=1`.
   - Runs all eight physical evaluation suites concurrently across the visible GPUs within a shared 3-hour wall-clock budget, terminates all `uid 10001` processes (`SIGKILL`), verifies no symlinks or special files exist in the outputs, and publishes an immutable `0400`/`0500` root-owned snapshot to `/sealed/run` with SHA-256 manifest `READY`.
2. **`grader` (`grader.py`)**:
   - Never imports or executes untrusted submission code.
   - Mounts `/sealed/run` read-only alongside verifier-only ground-truth references (`/labels/<suite>/references`).
   - Scores the physical suites (`discovery`, `phonons`, `elasticity`) in the pinned Python 3.11 / `pymatgen==2023.5.10` MP2020 compatibility runtime (`/opt/scoring-venv`) and the five off-equilibrium suites (`kappa`, `mof`, `amorphous`, `cleavage`, `neb`) via `score_extra.py`.
   - Writes `report.json` and `reward.json`, where `reward.json["score"]` and `reward.json["reward"]` match `harness/ssi_index_v3.py:atom()` to `< 1e-12`.

## Measuring & Updating the MOF `ZeroForce + D3(BJ)` Anchor

Because `environment/evaluation/extra.py` wraps every submitted calculator with `SumCalculator([calc, TorchDFTD3Calculator(device="cuda", damping="bj", xc="pbe", cutoff=40.0 * units.Bohr)])`, a zero-force MLIP still produces non-zero PBE-D3(BJ) dispersion curvature $E_{\text{D3}}(V)$ during the MOF equation-of-state scan. To measure the exact unrounded test-set `bulk_modulus_MAPE_pct` of `ZeroForceCalculator + D3(BJ)` on the 60 held-out test MOFs and atomically update both `environment/evaluation/atom_zero_force_anchors.json` and `harness/ssi_index_v3_inputs.json`:

```bash
python tests/measure_mof_zero_force_d3_anchor.py \
  --inputs "${TASK_DATA_DIR}/test/mof/inputs" \
  --references "${TASK_PRIVATE_DIR}/test/mof/references" \
  --device cuda \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 4 \
  --output mof_zero_force_d3_receipt.json \
  --update-anchors environment/evaluation/atom_zero_force_anchors.json \
  --update-harness-inputs ../harness/ssi_index_v3_inputs.json
```

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary task score (`score` / `reward` in `[0, 1]`) is the **equal-weight (`1/8`) arithmetic mean** of eight zero-force-calibrated physical suite skill scores, clipped to `[0, 1]` at the top level:

$$S_{\text{atom}} = \operatorname{clip}_{[0,1]}\!\left(\frac{1}{8}\sum_{m=1}^{8} s_m\right)$$

where each suite directly inherits an established community benchmark protocol and reference DFT dataset:

1. **Thermodynamic Stability Screening (`Discovery`, $1/8$)** — *Matbench Discovery* (Riebesell et al., *Nature Machine Intelligence* 2025, DOI: `10.1038/s42256-025-01055-1`). Unrelaxed WBM crystal candidates are relaxed with `FIRE + FrechetCellFilter` (`fmax = 0.02 eV/Å`, max 500 steps), re-evaluated with pinned `MaterialsProject2020Compatibility` (`pymatgen==2023.5.10`) against the Materials Project convex hull ($\Delta E_{\text{hull}} \le 0$ stable), and scored by $s_{\text{disc}} = (\mathrm{F1} - b)/(1 - b)$ where $b = \frac{2 \times 0.1648}{1 + 0.1648}$ is the trivial always-stable F1 at the test positive prevalence $p = 0.1648$.
2. **Harmonic Vibrational Phonons (`Phonons`, $1/8$)** — *MDR PBE Phonon Benchmark* (Loew et al.; Togo Phonopy workflow) following FAIRChem's locked finite-displacement phonon recipe (`0.01 Å` supercell displacements after `fmax = 0.005 eV/Å` primitive-cell relaxation), calibrated against the zero-force predictor (`0 K` frequency prediction) anchor $d_{\text{ph}} = 419.2277237301322\text{ K}$: $s_{\text{ph}} = 1 - \mathrm{MAE}(\bar{\omega}) / d_{\text{ph}}$.
3. **Crystalline Elasticity (`Elasticity`, $1/8$)** — *Materials Project PBE Elasticity* (de Jong et al., *Scientific Data* 2015; MP 2025.3 release) via equilibrium cell relaxation plus $\pm 0.5\%, \pm 1\%$ normal and $\pm 3\%, \pm 6\%$ shear strain–stress fits, averaging the zero-force-calibrated Voigt–Reuss–Hill bulk and shear modulus skills equally: $s_{\text{el}} = \frac{1}{2}\big[(1 - \mathrm{MAE}(K_{\text{VRH}})/107.020216) + (1 - \mathrm{MAE}(G_{\text{VRH}})/54.186936)\big]$.
4. **Anharmonic Lattice Thermal Conductivity (`Thermal conductivity`, $1/8$)** — *Matbench Discovery $\kappa_{\text{SRME}}$ (`phonondb-v1`)* (Riebesell et al., *Nature Machine Intelligence* 2025; Togo PhononDB PBE; Wigner transport equation of Simoncelli et al., *Nature Physics* 2019, `phono3py==3.30.0` at 300 K). Calibrated against the Matbench Discovery failure/zero-conductivity ceiling $d_\kappa = 2.0$: $s_\kappa = 1 - \kappa_{\text{SRME}} / 2.0$.
5. **Porous Framework Mechanics (`Metal–organic frameworks`, $1/8$)** — *MOFSimBench* (Krause et al., *npj Computational Materials* 2025), evaluating Birch–Murnaghan bulk moduli $B_0$ from 11-point $\pm 4\%$ equation-of-state volume scans against PBE-D3 references with explicit `TorchDFTD3Calculator(xc="pbe", damping="bj", cutoff=40.0 Bohr)` dispersion added via `SumCalculator`. Because `D3(BJ)` contributes non-zero dispersion curvature $E_{\text{D3}}(V)$ even when the learned potential returns identically zero energy, forces, and stress, the denominator $d_{\text{mof}}$ (`atom_zero_force_anchors["mof_bulk_modulus_MAPE_pct"]`) is calibrated to the exact unrounded test-set MAPE of `ZeroForceCalculator + D3(BJ)` so the trivial potential scores exact $0.0$.
6. **Disordered & Amorphous Solids (`Amorphous materials`, $1/8$)** — *AM26 Benchmark* (Fragapane & Deringer, 2026), evaluating offset-centered energy RelRMSE and force RelRMSE (normalized by per-system DFT standard deviation across a-C, a-Si, a-$\mathrm{SiO_2}$, a-Li–P–S, and a-GeSbTe) against the $100\%$ zero-force/mean-energy predictor anchors: $s_{\text{am}} = \frac{1}{2}\big[(1 - \mathrm{RelRMSE}_E/100) + (1 - \mathrm{RelRMSE}_F/100)\big]$.
7. **Crystal Surfaces (`Cleavage`, $1/8$)** — *Materials Project 36,718-Slab PBE Cleavage Benchmark* (Zenodo `16970768`), scoring slab-minus-bulk surface cleavage energy relative MAE against the $100\%$ zero-energy predictor anchor: $s_{\text{clv}} = 1 - \mathrm{relMAE} / 100$.
8. **Ion Transport (`Migration barriers`, $1/8$)** — *LiTraj `nebDFT2k`* (Dembitskiy et al.), running endpoint relaxations plus 300-step climbing-image NEB (`improvedtangent`, $k = 5\text{ eV/Å}^2$) for $\mathrm{Li}^+$ hops and normalizing barrier MAE against the zero-force zero-barrier anchor $d_{\text{neb}} = 0.6803895877435844\text{ eV}$.

Together, these eight suites directly operationalize the central scientific tension identified by *Deng et al.* ("Overcoming systematic softening in universal machine learning interatomic potentials by fine-tuning", *npj Computational Materials* 2025): naive near-equilibrium MP fine-tuning preserves convex-hull F1 while systematically softening potential-energy-surface curvature (degrading phonons, $\kappa_{\text{SRME}}$, cleavage energies, amorphous forces, and MOF stiffness).

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **No Successful-Only Filtering or Convergence-Rate Multipliers**: Dropping unconverged relaxations or diverging NEB bands would allow brittle potentials to cherry-pick easy crystals, whereas adding a separate ad-hoc convergence gate would double-penalize harmless step-limit stops near shallow minima. Instead, every case in the frozen roster is scored, and per-case errors on diverging nonlinear workflows are bounded by physical failure caps (`kappa_SRME = 2.0` for imaginary modes or broken space-group symmetry per Matbench Discovery; `200%` APE cap on MOF $B_0$; `2 |E_ref|` cap on cleavage; `2.0 eV` cap on NEB barriers).
2. **No Per-Suite Hard Veto or Geometric-Mean Collapse**: Earlier internal prototypes (`Crystal Potential Score v2`) used a geometric mean calibrated to a specific 3-epoch pretrained fine-tune. In V3/V4, the unclipped linear skill average (`1/8` per suite) anchored strictly to trivial zero-force predictors provides transparent, reference-model-free physical units where equal fractional error reductions toward DFT contribute equally across all eight regimes without arbitrary nonlinear distortion.
3. **No Architecture or Inference-Step Micro-Gates**: Beyond the shared 3-hour wall-clock budget on identical hardware and strict scratch-training / zero-network / composition-decontaminated pool rules, the evaluator treats the submitted ASE calculator as a black-box conservative/force predictor (`E, F, stress`) and controls every optimizer, displacement, supercell, and strain tensor itself.

### 3. Diagnostic Signals Retained in report.json

While excluded from the scalar `score` to prevent double-counting correlated observables, `report.json` (and flat fields in `reward.json`) retains rich physical diagnostics for analysis:
- **Stability screening**: `discovery_hull_MAE_meV_per_atom`, precision, recall, confusion counts (`TP, FP, FN, TN`), and `unique_prototypes` subset metrics.
- **Harmonic & anharmonic lattice dynamics**: `phonon_max_freq_MAE_K`, `phonon_entropy_MAE`, `phonon_heat_capacity_MAE`, `phonon_free_energy_MAE`, `kappa_SRE`, signed `kappa_SRD` (directly quantifying systematic PES softening vs stiffening), and `censored_or_failed` imaginary-mode / symmetry-breaking counts.
- **Disordered, framework & transport diagnostics**: per-system AM26 energy and force RelRMSEs (`a-C`, `a-Si`, `a-SiO2`, `a-Li-P-S`, `a-GeSbTe`), signed cleavage error (`cleavage_signed_eV_A2`), converged-only NEB barrier MAE (`barrier_MAE_converged_eV`), NEB `converged_frac`, `nonconverged_initial_relaxations`, `nonconverged_strained_relaxations`, per-suite wall-clock runtimes, unclipped `signed_score`, and historical `crystal_potential_score` (`Crystal Potential Score v2`).
