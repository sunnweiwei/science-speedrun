# Isolated final verification

CrystalGen final evaluation uses two trust domains.

`main` receives the submission and input-only requests. It validates and copies
the package into a root-owned read-only replay tree, runs `generate.sh` from
that snapshot as UID 10001 with no network, kills remaining submission processes,
and seals a hash-bound copy of the output. It does not mount MP-20, LeMat-Bulk,
hull tables, or MLIP weights.

`grader` receives the sealed snapshot, requests, exact pinned LeMat scorer, and
operator-prepared scoring assets. It never mounts, imports, or executes the
submission. The authoritative orchestration lives in `grader.py`; shared metric
code under `scoring/` is byte-identical to the copy exposed for development.
The grader first relaxes a separate copy of each submitted crystal with the frozen UMA
`uma-s-1` (`omat`), BFGSLineSearch and FrechetCellFilter, up to 200 steps at 0.02 eV/Å and
300 seconds per crystal. It retains finite unconverged iterates and uses the original
candidate when refinement fails. It then scores the resulting geometry with ORB-v3, MACE-MP
and UMA. Mean energy above hull defines stable (≤0 eV/atom) and metastable (0–0.1 eV/atom)
bins, with stable structures excluded from the metastable bin. The primary score written to
`reward.json["score"]` and `reward.json["reward"]` is `(a + b) / 2 = (SUN + 0.5 * MetaSUN) / 2,500`,
where `a = SUN / 2,500` and `b = (SUN + MetaSUN) / 2,500` over all requested slots.

For equal compositions, CrystalGen applies MatterGen's v1.0.3 ordered-disordered matcher.
The additional dilute-parent guard identifies a known lattice decorated with a substituent
occupying ≤25% of the host-plus-guest sites. It requires relative radius difference ≤0.3 and
electronegativity difference ≤1.0. Supercells, symmetry copies and dilute variants of a known
compound such as GdDy7Cu8 from DyCu are covered by these comparisons.

The optional `--as-submitted` evaluation scores the raw lattice and coordinates. It also
measures displacement from the raw geometry under all three potentials on 250 prespecified
slots (50 steps, 0.02 eV/Å), with coverage and a 95% structure-bootstrap interval.
This result is recorded separately from the primary UMA-200 reward.

The official report uses `crystalgen-report/v4-uma200-hull-aligned`. The as-submitted sensitivity uses `crystalgen-report/v3-submitted-geometry`.

Final production suites contain exactly one 2,500-sample unconditional request.
The generation seed is prepared outside Git. No test CSV or conditional targets
are needed. The v1 suite manifest rejects old mixed suites. See `requests/README.md`.

A declared package violation or a reproducible program failure earns a versioned invalid-submission
receipt. Main protects the score directory and administrative markers from the generation UID.
Verifier I/O and launch errors, kernel OOM events and unattributed SIGKILL produce an infrastructure
error with no reward. A real grade requires the matched MACE asset hashes, pinned scorer source,
and finite ORB, MACE and UMA hull energies for each valid input structure. The report is published
before reward.json, which signals completion to the other container.

## Local unit tests

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover -s crystal-structure-generation/tests -p 'test_*.py' -v
```

These use synthetic files and fixture scorer results. They do not download
MP-20, load an MLIP, train OMatG, or establish a scientific score.

## Full verifier

Set the operator root prepared as in `environment/data/MATERIALIZE.md`:

```bash
export TASK_PRIVATE_DIR=/secure/crystalgen   # final-suite/ is mounted at /benchmark
export TASK_DATA_DIR=/secure/crystalgen      # scorer-assets/ is mounted into the grader only
```

Then invoke the Harbor verifier or the Compose definition in this directory.
A package that fails validation, exits nonzero or exceeds the inference time
limit gets `{"score": 0.0, "reward": 0.0, ...}` with the reason; the grader then exits
without scoring. Otherwise `score` and `reward` equal `unconditional_ssi_score = (SUN + 0.5 * MetaSUN) / 2500`.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary task score (`score` / `reward` in `[0, 1]`) is the equal arithmetic mean of **strict stable-unique-novel yield** ($a$) and **stable-or-metastable unique-novel yield** ($b$) over all $N = 2{,}500$ unconditional requested slots:

$$S_{\text{crystal}} = \frac{a + b}{2} = \frac{\mathrm{SUN} + 0.5\,\mathrm{MetaSUN}}{2{,}500}, \qquad a = \frac{\mathrm{SUN}}{2{,}500}\;(\Delta E_{\text{hull}} \le 0),\quad b = \frac{\mathrm{SUN} + \mathrm{MetaSUN}}{2{,}500}\;(\Delta E_{\text{hull}} \le 0.1\text{ eV/atom})$$

Every stage of this evaluation directly inherits established crystal-generation literature and pinned upstream code:
- **Post-generation local structural relaxation (`UMA-200`) + multi-MLIP hull consensus**: In both *MatterGen* (Zeni et al., *Nature* 2025, DOI: `10.1038/s41586-025-08628-5`), *FlowMM* (Miller et al., *ICML* 2024), and *LeMat-GenBench* (NeurIPS 2025 / `arXiv:2512.04562`, pinned commit `58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6`), continuous coordinate/lattice generative models are evaluated **after local structure relaxation** into the nearest basin before computing energy above the convex hull ($\Delta E_{\text{hull}}$). Here, local basin refinement uses frozen `uma-s-1` (`omat` head, `BFGSLineSearch + FrechetCellFilter`, `fmax = 0.02 eV/Å`, max 200 steps), and $\Delta E_{\text{hull}}$ is the three-family ensemble mean across `ORB-v3` (`orb_v3_conservative_inf_omat`), `MACE-MP` (`medium-0b3`), and `UMA` (`uma-s-1`) against the `LeMat-Bulk-MLIP-Hull` (`70d505bb...`) convex hull.
- **Thermodynamic meaning of the $0.1\text{ eV/atom}$ metastability window and $0.5$ weighting**: *Sun et al.* ("The thermodynamic scale of inorganic crystalline metastability", *Science Advances* 2016, `2(11):e1600225`) demonstrated across experimentally realized ICSD polymorphs that ~90% of synthesized metastable inorganic crystals lie within $\sim 0.07\text{–}0.10\text{ eV/atom}$ above the DFT ground-state hull, establishing $\Delta E_{\text{hull}} \le 0.1\text{ eV/atom}$ as the standard S.U.N. threshold in *MatterGen* (*Nature* 2025) and *LeMat-GenBench*. Averaging $a$ ($\Delta E_{\text{hull}} \le 0$) and $b$ ($\Delta E_{\text{hull}} \le 0.1\text{ eV/atom}$) rewards both discovery of strictly ground-state crystals (weight $1.0$) and thermodynamically accessible metastable phases (weight $0.5$) while keeping all invalid, missing, duplicate, or non-novel attempts in the fixed $2{,}500$ denominator.
- **Disorder-aware & dilute-substitution novelty (`MatterGen v1.0.3` + Hume-Rothery guard)**: Structural uniqueness and novelty against `LeMat-Bulk` (`0dc17eea...`, ~5.3M structures) use *MatterGen* `v1.0.3`'s `DisorderedStructureMatcher` (`ltol=0.2, stol=0.3, angle_tol=5.0`) augmented with the dilute-parent guard ($\le 25\%$ sublattice substitution within Hume-Rothery radius $\le 0.3$ and electronegativity $\le 1.0$ tolerances), preventing trivial supercell/alloy-decoration inflation.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **Why `E_form < 0` Is Not Imposed as a Hard Reward Gate**: While most ground-state compounds have negative formation energy relative to elemental reference phases, *Sun et al.* (*Science Advances* 2016) explicitly document experimentally synthesized metastable phases (e.g., high-pressure/kinetically trapped nitrides, carbides, and flat-hull intermetallics) whose formation enthalpy sits slightly above elemental references ($0 \le E_{\text{form}} \le 0.1\text{ eV/atom}$) yet remain within the $0.1\text{ eV/atom}$ metastability window. Neither *MatterGen* (*Nature* 2025) nor *LeMat-GenBench* (`arXiv:2512.04562`) adds an extra $E_{\text{form}} < 0$ gate on top of $\Delta E_{\text{hull}} \le 0.1\text{ eV/atom}$; we therefore preserve upstream protocol fidelity and report `negative_formation_energy_fraction` and `unconditional_negative_formation_unique_novel` strictly as diagnostics.
2. **Why `relaxation RMSD` Is Not Used to Reject Valid Basin Discoveries**: As emphasized in *MatterGen* (*Nature* 2025) and *FlowMM* (*ICML* 2024), geometric proximity of unrelaxed samples to local minima (`RMSD`) and post-relaxation basin stability ($\Delta E_{\text{hull}}$) measure distinct capabilities (direct coordinate fidelity vs basin discovery). Rejecting a genuinely novel, low-energy relaxed crystal because its initial sample took larger BFGS steps would conflate two separate scientific observables; instead, raw-geometry sensitivity (`--as-submitted`) and multi-MLIP `relaxation_rmsd` are reported separately.
3. **Why No Element-Arity (`n_elements`) Quotas or Entropy Multipliers Are Added**: Standard unconditional crystal generation on MP-20 (*CDVAE*, *FlowMM*, *MatterGen*, *OMatG*, *LeMat-GenBench*) asks for unconditional samples without imposing post-hoc ternary/quaternary/quinary quotas. Penalizing a generator's chemical arity distribution after observing outputs would violate model-blind benchmark design; evaluating targeted chemical-system breadth belongs to conditional generation tracks rather than ad-hoc penalties on unconditional yield.

### 3. Diagnostic Signals Retained in report.json

To support thorough scientific analysis without distorting the primary literature metric, `report.json` (under `report["diagnostics"]` and `report["tracks"]["unconditional"]`) explicitly records:
- **Separate stability tiers**: `strict_sun_yield` ($\Delta E_{\text{hull}} \le 0$), `sun_plus_metasun_yield` ($\Delta E_{\text{hull}} \le 0.1\text{ eV/atom}$), global single-pool low-energy unique-novel yield (`global_unique_novel_low_energy_count`), validity, uniqueness, and novelty counts.
- **Formation-energy diagnostics**: `negative_formation_energy_fraction` ($E_{\text{form}} < 0$ fraction among valid structures) and `negative_formation_unique_novel_yield` (`details["negative_formation"]`).
- **Pre-relaxation & geometric displacement diagnostics**: `as_submitted` pre-relaxation CIF accounting (plus full submitted-geometry replay via `--as-submitted`), `relaxation` convergence/fallback counts, and 3-MLIP `relaxation_rmsd` (`displacement` with 95% bootstrap CIs).
- **Compositional & structural diversity**: element-arity distribution (`n_elements` counts and fractions across unary, binary, ternary, quaternary, quinary+ crystals), space-group/site-count/volume distributions, JSD/MMD/Fréchet distances, and elemental HHI.
