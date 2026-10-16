# Evaluation design (v2)

Goal: adapt a pretrained universal interatomic potential (UMA-S-1.1) into **one** model that stays
MP-compatible for stability screening **and** is accurate far from equilibrium: lattice thermal
transport, mechanics of large porous frameworks, and disordered (amorphous) solids. The agent is told
every objective and gets a labelled validation split for each; the test splits are held out.

## 1. Why this problem

- **Thermal transport.** Thermoelectrics and heat management need lattice thermal conductivity, which
  depends on 3rd-order force constants and is too expensive for DFT at scale (the boron-arsenide story:
  first-principles prediction confirmed experimentally, four-phonon scattering mattered).
- **Systematic softening.** Universal potentials underestimate potential-energy-surface curvature
  because their training data sit near equilibrium (Deng et al., npj Comput Mater 2025). It shows up as
  underpredicted phonon frequencies, thermal conductivity, surface energies and barriers.
- **Disorder and large frameworks.** Amorphous solids are called "a central challenge for future
  universal MLIPs" (Fragapane & Deringer 2026). MOFs are large, flexible, dispersion-bound frameworks
  where every current model is still about 20 % off in bulk modulus (MOFSimBench, npj 2025).
- **The tension measured here (§3).** Naively fine-tuning on MP-compatible data makes all of these
  worse, and longer training makes them worse still. Fixing that while keeping MP compatibility is
  mostly a data-engineering problem: what to train on, in what mix and order.

## 2. Evaluation

### Suites (all are optimisation targets with equal standing)

There is no primary/guard distinction: the agent is asked to improve every suite. The suites differ
only in how much headroom is left (see §3); the original #49 suites are close to the frontier, so gains
there are smaller.

#### New suites

| suite | what is computed | reference | val / test |
|---|---|---|---|
| Thermal conductivity κ | relax, FC2, FC3, Wigner RTA at 300 K (Matbench Discovery phonondb-v1 protocol, phono3py 3.30.0) | PhononDB PBE, Togo (via Matbench Discovery) | 40 / 63 materials, stratified by κ magnitude and structure type |
| MOF bulk modulus | cell relax, ±4 % volume scan with 11 points, Birch–Murnaghan fit; model + D3(BJ) | MOFSimBench PBE-D3 DFT, 100 MOFs | 40 / 60 MOFs |
| Amorphous solids AM26 | energy and force RelRMSE per system (a-C, a-Si, a-SiO₂, a-Li-P-S, a-GST) | AM26 DFT (Fragapane & Deringer 2026) | per-structure split, about 40 % / 60 % |

#### Original #49 suites (unchanged protocol and test panels) and further new suites

| suite | reference | notes |
|---|---|---|
| Stability (Matbench Discovery / WBM), F1 and hull MAE | WBM PBE | #49 panels: 1,000 val / 5,000 test |
| Phonons (MDR) and elasticity (MP) | PBE | #49 panels; validation enlarged to a few hundred cases each |
| Surface cleavage energies | 36,718-slab PBE benchmark (Zenodo 16970768) | sample of 2,000 val + 2,000 test |
| Li-ion migration barriers | LiTraj nebDFT2k, PBE (paths without +U elements) | converged-path MAE and fraction of diverging NEB paths |

The reward stays a set of raw metrics for now; the unified score map comes later. Every test material
or structure is excluded from the training pool by composition (see §4).

## 3. Measured headroom

Lower is better throughout. "Start" is UMA-S-1.1; "3ep" and "30ep" are #49's MP fine-tunes; frontier
numbers come from published per-material predictions recomputed on our splits.

| suite | start | 3ep | 30ep | frontier | trivial / failure |
|---|---:|---:|---:|---:|---:|
| κ SRME (test 63) | 0.361 | 0.514 | 0.687 | 0.066 (Prophet-OAME), 0.119 (EquiformerV3-OAM), 0.181 (eSEN-OAM) | 2.0 |
| κ SRD (all 103; negative = underprediction) | −0.28 | −0.38 | −0.56 | −0.05 (eSEN-OAM) | — |
| MOF bulk modulus MAPE (40) | 19.6 % | 21.9 % | — | 20.9 % (eSEN-OAM) | 157 % |
| AM26 force RelRMSE: a-C / a-Si / a-GST | 18 / 11 / 16 % | 21 / 19 / 18 % | 24 / 18 / 19 % | best model in the paper: UMA-M | — |
| Cleavage energy relative MAE (2,000) | 5.35 % | 6.83 % | 7.23 % | 4.8 % | 33.9 % |
| Li barrier MAE on converged paths, eV (163) | 0.065 | 0.078 | 0.080 | — | 0.49–0.54 |
| Discovery F1 (#49 validation) | n/a (not MP-compatible) | 0.856 | 0.850 | about 0.93 (full leaderboard) | — |

Takeaways:
- Every off-equilibrium property gets worse under MP fine-tuning, and worse again with more epochs.
- κ has the largest headroom: 0.36 → 0.07. Its validation and test splits score alike, so the
  validation split is representative.
- MOF mechanics are unsolved for every model (about 20 %), far better than trivial (157 %).
- Migration barriers without +U elements are close to solved for this model (small headroom); the suite
  also counts robustness (20 of 163 paths did not converge in 100 steps; 8 still diverge at 500).

## 4. Training data pool (target 100–200 GB on disk)

| source | content | full size | proposed |
|---|---|---|---|
| MPtrj | MP relaxation trajectories, PBE(+U), MP-compatible | 1.58 M frames | all |
| sAlex | Alexandria subsample, MP-compatible settings | 10.4 M frames (8.1 GB compressed) | all |
| OMat24 | rattled (300/500/1000), AIMD at 1000 K and 3000 K, rattled-relax; PBE(+U), OMat settings | 100.8 M frames (196 GB compressed) | deterministic 35 % of each subset's shards |
| MatPES-PBE | DIRECT-sampled MD snapshots across 85 elements | about 0.4 M | all |
| MPLiTrj subsample (LiTraj) | Li-containing MP trajectories | 118 k | dropped (label check failed) |

- **Exclusions.** All compositions and source IDs of every evaluation suite (κ, MOF, cleavage
  bulks, nebDFT2k, WBM, MDR, MP elasticity) are removed, extending #49's rule. AM26 is not excluded.
- **Mixed DFT settings.** OMat24 energies are not MP-compatible. How to combine them (separate
  heads, energy referencing, force-only use) is part of the agent's research and is documented, not
  hidden.
- **Throughput.** On #49's measured throughput, 8 H100 × 48 h trains about 30 M samples, so the
  agent must choose and mix data rather than use everything.

## 5. Baselines and packaging (as built)

- **Baselines.**
  - UMA-S-1.1 itself (all suites; discovery shows what the un-adapted OMat-scale energies give).
  - #49's MP fine-tune (3 epochs).
  - A data-mixing reference: MP-compatible data plus an OMat24 off-equilibrium subset, so the
    baseline is not just the degraded recipe.
- **Packaging.** Harbor layout as for the other verified tasks, with a verifier that runs every suite.
  Measured costs per model:
  - κ: about 1 h on 8 workers;
  - MOF: about 1 h on 8 GPUs for 40 MOFs;
  - AM26: minutes;
  - cleavage: about 30 min for 2,000 slabs;
  - NEB: about 40 min for 163 paths;
  - the #49 suites: about 1.5 h for the test panels.

  The verifier will run these in parallel on a single 8-GPU node.

## 6. Open items

- AM26 has no licence file at the pinned commit.
- The MOF reference is PBE-D3; the evaluator adds D3(BJ) to the submitted model, so a model trained
  on PBE without dispersion is compared on the same footing.
- Changes made while building: MPLiTrj was dropped from the pool (labels disagree with UMA-S-1.1:
  median per-frame force correlation 0.39); MatPES stresses were sign-flipped to ASE's convention;
  NEB runs 300 FIRE steps (the study used 100) and caps each error at 2 eV; the physical suites now
  score failures instead of invalidating the run, and no longer require the v1 energy audit.
