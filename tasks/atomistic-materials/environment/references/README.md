# References (not verified in this setup)

Material related to each evaluation. Numbers quoted here are from the papers or leaderboards and
were not reproduced in this workspace; the baselines' `RESULTS.md` hold the numbers we measured.

## Universal potentials and their training data

- **UMA** (Meta FAIR Chemistry, 2025): the model family of the starting checkpoint; a mixture of
  linear experts over an eSEN backbone, trained on OMat24, OMol25, OC20, ODAC and OMC.
  FAIRChem 2.16 code is in `/opt/venv`; fine-tuning templates are in
  `baseline/uma-mp-finetune/templates`.
- **OMat24** (Barroso-Luque et al., 2024): 110 M PBE(+U) single points of rattled, MD-sampled
  and relaxed bulk structures. The pool's `omat24/` is a deterministic sample of its training
  subsets. Its DFT settings differ from MP's (pseudopotential versions), so energies are offset.
- **MPtrj** (Deng et al., CHGNet, 2023) and **sAlex** (Alexandria subsample, Schmidt et al.):
  MP-compatible relaxation trajectories; the energy scale the discovery suite is scored on.
- **MatPES** (Kaplan et al., 2025): DIRECT-sampled MD snapshots, PBE without +U, one
  self-consistent setting (MatPESStaticSet).
- **eSEN-30M-OAM, EquiformerV3, Orb-v3, MACE-MPA/OMAT, SevenNet-MF-ompa**: Matbench Discovery
  leaders trained on OMat24 + MPtrj + sAlex. Their weights are not allowed here (Rules); their
  papers describe the data mixtures and fine-tuning schedules that work.

## Systematic softening

- Deng et al., "Overcoming systematic softening in universal machine learning interatomic
  potentials by fine-tuning", npj Comput. Mater. 2025: PES curvature is underestimated because the
  training data sit near equilibrium; a small amount of off-equilibrium data fixes much of it.

## Evaluations

- **Matbench Discovery** (Riebesell et al., Nat. Mach. Intell. 2025): WBM stability screening and
  the κ (SRME) protocol used here. The best published κ SRME on the full 103 materials is about 0.1
  for models trained on OMat24.
- **MDR phonon database** (Loew et al.) and **MP elasticity**: the original task's phonon and
  elasticity references.
- **PhononDB** (Togo): the κ reference; Wigner transport (Simoncelli et al., 2019).
- **MOFSimBench** (npj Comput. Mater. 2025): bulk moduli of 100 MOFs from PBE-D3 EOS scans; every
  universal model tested is about 20 % off.
- **AM26** (Fragapane and Deringer, 2026): DFT energies and forces of 26 amorphous systems; this
  task uses a-C, a-Si, a-SiO₂, a-Li-P-S and a-GeSbTe.
- **Cleavage-energy benchmark** (Zenodo 16970768): 36,718 PBE slabs from the Materials Project.
- **LiTraj** (Dembitskiy et al.): nebDFT2k Li-migration barriers from PBE NEB; this task uses the
  paths without +U transition metals.
