---
license: cc-by-4.0
pretty_name: AI4Sci atomistic-materials v2 (universal crystal potential task)
tags: [materials, interatomic-potential, dft, benchmark]
---

# AI4Sci atomistic-materials v2

Prepared data for the `atomistic-materials` task of
[T0-RSI/ai4sci-tasks](https://github.com/T0-RSI/ai4sci-tasks): develop one universal crystal
potential that stays MP-compatible and accurate far from equilibrium.

- `pool/`: training pool in ASE-LMDB shards, one directory per source: MPtrj, sAlex, MatPES-PBE,
  and a 35 % shard sample of the OMat24 training subsets. Every composition and parent ID of an
  evaluation material is removed. `pool/manifest.json` lists every file.
- `archives/val.tar`, `archives/test-inputs.tar`, `archives/test-references.tar`: validation and
  test splits of eight suites:
  - WBM discovery, MDR phonons, MP elasticity;
  - PhononDB κ, MOFSimBench bulk moduli, AM26 amorphous energies and forces;
  - cleavage energies, LiTraj nebDFT2k barriers.
- `archives/models.tar`: UMA-S-1.1 and two fine-tuned baselines (FAIR Chemistry License and
  Acceptable Use Policy).
- `archives/baselines.tar`: the original task's 500k MPtrj + sAlex fine-tuning sample, v2-filtered.
- `release-manifest.json`: size and SHA-256 of every artifact.

Install with `environment/data/materialize.sh release DIR` from the task repository.

Licenses of the components:
- MPtrj: MIT;
- sAlex, OMat24 and the cleavage benchmark: CC BY 4.0;
- MatPES: BSD-3-Clause;
- Matbench Discovery references: MIT / CC BY 4.0;
- MOFSimBench and LiTraj: MIT;
- AM26: as published by Fragapane and Deringer (no license file at the pinned commit);
- UMA weights and derivatives: FAIR Chemistry License.

Cite the original datasets when using them.
