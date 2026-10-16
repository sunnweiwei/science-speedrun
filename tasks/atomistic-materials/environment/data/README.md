# Data

Everything under `/workspace/data` except this file is mounted read-only. Historical model
weights in the HF release are operator-only and are not mounted in this pretraining track.

```text
data/
  pool/                 training pool (ASE-LMDB shards, one directory per source)
  baselines/mp-standard/  the original task's 500k/5k MP fine-tuning sample, filtered for v2
  val/<suite>/inputs/   validation inputs, exactly as the evaluator receives them
  val/<suite>/references/  validation references
```

## Training pool

| Source | Content | DFT settings | Frames in pool |
| --- | --- | --- | ---: |
| `pool/mptrj` | Materials Project relaxation trajectories (MPtrj, CHGNet) | MP: PBE/PBE+U, 520 eV; raw uncorrected energy | 1,065,435 |
| `pool/salex` | sAlex: Alexandria subsample (OMat24 release) | MP-compatible PBE/PBE+U | 9,434,535 |
| `pool/matpes` | MatPES-PBE 2025.1: DIRECT-sampled MD snapshots, 85 elements | PBE without +U, MatPESStaticSet (680 eV) | 193,171 |
| `pool/omat24` | OMat24 training subsets: rattled (300, 500, 1000 K Boltzmann), rattled-relax, AIMD at 1000 K and 3000 K (NVT/NPT); a deterministic 35 % of the shards | OMat24: PBE/PBE+U, PBE54 pseudopotentials | 31,716,685 |
| **total** | | | **42,409,826** (91 GB) |

Every frame is an ASE row with a single-point calculator: total energy (eV), forces (eV/Å) and
stress (eV/Å³, ASE sign: positive = tensile). `row.data` holds `sid`, `source` (for OMat24,
`omat24/<subset>`), `parent` (source material ID) and `settings`. Each source directory has a
`metadata.npz` with `natoms` in file order, so FAIRChem's `ase_db` loader reads a directory directly.
`pool/manifest.json` lists counts, bytes and SHA-256 of every file.

**Energies are not on one scale.** Only `mptrj` and `salex` are on the MP-compatible scale the
discovery suite is scored on. OMat24 uses newer pseudopotentials, so its energies are offset by
an element- and U-dependent amount. MatPES-PBE has no +U, so its energies of transition-metal oxides
and fluorides are offset differently. Forces and stresses are comparable across sources up to
the usual DFT-setting noise. How to combine them (energy referencing, separate heads,
force-only use) is part of the task.

**Exclusions.** A frame is removed when its composition (integer element ratio) or parent
material ID belongs to any evaluation suite, validation or test:
- the original WBM, MDR-phonon and MP-elasticity split;
- the thermal-conductivity materials;
- the MOFs;
- the bulk materials of the cleavage slabs;
- the Li-migration materials.

`pool/blocked.json` lists them. AM26 amorphous systems are not excluded: their compositions (C, Si,
SiO₂, Li-P-S, Ge-Sb-Te) are common in every source.

Choose training frames and mixture yourself; a full pass over the pool is not required. As a rough scale, the MP fine-tuning
baseline passes 1.5 M frames in 2 h 16 min on eight H100s.

## `baselines/mp-standard`

The original task's fine-tuning sample: 500,000 training and 5,000 labeled-development frames, half
MPtrj and half sAlex, at most four per parent, composition-disjoint. Frames touching a v2 evaluation
material are removed (1,159 and 5), leaving 498,841 and 4,995. The format is the one `baseline/uma-mp-finetune/prepare_training.py`
reads (`train.jsonl`, `val.jsonl`, `manifest.json`, `element-references.yaml`). The random starter
`baseline/uma-s-pretrain/prepare_training.py` accepts the same sample, but ignores the published
element references and fits offsets/normalizers only on its training split. These 498,841 frames
are 1.18% of the available pool; using more sources/frames is part of model development.

For other pool selections, split by **reduced composition** across all sources and assert parent
IDs are also disjoint. Keep labeled-dev and all physical-validation labels out of training.

## Validation splits

| Suite | `val/<suite>/inputs` | Size (val / test) | Reference |
| --- | --- | ---: | --- |
| `physical` | `cases.jsonl`: WBM discovery, MDR phonons, MP elasticity cases | 1,000 + 300 + 300 / 5,000 + 500 + 500 | WBM, MDR PBE, MP PBE |
| `kappa` | `inputs.extxyz` with supercell and q-mesh settings | 40 / 63 materials | PhononDB PBE κ at 300 K |
| `mof` | `structures/*.cif` | 40 / 60 MOFs | MOFSimBench PBE-D3 bulk modulus |
| `amorphous` | `inputs.extxyz` | 376 / 558 structures | AM26 energies and forces |
| `cleavage` | `inputs.json.gz`: slab and bulk structures, area | 2,000 / 2,000 slabs, disjoint bulk materials | PBE cleavage energies |
| `neb` | `paths/<edge>_init.xyz`: interpolated path | 256 / 390 paths | nebDFT2k PBE barriers |

Validation and test are disjoint by material (κ: stratified by κ magnitude and structure type;
amorphous: by structure). The test splits are held out from this workspace.
`/workspace/contract.md` gives the commands that run and score a split.

## Licenses

MPtrj: MIT. sAlex, OMat24 and the OMat24 element references: CC BY 4.0. MatPES: BSD-3-Clause.
WBM / Matbench Discovery, MDR and the κ reference: as published with Matbench Discovery (MIT and
CC BY 4.0). MOFSimBench: MIT. Cleavage benchmark: CC BY 4.0 (Zenodo). LiTraj: MIT. AM26: as
published by its authors (no license file at the pinned commit). UMA weights and derivatives:
FAIR Chemistry License and Acceptable Use Policy.
