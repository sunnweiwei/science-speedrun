# CrystalGen data

```text
/workspace/data/
  README.md                this file
  mp20/train.csv           MP-20 train, 27,136 structures: the only data for fitting parameters
  mp20/val.csv             MP-20 validation, 9,047 structures: model selection and diagnostics
  mp20/manifest.json       source revision, row counts and SHA-256 of each split
  scorer-assets/           evaluation tools for /workspace/evaluation (also reachable as /scorer-assets)
```

Everything is mounted read-only. Write derived files (LMDB caches, features, generated structures)
under `/workspace/research`. The MP-20 test split is not mounted and is not needed: the final
evaluation is unconditional generation, not reconstruction of test structures.

## MP-20

The copy under `data/mp_20` in `txie-93/cdvae` at commit
`f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea`.

| Split | Rows | Source SHA-256 | Use |
| --- | ---: | --- | --- |
| `train.csv` | 27,136 | `133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d` | Parameter fitting |
| `val.csv` | 9,047 | `7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29` | Selection and diagnostics |
| `test.csv` | 9,046 | `dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a` | Not mounted |

The measured total is 45,229 rather than the 45,231 stated in the upstream README. CIF values are
multiline quoted CSV fields, so line counts are wrong; use a CSV parser.

Source columns are `material_id`, `formation_energy_per_atom`, `band_gap`, `pretty_formula`,
`e_above_hull`, `elements`, `cif` and `spacegroup.number`. The task adds `density` in g/cm3,
computed from each CIF with pymatgen. These labels are optional training-split analysis; no target
conditions are requested.

## Scorer assets

The prepared inputs of the pinned LeMat-GenBench scorer and of the fixed UMA relaxation:

| Asset | Revision or model | Role |
| --- | --- | --- |
| LeMat-Bulk `compatible_pbe` | `0dc17eea904b860ad7288141e9870f67f8e6bb2c` | novelty reference set |
| LeMat-Bulk MLIP hull | `70d505bb294c16658a9551e2b81ace69a28d9790` | stability hulls for ORB, MACE and UMA |
| UMA | `facebook/UMA` `f611b917…`, model `uma-s-1`, task `omat` | pre-scoring relaxation and scoring |
| ORB | `orb_v3_conservative_inf_omat` | scoring |
| MACE | `mace_mp()` from `mace-torch==0.3.13` | scoring |

They are evaluation tools only. Do not train or distill on them, index them for retrieval, copy
them into the submission, or make the package depend on them.

## License and attribution

MP-20 derives from the Materials Project (CC BY 4.0); the CDVAE code is MIT. LeMat-Bulk is CC BY
4.0. UMA is distributed under the FAIR Chemistry License; ORB and MACE under their upstream
licenses.
