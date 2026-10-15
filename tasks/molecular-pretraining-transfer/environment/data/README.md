# Data

Everything under `/workspace/data` is read-only. Write derived data under `/workspace/research`.

## `pretrain/`: pretraining corpus

The Uni-Mol2 pretraining set (Hugging Face `dptech/unimol2-dataset` at revision 4748959, Apache-2.0), shards 0–11
of 80. Each LMDB maps a 4-byte big-endian index to a pickled `{"smi", "atoms", "coordinates"}`. `atoms` lists
every atom including hydrogens, and `coordinates` is one 3D conformer (an N × 3 array). This is the format the
official `unimol2` pretraining task reads.

| file | molecules |
|---|---:|
| `train_part_0.lmdb` … `train_part_10.lmdb` | 10,485,658 – 10,485,691 each |
| `train_part_11.lmdb` | 10,385,683 |
| `valid_part_0.lmdb` | 100,000 (the last molecules of shard 11, held out) |
| total training | 125,728,125 |

**Leakage filter.** Every molecule whose InChIKey connectivity layer (first 14 characters) matches a molecule of
any evaluation set was removed. The evaluation sets are QM9, COMPAS-1D, ExpansionRx (train, test and raw) and ASAP,
145,619 keys in all. The filter removed 995 molecules; no molecule was unparsable. Per-shard counts are in
`filter_report.json`.

The shards have slightly different lengths. The official loader keeps the first shard's batch sampler for every
epoch, so the baseline truncates every shard to one length; see `baseline/unimol2-ref/unimol2x`.

## `downstream/`: fine-tune data of the 24 evaluation tasks

`downstream/<task>/{train,valid}.lmdb` use the Uni-Mol `mol_finetune` format: pickled `{"atoms", "coordinates"
(11 conformers), "smi", "target"}` under keys `"0"`, `"1"`, …. The files also include:

- `manifest.json`: group, number of targets and sizes per task, including the test size;
- `task_metainfo.json`: train-split mean/std for target normalisation of the tasks that the official code does
  not know. The official QM9 statistics are built into Uni-Mol2.

The test splits are held by the verifier only.

| group | tasks | how the data were built |
|---|---|---|
| QM9 | `qm9dft_v2` (HOMO, LUMO, gap), `qm9dft_v2_{alpha,cv,mu,r2,zpve}` | Uni-Mol's QM9 LMDBs with DFT geometries (11 identical copies) and targets from `qm9.csv`. Uni-Mol2's split is not released, so the 133,885 molecules are split 8:1:1 by scaffold similarity (Bemis–Murcko scaffolds, Butina clusters at Tanimoto distance 0.6). A seeded subset of 5,000 train / 1,000 valid is used for fine-tuning; the test set is complete (13,378). |
| COMPAS-1D | `compas1d_{aea,aip,disp,dipmom}` | COMPAS-1D B3LYP/def2-SVP (gitlab porannegroup/compas, CC BY 4.0), 8,678 molecules with DFT geometries. Scaffold-similarity split with the cutoff tightened to 0.1, then 5,000 train / 870 valid / 862 test. |
| ExpansionRx | `exrx_{logd,ksol,hlm_clint,mlm_clint,caco_2_permeability_papp_a_b,caco_2_permeability_efflux,mppb,mbpb,mgmb}` | OpenADMET-ExpansionRx challenge data (HF `openadmet/openadmet-expansionrx-challenge-data` at 6b898cc, CC BY 4.0), with the official train/test time split. Validation is the latest 10 % of each endpoint's training molecules by molecule ID. |
| ASAP | `asap_{hlm,ksol,logd,mdr1_mdckii,mlm}` | ASAP antiviral-ADMET-2025 (Polaris `asap-discovery/antiviral-admet-2025-unblinded`, CC0), official Train/Test. Validation is a seeded random 10 % of training. |

ADMET molecules have 11 conformers: 10 RDKit ETKDG embeddings with MMFF optimisation, then one 2D layout, the
same as Uni-Mol's `smi2coords`. ADMET targets are log10(clip(y, 0) + 1), except LogD, which is unchanged. These are
the challenges' scoring units.
