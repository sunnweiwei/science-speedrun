# Autoencoder reconstruction holdout

This check uses the released autoencoder to encode and reconstruct the 640 fixed structures in the official `ae_pdb_temporal_holdout_2026_10_04` dataset. The five file panels contain 128 structures each. It tests local structure reconstruction with known inputs. The four-suite inference search comparison and S15 monomer generation have different endpoints. [Official dataset](https://huggingface.co/datasets/nvidia/RFD4-Proteina/tree/95a7d244d64343de40784ecb9238ef9de4e0db37/benchmarks/ae_pdb_temporal_holdout_2026_10_04) · [Released validation config](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/configs/datasets/val/ae/pdb_temporal_holdout.yaml)

Only the evaluation files and the separate released AE checkpoint are needed. These pinned commands can populate a fresh controller cache.

```bash
hf download nvidia/RFD4-Proteina --repo-type dataset --revision 95a7d244d64343de40784ecb9238ef9de4e0db37 \
  --include 'benchmarks/ae_pdb_temporal_holdout_2026_10_04/*' --local-dir /path/to/data
hf download nvidia/RFD4-Proteina --revision a06e2509c7033b010fc43614e9fd840303588502 \
  ae_checkpoints/rfd4_proteina_ae-ema.ckpt --local-dir /path/to/weights
```

The driver checks all file names, counts and content hashes, then runs the upstream Lightning `AutoEncoder.validation_step` and `on_validation_epoch_end` on the published validation preprocessing config. It loads the released native EMA checkpoint with strict parameter matching. It checks that every input ID appears once. Batch size is one, the validation seed is 43 and the GPU uses bf16 mixed precision. The reported values are the upstream permutation-aware **means of per-residue atom RMSDs in Å**, weighted equally over supervised residues within each residue group. The residue groups span the five file panels.

Use the same Python environment as native RFD4 and a fresh output directory. The first argument is the directory containing the five published subdirectories. The checkpoint is `ae_checkpoints/rfd4_proteina_ae-ema.ckpt` from model revision `a06e2509c7033b010fc43614e9fd840303588502`.

```bash
python docs/ae-holdout/validate.py /path/to/ae_pdb_temporal_holdout_2026_10_04 \
  /path/to/rfd4_proteina_ae-ema.ckpt /path/to/new-output
```

Add `--smoke` to check one input from each file panel on CPU. The GPU run writes its measured example count, residue counts and reconstruction errors to `summary.json`.

## Local GPU results

The RTX A6000 run processed all 640 released inputs exactly once in 143.6 seconds of validation. The rows group residues across inputs. Each file panel contributed 128 structures. [Machine-readable result, hashes and settings](results.json)

| Residue group | Supervised residues | Mean per-residue RMSD (Å) |
| --- | ---: | ---: |
| Standard amino acids | 154,844 | 0.0864 |
| Nonstandard amino acids | 198 | 0.1193 |
| Glycans | 229 | 0.0853 |
| DNA | 6,108 | 0.1338 |
| RNA | 4,541 | 0.1008 |
