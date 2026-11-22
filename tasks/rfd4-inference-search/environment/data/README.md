# Inputs and assets

All assets under `/workspace/data/assets/` are read-only and verified by SHA-256:

- `assets/benchmarks/`: **all 141 official Figure 2 conditioned `.cif.zst` inputs** across the four biomolecular design tracks (`ppi_backbone` 24 targets, `studio_figure_2c_unfixed` 77 targets, `mcsa_fixed` 37 targets, `dna_unfixed` 3 targets).
- `assets/monomers/`: three length-only annotated CIFs (`100`, `150`, `250` residues) for unconditional monomer checks.
- `assets/rfd4/`: frozen RFD4-Proteina checkpoint (`config.json` and `model.safetensors`).
- `assets/af3/`: AlphaFold 3 parameters used by the common evaluator (`assets/of3/` optional OpenFold3 weights when provisioned).
- `assets/foldseek`: Foldseek `10-941cd33` executable for `TM-score >= 0.5` structural clustering.

Generation, search and evaluation run offline. Each host obtains AF3 weights directly from Google for eligible non-commercial research. [AlphaFold 3 parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
