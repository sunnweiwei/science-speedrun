# Inputs and assets

The three annotated CIFs under `assets/monomers/` specify a single unconstrained chain of length 100, 150 or 250 for the standard suite. For another supported length, the evaluator materializes a length-only CIF from the same template. Development and final evaluation use different random seeds. The frozen native model is under `assets/rfd4/`. The independent refolder uses `assets/af3/`. Foldseek clusters the generated monomers using the executable at `assets/foldseek`. All assets are read-only.

The sampler and evaluation run offline. Each host obtains the AF3 weights directly from Google for eligible non-commercial research. [AlphaFold 3 parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
