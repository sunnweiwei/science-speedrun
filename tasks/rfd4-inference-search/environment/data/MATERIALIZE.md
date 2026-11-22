# Prepare the two views

Run the controller on Linux x86-64 with Python 3.11 or newer, AVX2, network access and about 3 GB for data. Run the preparation command from the repository root.

```bash
export EVAL_ROOT=/path/to/speedrun-results
runner/prepare.sh rfd4-inference-search /path/to/rfd4-data
```

Alternatively, `bash environment/data/materialize.sh /path/to/rfd4-data` from the task directory runs the same recipe. The controller checks SHA-256 on all 148 asset files (including all 141 Figure 2 benchmark inputs in `figure2_benchmarks.json`), then unpacks the checked Foldseek ELF. A second run verifies the files and retains the existing private seed.

| Directory | Contents | Provenance |
| --- | --- | --- |
| `development/assets/rfd4/` | Native merged safetensors (1,742,660,596 bytes) and model config | NVIDIA model revision `a06e2509c7033b010fc43614e9fd840303588502` |
| `development/assets/benchmarks/` | All 141 official Figure 2 `.cif.zst` inputs across 4 tracks (`2.07 MB` total) | RosettaCommons revision `6306a210dd17dcae8e4e0f381b5b74a719d9b30e` |
| `development/assets/monomers/` | Three length-only annotated CIFs, 774 bytes each | RosettaCommons revision `6306a210dd17dcae8e4e0f381b5b74a719d9b30e` |
| `development/assets/af3/` | Standard AF3 parameters, 1,020,545,840 bytes compressed | Google Cloud Storage object generation `1780568696389861` |
| `development/assets/foldseek` | Linux AVX2 executable | Foldseek 10-941cd33 |
| `private/seed` | 32 bytes of entropy drawn once per evaluation campaign | Verifier only |

Each participating host downloads AF3 directly from Google under its parameter terms. Keep model parameters outside the repository and images. [Official AF3 download](https://storage.googleapis.com/alphafold3/af3.bin.zst) · [Parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
