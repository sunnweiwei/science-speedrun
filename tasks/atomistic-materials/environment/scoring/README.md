# Isolated historical scoring runtime

Use this project's `uv.lock` for trusted scoring. The simulation runtime uses
modern pymatgen and NumPy to run UMA; the scoring runtime uses Python 3.11,
NumPy 1.26.4, and pymatgen 2023.5.10 to reproduce historical MP2020 corrections.
The old pymatgen source build additionally pins Cython 0.29.37 and NumPy 1.26.4.

From the task root:

```bash
uv sync --project environment/scoring --frozen
uv run --project environment/scoring --frozen python environment/evaluation/score.py \
  --references .data/physical-v1-val/references \
  --predictions .results/physical-val --output .results/physical-val/score.json
```

This separation fixes a measured scoring error: pymatgen 2025.6.14 with
`strict_anions="no_check"` still differs from the released WBM labels by
−119.3 and +126.3 meV/atom on two published corner cases. Pymatgen 2023.5.10
reduces both differences below 0.031 meV/atom, within the upstream reference
consistency tolerance. Simply toggling the modern anion setting is insufficient.

The scorer imports no model code and requires no GPU, network or credentials.
Run the simulation and trusted scorer in separate restricted containers for
untrusted submissions; separate virtual environments alone do not isolate files.
