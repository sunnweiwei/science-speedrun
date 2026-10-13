# Provenance

- Upstream repository: `https://github.com/gerstung-lab/Delphi.git`
- Upstream commit: `fb72166be6b29d8db819227a59487e51c1df1454`
- Commit date: 2026-08-07
- License: MIT for code; see `LICENSE.upstream`

Vendored files:

- `model.py`, `train.py`, `utils.py`, and `configurator.py`;
- `config/train_delphi.py` and `config/train_delphi_demo.py`;
- the upstream `requirements.txt`;
- the upstream `ukb_simulated_data` training/validation binaries and metadata, excluding only the conversion notebook.

Local additions are `README.md`, `train_demo.sh`, and `smoke_train.sh`; the repository-level `tests/test_scaffold.py` checks the reference assets. No changes were made to the vendored Python source or data files.

The upstream code and synthetic demo data are for reproduction and engineering tests. The original UK Biobank cohort and trained Delphi-2M weights are not included. Upstream states that separately distributed weights use CC BY-NC-ND 4.0 and must not be used for clinical decision-making.
