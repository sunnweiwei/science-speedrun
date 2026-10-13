---
license: mit
pretty_name: AI4Sci MassSpecGym Retrieval — Grader Labels
language:
- en
---

# MassSpecGym retrieval: grader-only labels

Private operator distribution for `Corning/ai4sci-massspecgym-retrieval`.
The unchanged `labels.jsonl` and `manifest.json` contain the correct candidate
ID and true SMILES for all **17,556** official MassSpecGym v1.5 test queries.
Their total size is **3,810,324 bytes**. `release-manifest.json` binds the exact
payload paths, byte counts and SHA-256 hashes.

Mount only into the trusted grader. Do not put these files or HF credentials
into the agent, its model package, inference inputs, or a public release.
The underlying historical benchmark is public upstream; this private
repository is a runtime isolation mechanism rather than a novel secret test.

The source is MIT-licensed `roman-bushuiev/MassSpecGym` at full revision
`d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`. Retain `UPSTREAM_LICENSES.md`
and cite *MassSpecGym: A benchmark for the discovery and identification of
molecules* (NeurIPS Datasets and Benchmarks 2024). No model weights are included.
The GitHub task pins this repository to an immutable full commit SHA and checks
both release-manifest and payload hashes before use.
