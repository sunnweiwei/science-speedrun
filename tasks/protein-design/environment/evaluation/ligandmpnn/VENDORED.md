# Vendored LigandMPNN

Source for the inverse-folding half of the designability scorer, vendored so that
building either image does not depend on a third-party clone succeeding, and so the
exact code that produced a score is in this repository.

| | |
| --- | --- |
| upstream | https://github.com/dauparas/LigandMPNN |
| revision | `26ec57ac976ade5379920dbd43c7f97a91cf82de` |
| license | **MIT**, Copyright (c) 2024 Justas Dauparas - see `LICENSE`, retained verbatim |
| lands at | `/opt/ligandmpnn` in both the agent and the verifier image |

**This is the original repository, deliberately** - not foundry's `models/mpnn`
re-implementation. Upstream advises against the latter for benchmarking, and it was
measured here returning byte-identical sequences *and* confidences with a bound ligand
present and deleted, which makes it useless for the two ligand tracks. The measurement is
recorded in `tests/pipeline.py`.

## What was taken, and what was left out

Everything on `run.py`'s import closure, plus the license and the upstream docs:

    LICENSE  README.md  requirements.txt  get_model_params.sh
    run.py  score.py  data_utils.py  model_utils.py  sc_utils.py
    openfold/

Left out, because nothing imports them and they are 7 MB of the 7.6 MB checkout:

| left out | what it is |
| --- | --- |
| `inputs/` | example PDB inputs for the upstream README's walkthrough (1.5 MB) |
| `outputs/` | checked-in example results, including `.pt` score dumps (5.4 MB) |
| `training/` | upstream's own MPNN training scripts and a `train.json` fixture |

Verified by parsing every vendored `.py` and resolving every top-level import: the only
names that do not resolve to a vendored module or an installed dependency are `absl`,
`pdbfixer`, `simtk` and `tree`, and the first three appear solely in
`openfold/np/relax/` and `openfold/data/tools/kalign.py`, which are the Amber relaxation
and alignment paths that LigandMPNN never enters. `tree` is `dm-tree`, which both images
already install.

To get the omitted directories, clone upstream at the revision above. Nothing in this
task needs them.

## The NumPy 2 patch is applied at build, not here

This tree is upstream's code unmodified. `openfold` predates NumPy 2 and uses `np.int`,
`np.object` and `np.bool`, all removed in NumPy 2 - and both images pin `numpy>=2.1`
because `biotite` 1.4.0 needs the NumPy 2 ABI. `run.py` imports `sc_utils` ->`openfold`
at module scope, so unpatched, every scorer call would die at import on every track.

Both Dockerfiles rewrite the six occurrences across three files immediately after
copying this directory, and then assert none remain. The patch is left in the build
rather than baked in here so that this directory stays diffable against upstream.

## Weights are not here

The two checkpoints (`ligandmpnn_v_32_010_25.pt`, `proteinmpnn_v_48_020.pt`, ~17 MB) are
a public download with pinned sha256, fetched into `/assets/ligandmpnn_params` by
`environment/fetch_designability_weights.sh`. They are deliberately not committed:
`.dockerignore` excludes `**/*.pt` from the build context, so a committed checkpoint
would be silently dropped from the image rather than used.
