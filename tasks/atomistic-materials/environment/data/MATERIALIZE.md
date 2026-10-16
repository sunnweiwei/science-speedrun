# Materializing the data (controller side)

This file is not copied into the agent's workspace. One materialized directory serves both sides:
the agent's compose file mounts `pool/`, `baselines/mp-standard/` and `val/`;
historical models remain operator-only; the verifier's `main`
mounts `test/<suite>/inputs` and the grader also `test/<suite>/references`.

```bash
export TASK_DATA_DIR=/data/atomistic-materials TASK_PRIVATE_DIR=/data/atomistic-materials
```

## Prepared release (default)

```bash
bash environment/data/materialize.sh release /data/atomistic-materials
```

The release is the Hugging Face dataset pinned in `download-source.json` (repository, commit and
SHA-256 of `release-manifest.json`). The manifest pins the size and SHA-256 of every artifact:
- each pool shard (`pool/**.aselmdb`, `metadata.npz`);
- tars of `val/`, `test/*/inputs`, `test/*/references`, `models/` and `baselines/`.

The installer verifies each artifact, assembles a sibling `.partial` directory, runs
`release.py validate`, and renames it into place. Allow 90 GB of disk plus the largest artifact;
a full install took 40 min. `release.py install --groups pool,val,baselines` installs the agent side only, and
`--groups test-inputs,test-references` the verifier side.

## Rebuilding from the sources

```bash
bash environment/data/materialize.sh rebuild /data/atomistic-materials
```

The steps, all in the task's runtime image:
1. The original task's pinned HF bundle (`materialize.py`, pinned in `release.json`) supplies:
   - the v1 split and panels;
   - the 6,000-case physical test panel;
   - the 500k MP sample;
   - UMA-S-1.1;
   - the 3-epoch checkpoint.
2. `fetch_pool.sh` downloads MPtrj, sAlex, MatPES-PBE and the 11 OMat24 training subsets, keeping
   35 % of each subset's shards. That is about 225 GB of transfer.
3. `extra_suites.py` builds the five new suites from pinned URLs with SHA-256 checks:
   figshare (κ), GitHub (MOFSimBench, AM26), Zenodo (cleavage) and LiTraj (nebDFT2k).
4. `evaluation/panels.py --enlarge-v1` and `evaluation/prepare.py` build the 1,600-case physical
   validation panel. Test stays the v1 panel.
5. `build_pool.py` converts and filters the pool against every evaluation composition and parent
   ID. About 1.5 h on 96 cores, dominated by streaming MPtrj.
6. `filter_standard.py` removes v2-evaluation frames from the 500k MP sample.
7. Train `baseline/uma-mixed-replay` and export it. `release.py assemble` then hard-links
   everything into the final layout and validates it.

`release.py publish` packs and uploads a validated release; it resumes from its staging state.

## Measured sizes

| Part | Size |
| --- | ---: |
| pool | 91 GB |
| val + test (inputs and references) | 0.24 GB |
| models (UMA-S-1.1 + 2 baselines) | 2.3 GB |
| baselines/mp-standard | 1.4 GB |
