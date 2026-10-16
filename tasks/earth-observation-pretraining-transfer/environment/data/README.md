# Data available to the research agent

The operator prepares and mounts the allowed data before the session. Read these
mounts without modifying them; put crops, derived targets and feature caches
under `/workspace/research`. Runtime dependencies are installed in the image.

| Path | Contents and allowed use |
|---|---|
| `/workspace/data/19837/` | 19,837 geographically filtered public pretraining samples; use these for encoder training and derived WorldCover/NDMI supervision. |
| `/workspace/data/eval/awf/` | `inputs.h5` and `labels.json`: 1,115 training and 344 validation rows. Labels are for frozen probe fitting and model/probe selection. |
| `/workspace/data/eval/lfmc/` | `inputs.h5` and `labels.json`: 31,175 training and 8,262 validation rows; no test rows or test labels. |
| `/workspace/data/eval/manifest.json` | Full development-view membership, source hashes and split counts. |
| `/workspace/data/sources.json` | Pinned public sources, download sizes, hashes and licenses. |

The pretraining pool combines public shards 0000 and 0001, then applies the
existing geographic exclusions: the AWF region and a 0.1-degree radius around
LFMC sites. It retains 19,837 of 20,015 samples. Preserve these exclusions when
changing data processing, sampling or derived targets.

The baseline uses Sentinel-1, Sentinel-2 and auxiliary WorldCover maps from this
pool. Its cache and moisture targets are generated during the agent session;
see [the baseline guide](../baseline/starter/README.md). AWF/LFMC labels never
supervise the encoder or auxiliary training heads.

Evaluation HDF5 files contain imagery, IDs and observation masks; their label
JSON files carry split assignments, label values and acquisition metadata.
See [the contract](../contract.md) for the input schema and
[development evaluation](../evaluation/README.md) for the fixed frozen probes.
Missing observations remain in the benchmark and are masked rather than dropped.

The mounted evaluation view is already filtered. Its label `index` fields refer
to the canonical original release; do not feed it back through source-view
preparation. Use IDs to align features and labels, as the supplied probes do.
The final LFMC test release is supplied only to the independent verifier.
Public LFMC test was exposed during earlier development; this is not a claim of
an untouched blind test.

The operator's [materialization guide](MATERIALIZE.md) describes downloads and
mount destinations. Downloaded archives, private evaluation labels and operator
caches are not part of the agent workspace. There is no published baseline
checkpoint required by the recipe; train candidate parameters from scratch or
resume your own permitted candidate lineage.

Attribution: Ai2 OlmoEarth data and implementation; African Wildlife Foundation
annotations; Marta Yebra et al. Globe-LFMC 2.0; ESA Sentinel and WorldCover;
Microsoft Planetary Computer; NASA/USGS Landsat and SRTM; USDA CDL; OpenStreetMap;
WorldCereal; and WRI/Meta canopy height. Exact source licenses are in
`sources.json`. Retain the model's `LICENSE-OlmoEarth` when exporting a package.
