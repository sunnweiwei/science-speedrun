---
license: cc-by-4.0
pretty_name: AI4Sci Fruit Fly Brain prepared resources
language:
- en
tags:
- neuroscience
- connectomics
- electron-microscopy
- ai4science
size_categories:
- n<1K
---
# AI4Sci Fruit Fly Brain prepared resources

Prepared MaleCNS v1.0 resources for the single-checkpoint FFN / PATHFINDER track in [ai4sci-tasks](https://github.com/T0-RSI/ai4sci-tasks/tree/codex/pathfinder-reproduction/fruit-fly-brain).

This is a maintainer-produced derivative of public source data. It contains 722 logical files (14,537,257,535 bytes), stored as SHA-256-addressed objects under `blobs/`. `release.json` maps original relative paths to object hashes and byte counts. It includes no trained weights, model predictions, experiment logs or budget ledgers.

## Contents and visibility

- `development/training`: cropped 16nm EM and segmentation-derived volume supervision for spatial training and validation, plus SHAPE mesh examples including synthetic negatives. Shared observed public neuron IDs are excluded from training according to the pinned spatial plan.
- `development/validation`: validation EM, sparse skeleton reference and evaluator.
- `development/runtime`: pinned Google FFN and connectomics source checkouts (including version metadata), PointNeXt FPS sources/licenses and a compiled FPS extension. The binary was built for the native Python 3.11 / Torch 2.6 / A5000 environment; target-container ABI compatibility is a separate acceptance gate.
- `private/image` and `private/suite`: frozen r3 evaluation EM and sparse scoring reference. “Private” describes the evaluation container mount boundary. These public-source evaluation resources are downloadable by operators; never mount this tree or the download cache into the development agent.

The development manifests contain only training/validation splits. The test evaluates a 512-cubed core inside a 1762-cubed image. Skeletons are locally derived from segmentation, not the official full-connectome skeleton release. Sparse annotations, omitted nodes and local metrics limit comparisons with published dense evaluations.

## Download

Use `fruit-fly-neuron-reconstruction/environment/data/materialize.sh` from the linked task repository. Its checked-in source metadata pins an immutable HF commit and the release manifest SHA-256. Downloading is public and requires no token. The operator materializer verifies all file sizes and SHA-256 values and writes disjoint development/private roots. Downloads resume through the HF cache.

## Attribution and licenses

Source: [MaleCNS official downloads](https://male-cns.janelia.org/download/), v1.0, by FlyEM (HHMI Janelia), University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. Data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the source site. Cite *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, [DOI](https://doi.org/10.1016/j.cell.2026.08.015). No endorsement by the source authors is implied.

Changes: selected spatial crops, shared-ID exclusions, derived sparse skeletons, training targets and synthetic-negative mesh pairs, and repackaging. Source endpoints, frozen regions and exclusions are retained under `provenance/` and in resource manifests. This release packages the already prepared campaign data; it does not establish a new clean download from upstream.

Software retains its own licenses, rather than inheriting the data license: Google FFN/connectomics and PointNeXt/OpenPoints license texts are retained both in the resource inventory and under `licenses/`. Included third-party notices remain in the source trees.

Method reference: *Accelerating Neuron Reconstruction with PATHFINDER*, [DOI](https://doi.org/10.1101/2025.05.16.654254). This adaptation is not an author release or a paper-parity claim. Native training and reconstruction/scoring have completed within documented scopes; see the task README for current results and formal-entry deployment coverage.
