# Hybrid ESMC-6B + Pairformer/diffusion ~50M

This is intentionally a first-party baseline, not an official paper
reproduction. It enables Protenix's existing ESM embedding path on the verified
48.68M folding core and supplies the frozen cached ESMC-6B mixed-256 features.
The only added trainable component is the official Protenix `linear_esm`
projection from 256 dimensions to `c_s_inputs`; Pairformer, all-atom diffusion,
confidence, losses, permutation handling, optimizer, scheduler, checkpoints,
and inference all remain in the pinned upstream Protenix code.

It uses the same frozen FoldingTrain split as every other trainable baseline,
with MSA disabled. The frozen ESMC-6B parameters are excluded from the folding
model budget, while the 256-to-321 projection is included.

Both training and inference require
`--unified-esmc-manifest /workspace/data/manifests/foldingtrain-100k-v1/features.esmc6b_mixed256.jsonl`.
The wrapper removes that adapter-only option before invoking the upstream CLI.
It uses Protenix's own entity/residue mapping after the official crop and loads
the corresponding frozen `single` tensor from the common cache. Missing,
mismatched, or non-finite features fail closed; they are never replaced with
zeros and no upstream ESM model is run online.

Training additionally requires the same `--unified-native-view-audit` and
explicit native-index arguments as the strict Protenix wrapper, so enabling the
PLM path cannot silently change the admitted structures. The v2 projection
checks every protein entity in each complete official bioassembly against the
frozen feature cache because the released spatial crop may select any of them;
missing coverage is an audited `missing_esmc_feature` rejection.

`Dockerfile.protenix` extends the official Protenix image with only the pinned
`safetensors` data-format reader required by that cache.

The frozen-100K pass, portable bundle gate, and all nine controlled FoldBench
tracks are complete. See [`RESULTS.md`](RESULTS.md) for training provenance and
per-track scores.

`validate_data_adapter.py` runs the adapter against a real Protenix-native PDB
batch and verifies that its PDB ID belongs to the frozen unified train split
and that the post-crop `esm_token_embedding` is present, finite and non-zero.
