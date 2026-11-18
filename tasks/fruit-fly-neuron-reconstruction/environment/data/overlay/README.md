# Release overlay: FPS library for H100

Two files that replace files of the prepared release (`download-source.json`: `sanxing/ai4sci-fruit-fly-brain` at
`86286a6b…`, manifest SHA-256 `db319f33…`). `download_hf.py` applies them when it materializes the release, checked by
size and SHA-256 like every other file. They are stored as `blobs/{sha256}`, the layout of the release.

| Release path | SHA-256 | Bytes |
| --- | --- | --- |
| `development/runtime/point-ops/pathfinder_fps.so` | `34e7c76ca20aed1496697f3707328e76ab7561c6f6ee50f61a28c0c750f575a9` | 324,664 |
| `development/runtime/runtime-manifest.json` | `cbf38bcf01b02a5e14d256cd8607e96cf09009218bf42ce3fe40337321821539` | 36,926 |

The released `pathfinder_fps.so` (PointNeXt furthest-point sampling, used by the SHAPE stage) was built for sm_86 only
and carries no PTX, so on an H100 the first call fails with "no kernel image is available for execution on the
device". This build has sm_86 and sm_90 code. The runtime manifest differs from the released one only in that
library's hash.

Recipe: `upstream/pathfinder/runtime/fps-build.Dockerfile` (Python 3.11.13, the task's hash-locked torch 2.6.0+cu124,
CUDA 12.4, g++ 12.2, `TORCH_CUDA_ARCH_LIST=8.6;9.0`, PointNeXt sources from the release, MIT license). Receipt:
`upstream/pathfinder/receipts/fps-sm86-sm90-rebuild.json`. On an RTX A6000 its FPS indices equal the released
library's in 120 cases; it has not yet run on an H100.
