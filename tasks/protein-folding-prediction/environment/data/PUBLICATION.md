# Frozen data publication — 2026-09-19

This is a storage/publication change, not a new scientific data selection or
benchmark result. GitHub retains the Harbor task, contract, starter/reference
code and evaluator. Hugging Face hosts the actual large data files.

## Location and immutable identity

- Public, ungated dataset:
  [sunweiwei/ai4sci-foldbench-data](https://huggingface.co/datasets/sunweiwei/ai4sci-foldbench-data).
- Pinned HF commit: `bc9f10953a644ca51428f88fbc8e1adce909f761`.
- Release: `foldbench-100k-esmc-int8-v1`.
- Archive-manifest SHA-256:
  `8dcea4a848171652e7647e43397cacff55316a8998186403f856c5395d9ac622`.
- Per-file inventory SHA-256:
  `976eaa6271d133f0fcb44f9c8a2d697766c79096de82ebdba3871773052ffa0a`.

`download-source.json` and `release-manifest.json` are the small machine-readable
download contract checked into Git. The full 36,565,890-byte per-file inventory
is hosted on HF along with the payload, not committed to GitHub.

## Published payload

| Restored subtree | Files | Bytes | Meaning |
| --- | ---: | ---: | --- |
| `manifests/` | 3 | 291,989,362 | 100K logical training records and feature/structure indexes |
| `objects/` | 90,000 | 7,994,107,456 | Unique training structure NPZ files |
| `features/` | 91,366 | 61,910,892,072 | Complete model-independent INT8 training features |
| `foldbench/` | 4,623 | 2,116,164,115 | Full evaluation suite, native structures and 1,568 feature files |
| `models/` | 3 | 755,429,835 | Reference-only ESMFold2-Fast weights/config/upstream card |
| **Total** | **185,995** | **73,068,582,840** | **73.1 GB decimal / 68.0 GiB** |

There are 75 byte-preserving uncompressed tar shards totaling
73,306,757,120 bytes. The largest is 1,019,514,880 bytes. Sharding avoids a flat
91K-file feature folder on the Hub while restoring the unchanged local layout.
The HF repository contains exactly those shards, the inventory, the release
manifest, a dataset card and `.gitattributes` (79 files); it does not contain
the Harbor source tree.

Not published: ESMC-6B weights, unquantized feature masters, MSA resources,
organizer caches, credentials or experiment runs. The reference checkpoint is
not allowed as a submitted model or a candidate weight initialization. Upstream
licenses and model-card notices still apply; no blanket relicensing is claimed.

## Integrity and reproducibility checks

- The clean source release passed `validate_release.py`: 91,366 training
  features, 1,568 suite features, no exposed ESMC checkpoint and the expected
  ESMFold2-Fast reference checksum.
- Packing hashes every file while writing archives. The canonical training
  manifest and suite manifest are unchanged; no tensor is requantized.
- Anonymous Hub metadata checks verified all 75 remote archive SHA-256 values,
  the inventory SHA-256, every corresponding byte size and the exact remote
  file membership at the pinned commit.
- The full release was downloaded anonymously into a new empty directory,
  using public HF payloads, not local source copies or hardlinks. All 185,995 files
  was restored and SHA-256 checked, and the existing scientific release audit
  passed with 91,366 training and 1,568 suite features. The official HF CLI
  additionally prefetched later shards in parallel into the same download
  cache to shorten this acceptance test; the materializer still verified and
  extracted every shard. No disposable downloaded tar copies remained.
- A second full `--verify-only` audit passed in a container with networking
  disabled and both code and downloaded data mounted read-only, including
  all file hashes and the scientific release audit. No HF credentials or
  writes to the data directory were needed.
- Offline archive tests cover a byte-identical round trip, reproducible packing,
  interrupted/partial restores, default archive removal and explicit retention,
  offline verification, corruption rejection, path traversal, symlinks, duplicate
  or missing archive members, conflicting existing files and forbidden weights.
- All **42** FoldBench regression tests passed in the existing development
  image with networking disabled. This includes the real pinned native-scorer
  fixture using benchmark data downloaded from HF. That fixture uses a native
  as a plumbing oracle: it is **not a trained-model benchmark result**.

Materialization downloads into an operator-selected data directory. Successfully
extracted disposable archive copies are removed by default; original source data
and conflicting existing files are never removed or overwritten. Runtime mounts
remain the existing five allowlisted development subtrees and the independent
verifier's benchmark-only subtree. No runner, contract, metric or prompt change
is required.
