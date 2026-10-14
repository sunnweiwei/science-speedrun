# Materializing the FoldBench data (controller and maintainers)

This file is not copied into the agent's workspace. Set `TASK_DATA_DIR` and `TASK_PRIVATE_DIR` to
the materialized release. The development compose mounts `manifests/`, `objects/`, `features/`,
`foldbench/` and `models/` under `/workspace/data`, and the verifier compose mounts `foldbench/` at
`/benchmark`.

## Operator preparation (before launching the agent)

The public data repository is
[sunweiwei/ai4sci-foldbench-data](https://huggingface.co/datasets/sunweiwei/ai4sci-foldbench-data).
It contains the actual large payloads, not the Harbor task code. Run the
GitHub task's `materialize.sh DESTINATION` on the host before mounting the
prepared subdirectories. The agent itself does not download data.

`download-source.json` pins the immutable Hub commit and the SHA-256 of
`release-manifest.json`. The latter pins all 75 tar shards and the per-file
inventory. The script checks every archive and all 185,995 restored files,
then calls `validate_release.py` for the canonical manifests, feature coverage
and reference-weight checks. Unpacking is byte-preserving; paths above and
all existing training/evaluation interfaces stay unchanged.

Restored data total 73,068,582,840 bytes (73.1 GB decimal). Allow at least
76 GB free, plus any additional HF cache requirements. Downloaded shards are
removed after successful extraction by default; `--keep-archives` retains
another 73.3 GB. A rerun resumes missing files and verifies existing files;
`--verify-only` checks a completed release without network access. Local
conflicting files are reported instead of overwritten. Use an otherwise
empty destination for the first download.

Publication includes neither ESMC-6B weights nor unquantized feature masters,
MSA payloads, credentials or experiment artifacts. The model weights under
`models/` remain ESMFold2-Fast **reference-only** data under the rules above.
