# Resource materialization (operator)

Materialize outside the agent container. A release contains only prepared data
and trusted runtime dependencies, split into `development/` and `private/`.
Neither split includes model weights, predictions, experiment logs or ledgers.

The public release is hosted at
[Hugging Face](https://huggingface.co/datasets/sanxing/ai4sci-fruit-fly-brain).
`download-source.json` pins its immutable commit and original release manifest
SHA-256. No HF token is required. Install operator download dependencies in a
Python 3.10+ environment, then run:

```bash
python3 -m pip install -r fruit-fly-neuron-reconstruction/environment/data/requirements.txt
bash fruit-fly-neuron-reconstruction/environment/data/materialize.sh \
  --cache "$TASK_DOWNLOAD_CACHE" \
  --data "$TASK_DATA_DIR" --private "$TASK_PRIVATE_DIR"
```

All three paths must be disjoint. Keep the cache outside agent mounts: it holds
both development and evaluation objects. `private` means hidden from the agent,
not restricted access on HF. The downloader requests anonymous access, resumes
incomplete HF downloads, validates the pinned manifest, and checks the size and
SHA-256 of every materialized file. Existing correct files are retained.
`--verify-only` validates local resources offline using the cached manifest.
The cache can require another 14.54GB; a passed receipt is written to
`TASK_DOWNLOAD_CACHE/receipt.json`. New runs recheck files rather than trusting
a previous receipt. Only a zero exit status establishes successful completion.

The release has 722 logical files / 14,537,257,535 bytes and 692 unique blobs.
HF filenames are content hashes; original paths (including pinned Git version
metadata) are restored from the manifest. This preserves the previously
validated release byte-for-byte. See `docs/validation/huggingface-release.json`
for the independent anonymous download check.

Explicit maintainer exports still support the original local/HTTPS interface:

```bash
bash fruit-fly-neuron-reconstruction/environment/data/materialize.sh \
  --manifest /release/release.json --sha256 "$RELEASE_SHA256" \
  --source /release --data "$TASK_DATA_DIR" --private "$TASK_PRIVATE_DIR"
```

Do not substitute the historical experiment directory for either mount.
Use `export_resources.py --help` to export explicit prepared inputs from the
recorded MaleCNS campaign: train/validation volume blocks, train/validation
SHAPE meshes, validation image/suite, private test image/suite, and curated
runtime. The exporter removes the test split from agent-visible manifests.
Dataset generation itself remains in the source adapters `fetch_malecns.py`,
`prepare_malecns_training.py`, `prepare_malecns_shape.py` and
`prepare_malecns_context.py`; the maintainer records their pinned plans.

Expected output:

- `TASK_DATA_DIR/training/{volume,shape}`: visible training/validation supervision.
- `TASK_DATA_DIR/validation/{image,suite}`: visible development evaluation.
- `TASK_DATA_DIR/runtime`: clean pinned Google FFN/connectomics checkouts and
  FPS binary/source/licenses. The binary must match the target Python/Torch/CUDA
  ABI and GPU architecture. `runtime-manifest.json` hashes its files.
- `TASK_PRIVATE_DIR/{image,suite}`: private inference input and scoring reference.

The same sparse annotations and shared-ID exclusions as the native experiment
are preserved. This repackaging changes visibility and paths, not splits or
labels. A fresh upstream download, target-image FPS ABI testing
and full verifier execution remain separate acceptance gates.

Maintainers can use `package_hf.py RELEASE NEW_STAGING` to validate and package
an export as `blobs/{sha256}` plus `release.json` and the checked-in dataset card.
Retain source licenses and frozen plan documents alongside the upload as in the
published release. Upload the staging directory to a dataset repository, record
the resulting immutable commit in `download-source.json`, then test anonymous
materialization into empty destinations before accepting the release.
