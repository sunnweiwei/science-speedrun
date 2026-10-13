# Materializing the MassSpecGym data (controller and maintainers)

This file is not copied into the agent's workspace. `TASK_DATA_DIR` is the public materialization;
its `development/` entries are mounted under `/workspace/data`, and `verifier/` is mounted in the
verifier. `TASK_PRIVATE_DIR` is the protected label directory, mounted only into the trusted grader.

## Prepared Hugging Face release (default)

Download the completed task views directly; rebuilding source preprocessing is
optional. The public dataset is `Corning/ai4sci-massspecgym-retrieval`; labels
are in private `Corning/ai4sci-massspecgym-retrieval-evaluation`, accessible only
to the benchmark operator. Full commit SHAs and release-manifest SHA-256 values
are frozen in `hf_release.json`. Use existing Hugging Face authentication with
read access to the private dataset. Never mount that credential or the private
label directory into development or model inference.

```bash
bash mass-spectrometry-molecule-retrieval/environment/data/materialize.sh \
  /absolute/path/to/public-data /absolute/path/to/grader-labels
```

This default `prepared` mode requires Python and `huggingface_hub==1.32.0`;
it does not require RDKit, the GLACIER runtime, source TSVs or Zenodo access.
It downloads the unchanged 17 prepared public files (**38,019,369,914 bytes**)
and two grader files (**3,810,324 bytes**), plus small release manifests. Most
bytes are the lossless 36,743,610,368-byte synthetic SQLite database. Allow at
least 39 GB free for the destination. HF metadata and blobs remain in an
operator-only sibling `.massspecgym-hf-cache` directory (mode 0700), outside
all model-visible data mounts. Only declared payload files enter atomic staging;
same-filesystem hardlinks avoid a second physical data copy. On filesystems
without hardlink support the checked copy fallback needs up to another 38 GB.
The persistent cache is not deleted during materialization, avoiding open-file
cleanup races on NFS. Network transfer duration depends on the connection. The downloader checks full pins,
complete file membership, sizes, SHA-256, nested manifests and split counts.
Existing complete releases are reverified and reused offline; an incomplete or
unrelated destination is never overwritten. Failed fresh downloads do not
publish a partial destination. Public and grader-label roots must be separate.

An operator runs this helper with network access before the agent budget. The
helper scopes `HF_HUB_OFFLINE=0` to its own process; the training/verifier images
remain offline. Harbor does not run materialization automatically. Development
mounts only `public-data/development`, inference only `public-data/verifier`,
and the trusted grader alone mounts `grader-labels`, as already declared in the
compose files. Keeping historical test labels in a private distribution does
not make the upstream public benchmark a secret evaluation set.

For provenance or a deliberate rebuild, select source mode explicitly:

```bash
MASSSPECGYM_MATERIALIZE_MODE=source \
  bash mass-spectrometry-molecule-retrieval/environment/data/materialize.sh \
  /absolute/path/to/rebuilt-public /absolute/path/to/rebuilt-labels
```

The source path retains the original pinned upstream files, sanitation and MAGMa
steps below. The HF command names both exact files explicitly, and MAGMa runs
only after core data exist. The previously merged materializer dependency/order
fixes are retained. Source rebuilding needs the heavier documented runtime.

Maintainers generate a release manifest without copying data:

```bash
python3 mass-spectrometry-molecule-retrieval/environment/data/release_manifest.py build \
  --root /path/to/prepared/public --component public \
  --manifest /tmp/public-release-manifest.json
```

Repeat with `--component private` for grader labels. Upload each manifest as
`release-manifest.json` beside its unchanged payload, retain MIT attribution,
then update both full revisions and manifest digests in `hf_release.json`.
Publish no checkpoints or workspace caches with these datasets.

## Source reconstruction

Run the materializer from the repository root inside the development image
with network access (it provides the `hf` CLI, `curl`, RDKit and the native
GLACIER runtime that the MAGMa preparation step needs); a host environment
must supply the same tools plus `MASSSPECGYM_GLACIER_PYTHON` /
`MASSSPECGYM_GLACIER_SOURCE`:

```bash
export TASK_DATA_DIR=/absolute/path/on/large-disk/massspecgym-retrieval
export TASK_PRIVATE_DIR=/absolute/path/on/protected-disk/massspecgym-retrieval-labels
MASSSPECGYM_MATERIALIZE_MODE=source \
  bash mass-spectrometry-molecule-retrieval/environment/data/materialize.sh \
  "$TASK_DATA_DIR" "$TASK_PRIVATE_DIR"
```

Set `MASSSPECGYM_CACHE_DIR` to choose where the two pinned source files are
stored and `MASSSPECGYM_AUGMENTATION_CACHE_DIR` to place the 6.5 GB Zenodo
archive on a large disk. The helper checks all pinned digests and never
overwrites a prepared output. On a repeated run it reuses the public/private
pair only after validating its manifests and file hashes; it likewise verifies
every augmentation file before reusing it. It packs the 2.33 million tiny JSON
files into a lossless SQLite lookup, removes held-out overlaps and invalid
structures, and runs a second split audit. Allow roughly 45 GB total for the retained
archive, staged output, and final augmentation (or about 38 GB beside the
archive when the cache is on another filesystem), and install the packages in
`environment/references/requirements-replay.txt` in the operator environment.

For a quick core-data/interface smoke test only, set
`MASSSPECGYM_PREPARE_SYNTHETIC=0` and `MASSSPECGYM_PREPARE_GLACIER=0`.
Such an output deliberately fails the full
development-container health check and is not a complete task materialization.

The Python materializer temporarily loads the 454 MB candidate dictionary in
memory; allow several GiB of host RAM. It converts answer-keyed upstream
candidates into answer-independent pool IDs and deterministic candidate order.
The validated pinned release produced approximately 533 MiB of public prepared
data and 3.7 MiB of private labels (source cache approximately 683 MiB).

