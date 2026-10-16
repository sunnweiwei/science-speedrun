# Materializing the HydroGym release (operators)

Controller-side. The agent does not run these steps. Unlike most tasks, the whole directory is
copied into the image (`/workspace/data` and `/opt/hydrogym-task/data`), because the evaluator
imports `public_release.py` and reads the release descriptors from its sibling `data/` directory.

## Materialize

Install the Hugging Face `hf` CLI in the **operator's** preparation environment.
From the repository root:

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
# Default: inspect selected sizes and pinned commands; downloads nothing.
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2
# Download only the selected directories, then verify every Git/LFS hash:
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2 --download
# Full offline integrity recheck:
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2 --verify-only
```

Existing bytes can be reused on the same filesystem without a second 454 GB copy:

```bash
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-public-research-v2 \
  --from-local /existing/resource-root
```

`/existing/resource-root/environments/CASE` must contain the pinned original
files. The helper uses hardlinks, verifies every byte, and refuses replacement
of conflicting files. Keep both views immutable. Do not bind retired/incomplete
historical views as the current release.

Expected prepared layout:

```text
hydrogym-public-research-v2/
  release.json
  materialized.json
  environments/
    Cylinder_2D_Re200/...
    ...                         # all 39 selected directories
```

Set `TASK_DATA_DIR` to that root before Harbor starts. Both native Compose
definitions bind it read-only (development: `/workspace/data/release`, linked from `/data`; verifier: `/data`); missing roots fail instead of creating
empty directories. Download/preparation is explicit, not an automatic Harbor
hook. Credentials stay with the operator. No local author-machine path is required.

The materialization receipt verifies resource identity, **not model quality**.
Its historical `benchmark_validated=false` is deliberately not converted into a
scientific claim. Runtime preflight checks that release identity, selected files,
configuration hashes and frozen adapter bindings match. No `suite.json` is
required by the current physical evaluator.

## Hugging Face data hosting

**GitHub holds the Harbor task; Hugging Face holds the actual large data.**
The public, ungated
[sunweiwei/ai4sci-hydrogym-research-v2](https://huggingface.co/datasets/sunweiwei/ai4sci-hydrogym-research-v2)
repository contains `environments/CASE/...`: all **558 files and
453,756,315,982 bytes**, plus the data card and frozen release manifest. It does
not contain the Harbor task, starter or evaluator. Those remain in this Git
repository. No download token is required.

The materializer's default commands use the exact commit in
`download-source.json` and recreate the expected local `environments/` layout.
It never substitutes floating `main`. The mirror's complete membership, sizes
and hashes were verified against the frozen original source; mesh/restart bytes,
task membership, runtime profiles and metrics are unchanged.

`release.json`, `source-files.json` and the receipt's original `repo_id`/`revision`
retain the authors' source provenance. The actual delivery repository/revision
is separately recorded in `materialized.json.download_source`. Existing verified
local datasets remain valid without redownload. Archived selections and broader
inventory diagnostics that are not in this mirror retain the original upstream
download path; they are not part of the current task's training data.

An earlier code/manifest-only HF publication was incorrect and has been removed
from the current Hub tree. Do not use that historical revision as the data release.

