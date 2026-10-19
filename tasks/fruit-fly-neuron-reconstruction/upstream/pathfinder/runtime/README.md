# PATHFINDER runtime environment

Python 3.11, Linux x86_64. `requirements.in` pins direct dependencies already used; `requirements.lock` records resolved dependency versions and distribution SHA-256 values, including PyTorch 2.6.0/CUDA 12.4 wheel dependencies, skeleton preparation, and static plotting. Data and model packages are mounted separately, outside the image.

Create an empty environment in X temp (set `ROOT` to the existing budget root; do not create/reset the GPU ledger):

```bash
export UV_CACHE_DIR="$ROOT/cache/uv-pathfinder"
export TMPDIR="$ROOT/tmp"
uv venv --python /path/to/python3.11 "$ROOT/runtime"
uv pip sync --python "$ROOT/runtime/bin/python" --require-hashes --no-build \
  "$CODE/runtime/requirements.lock"
```

`--no-build` requires wheels, avoiding unpinned build dependencies. Do not point `PYTHONPATH` at dependency directories from an older environment. CPU job 90000022 successfully installed into a new `portable-runtime-v1` environment; all 81 CPU tests passed, and full model package v4 passed strict loading/FPS import. GPU forward execution was not performed.

Two image recipes are provided, both using `fruit-fly-brain/` as the build directory:

```bash
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer-native"
export APPTAINER_TMPDIR="$ROOT/tmp"
apptainer build --fakeroot --ignore-fakeroot-command --mksquashfs-args '-processors 2' \
  "$ROOT/images/pathfinder.sif" upstream/pathfinder/runtime/pathfinder.def
# Where a Docker build service is available:
docker build -f upstream/pathfinder/runtime/Dockerfile -t pathfinder-runtime .
```

Actual Apptainer builds and tests run as Slurm CPU jobs; all caches, build temporary files, and SIFs must stay in X temp. The Python base image pins a Linux amd64 manifest digest; wheel installation requires hashes. Debian apt archives are not pinned to a snapshot; the Apptainer image records actual `installed-os.txt` and `installed-python.txt`, without promising byte-identical builds. Record the final SIF SHA.

The FPS runtime binary comes from the model asset package, which also includes source. Recompilation requires a compatible CUDA toolchain/compiler; see actual rebuild records in the parent RUNTIME.md. This image provides a runtime, without a CUDA compilation toolchain.

This user has no subuid mapping on the current cluster; builds use an ordinary user's root-mapped namespace. Job 90000023 failed because host fakeroot required GLIBC 2.38, unavailable in the Debian base image. Job 90000024 bypassed host fakeroot but failed when apt tried to switch to unmapped user `_apt`. The current recipe uses `--ignore-fakeroot-command` during builds and sets `APT::Sandbox::User=root` only on container apt commands, retaining ordinary host-user privileges. This follows [Apptainer fakeroot documentation](https://apptainer.org/docs/user/latest/fakeroot.html#fakeroot-inside-def), without modifying host user mappings, global apt configuration, or runtime scoring isolation.

Image v1 supplies the verified dependency environment; source inside it is a build-time snapshot. Mount latest source read-only at `/code` and execute `/code/pathfinder/...`. Provenance recording without Git was corrected after the build: the new common.py allows model-asset export in isolated CPU execution, as recorded in parent RUNTIME.md and `receipts/isolated-model-export.json`. The older in-image export helper requires explicit `UPSTREAM_SOURCE_COMMIT` or execution from the new source mount; the old snapshot has not been updated in place.

Official candidate decision-point tooling also requires `absl-py`, `dataclasses-json`, and `pandas`, now added to the lock. Historical portable-runtime-v1 and image v1 lack these dependencies; earlier CPU checks did not cover full official decision-point imports. Running job environments remain unchanged; supplemental dependencies are verified in a separate Xtemp directory before use in new frozen jobs.
