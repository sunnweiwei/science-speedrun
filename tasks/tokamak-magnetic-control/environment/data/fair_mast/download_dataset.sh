#!/usr/bin/env bash
# Download the FAIR-MAST subset into raw/ next to this script and derive the task's statistics (see README.md);
# the same commands as the 2026-10-02/03 runs. Needs python3 >= 3.11 (the runs used 3.12.4), about
# 3 GB of disk and about 90 minutes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw

# 1. Per-shot summary table of all MAST shots (standard library only, ~10 min).
python3 fetch_fair_mast.py summary --out raw

# 2. Environment for the EFIT stage, pinned to the versions of the run.
python3 -m venv .venv
.venv/bin/pip install -q "xarray==2026.9.0" "zarr==3.4.0" "s3fs==2026.9.0" "fsspec==2026.9.0" \
    "netCDF4==1.7.4" "numpy==2.5.3" "pandas==3.0.6"

# 3. Level-2 EFIT traces of every plasma shot, 32 processes (~55 min, resumable).
.venv/bin/python fetch_fair_mast.py efit --out raw --workers 32 > raw/efit_fetch.log 2>&1
grep -v -i warning raw/efit_fetch.log | tail -1

# 4. List the shots whose EFIT file has a time axis but no data.
.venv/bin/python - <<'EOF'
import glob, json
import xarray as xr

empty = []
for path in sorted(glob.glob("raw/efit/*.nc")):
    with xr.open_dataset(path) as ds:
        if "beta_pol" not in ds:
            empty.append(int(path.split("/")[-1][:-3]))
json.dump({"empty_efit_shots": empty}, open("raw/efit_empty_shots.json", "w"))
print(f"{len(empty)} empty EFIT files listed in raw/efit_empty_shots.json")
EOF

# 5. Ip and loop voltage of every plasma shot, for the plasma-resistance range (~15 min, resumable).
.venv/bin/python fetch_fair_mast.py vloop --out raw --workers 32 > raw/vloop_fetch.log 2>&1
grep -v -i warning raw/vloop_fetch.log | tail -1

# 6. Derived statistics used by the task (writes derived_ranges.json, ~30 s).
.venv/bin/python derive_ranges.py --workers 4 > /dev/null
du -sh raw
