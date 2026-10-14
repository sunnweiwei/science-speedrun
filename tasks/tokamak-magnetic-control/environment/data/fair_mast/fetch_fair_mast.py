"""Fetch the FAIR-MAST subset used to ground the randomization ranges (docs/FIX_PROPOSAL.md, fix 3).

Stage `summary`: the per-shot metadata table of every MAST shot (https://mastapp.site/json/shots),
including the CPF summary scalars (beta_p, li, q95, elongation, minor radius, Ip, NBI power).
Stage `efit`: per-shot EFIT time traces (level-2 `equilibrium` group) for the plasma shots in the
summary, read anonymously from the public S3 endpoint and saved as one NetCDF file per shot.

Usage:
  python fetch_fair_mast.py summary --out raw
  python fetch_fair_mast.py efit --out raw [--max-shots N] [--workers 8]
  python fetch_fair_mast.py vloop --out raw [--workers 8]   # Ip and loop voltage only
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://mastapp.site/json/shots"
S3_ENDPOINT = "https://s3.echo.stfc.ac.uk"


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return json.load(response)
        except Exception as error:  # transient API errors: back off and retry
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} after {error}", file=sys.stderr)
            time.sleep(2 ** attempt)


def fetch_summary(out):
    """Page through the shots endpoint (the server caps pages at 50 rows)."""
    rows, cursor = [], None
    while True:
        query = {"limit": 50}
        if cursor:
            query["cursor"] = urllib.parse.unquote(cursor)
        page = get_json(API + "?" + urllib.parse.urlencode(query))
        rows += page["items"]
        cursor = page.get("next_page")
        if len(rows) % 1000 < 50:
            print(f"{len(rows)} / {page.get('total')} shots", flush=True)
        if not cursor or not page["items"]:
            break
    path = out / "shots_summary.json"
    path.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} shots to {path}")


# EFIT quantities for fix 3: plasma state ranges and traces, plus sensor fit residuals for noise levels
EFIT_VARS = ["ip", "ip_measured", "beta_pol", "beta_tor_normal", "li", "q95", "q_axis", "elongation",
             "triangularity_upper", "triangularity_lower", "minor_radius", "geometric_axis_r",
             "geometric_axis_z", "magnetic_axis_r", "magnetic_axis_z", "wmhd",
             "b_field_pol_probe_measured", "b_field_pol_probe_reconstructed",
             "flux_loop_measured", "flux_loop_reconstructed"]


# Loop voltage for the plasma-resistance range (stage `vloop`, added after the first run)
VLOOP_VARS = ["ip", "vloop_static", "vloop_dynamic"]

_LOOP = None  # one event loop per worker process: s3fs caches its client on the first loop it sees


def fetch_one_efit(args):
    """Read the selected arrays of one shot's level-2 `equilibrium` group (Zarr v3) concurrently."""
    global _LOOP
    import asyncio
    import xarray as xr
    import zarr.api.asynchronous as azarr
    from zarr.storage import FsspecStore

    shot, target, variables = args
    path = Path(target) / f"{shot}.nc"
    if path.exists():
        return shot, "cached"
    storage = {"anon": True, "client_kwargs": {"endpoint_url": S3_ENDPOINT}}

    async def read():
        store = FsspecStore.from_url(f"s3://mast/level2/shots/{shot}.zarr/equilibrium",
                                     storage_options=storage, read_only=True)
        group = await azarr.open_group(store=store, mode="r", zarr_format=3)
        names = [n for n in variables + ["time"] if await group.contains(n)]
        arrays = await asyncio.gather(*(group.getitem(n) for n in names))
        values = await asyncio.gather(*(a.getitem(...) for a in arrays))
        return {n: (tuple(a.metadata.dimension_names), v) for n, a, v in zip(names, arrays, values)}

    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
    try:
        data = _LOOP.run_until_complete(read())
        time_axis = data.pop("time")[1]
        ds = xr.Dataset(data, coords={"time": time_axis}, attrs={"shot_id": shot, "source": "FAIR-MAST level2 equilibrium (EFIT)"})
        ds.to_netcdf(path.with_suffix(".tmp"))
    except Exception as error:
        if type(error).__name__ == "GroupNotFoundError":  # shot archived without an EFIT run
            return shot, "no_efit"
        return shot, f"failed: {type(error).__name__}: {error}"[:200]
    path.with_suffix(".tmp").rename(path)
    return shot, "ok"


def fetch_efit(out, max_shots, workers, subdir="efit", variables=None):
    import multiprocessing as mp

    rows = json.loads((out / "shots_summary.json").read_text())
    # plasma_max_current is in kA and negative for reversed-current shots
    shots = sorted(r["shot_id"] for r in rows if abs(r.get("plasma_max_current") or 0) > 100)
    if max_shots:
        shots = shots[:max_shots]
    target = out / subdir
    target.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(target), list(variables or EFIT_VARS)) for s in shots]
    counts = {}
    # separate processes: fsspec funnels every thread of one process through a single I/O loop
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (shot, status) in enumerate(pool.imap_unordered(fetch_one_efit, jobs), 1):
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed" or i % 250 == 0 or i == len(shots):
                print(f"[{i}/{len(shots)}] {shot}: {status} | totals {counts}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["summary", "efit", "vloop"])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "raw"))
    parser.add_argument("--max-shots", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        fetch_summary(out)
    elif args.stage == "efit":
        fetch_efit(out, args.max_shots, args.workers)
    else:
        fetch_efit(out, args.max_shots, args.workers, subdir="vloop", variables=VLOOP_VARS)


if __name__ == "__main__":
    main()
