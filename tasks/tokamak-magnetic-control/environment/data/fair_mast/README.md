# FAIR-MAST subset for data-grounded ranges

This folder holds the MAST data for fix 3 of `docs/FIX_PROPOSAL.md`: realistic ranges and time traces for the plasma parameters, and real sensor-noise levels.
The data come from FAIR-MAST (UKAEA), the public archive of the MAST campaigns M5-M9, licensed CC BY-SA 4.0.
Anything derived from them and committed to the repository needs attribution and the same license.
MAST is MAST-U's predecessor with a different divertor, so use the data as proxies for core-plasma ranges and sensor noise, not as MAST-U values.

## Contents

| Path | Content | Size |
| --- | --- | --- |
| `download_dataset.sh` | The full download and derivation, as run on 2026-10-02/03 | - |
| `fetch_fair_mast.py` | Download script (stages `summary`, `efit`, `vloop`) | - |
| `raw/shots_summary.json` | Metadata and CPF summary scalars of all 15,969 MAST shots (281 fields per shot) | 152 MB |
| `raw/efit/<shot>.nc` | Level-2 EFIT (`equilibrium` group) of 11,052 plasma shots: 16 scalar time traces and measured/reconstructed values of 78 pickup probes and 46 flux loops | 2.8 GB |
| `raw/efit_fetch.log` | Per-shot download status | - |
| `raw/efit_empty_shots.json` | 68 shots whose EFIT file has a time axis but no data | - |
| `raw/vloop/<shot>.nc` | `ip`, `vloop_static` and `vloop_dynamic` of the same 11,052 shots | 140 MB |
| `derive_ranges.py` | Statistics the task uses: beta_p and li_3 spreads and drifts, pressure crashes, plasma resistance, sensor residuals | - |
| `derived_ranges.json` | Its output (committed; CC BY-SA 4.0 like its source) | 4 kB |
| `calibrate_profiles.py` | Maps the task's FreeGSNKE profile parameters to beta_p and li_3 around the base case (needs FreeGSNKE) | - |
| `profile_calibration.json` | Its output (committed) | 2 kB |

`raw/` and `.venv/` are git-ignored, and the task's `.dockerignore` keeps this whole folder out of the Docker build context, so nothing here is sent to an image build.
That matters beyond size: the sealed plant's ranges are derived from these files, so the agent's image must not carry them.

The scalar traces are `ip`, `ip_measured`, `beta_pol`, `beta_tor_normal`, `li`, `q95`, `q_axis`, `elongation`, `triangularity_upper`, `triangularity_lower`, `minor_radius`, `geometric_axis_r`, `geometric_axis_z`, `magnetic_axis_r`, `magnetic_axis_z` and `wmhd`.
The sensor arrays are `b_field_pol_probe_measured`, `b_field_pol_probe_reconstructed`, `flux_loop_measured` and `flux_loop_reconstructed`; their differences give EFIT fit residuals as a noise proxy.

## Coverage

Plasma shots are those with peak plasma current above 100 kA in the summary table: 11,569 of 15,969.
Of these, 10,984 have usable EFIT data, 68 have an empty EFIT run, 514 have no EFIT group in the archive, and 3 (12079, 13761, 13763) have an EFIT group without a time axis.
10,853 shots have at least 20 valid EFIT time slices, with a median of 73 valid slices per shot.
The EFIT peak plasma current has a median of 734 kA (5-95%: 431-927 kA), close to the task's 620 kA base case.

## How the task uses them

`tokagym/env.py` (visible randomization, `BASE_PLASMA_RESISTIVITY`) and `tests/sealed/perturbations.json` cite these files.
The visible plant spans the interquartile range of double-null flat-tops at 550-850 kA (997 shots; 863 with loop voltage), the sealed plant their 5-95% range.
beta_p and li_3 spreads become axis-pressure and alpha ranges through `profile_calibration.json` (d ln beta_p / d ln paxis = 1.16, d li_3 / d alpha_n = 0.36, d li_3 / d alpha_m = -0.17).
The base plasma resistivity, 3.8e-7 Ohm m, reproduces the median flat-top resistance of 1.73 uOhm (loop voltage over current); the task's former 1e-6 Ohm m gave 4.51 uOhm.
Sealed drift rates span the within-shot interquartile range over one 0.35 s episode, and the pressure-crash rate (4.2 per second) and depths (13-27% of the axis pressure) come from persistent 5 ms drops of beta_p.
Sensor noise and offsets come from EFIT fit residuals, which include model error and therefore bound the diagnostic noise from above.
The base case sits at the low end of MAST: its beta_p (0.19) and li_3 (0.71) are near the 5th percentile of these shots, so the ranges are applied as relative spreads around it.

## Running

`bash download_dataset.sh` repeats the original runs: summary table, pinned virtual environment in `.venv/`, EFIT stage, empty-file list, loop-voltage stage and `derive_ranges.py`.
`calibrate_profiles.py` needs FreeGSNKE and runs separately (`python environment/data/fair_mast/calibrate_profiles.py` from the task directory, about 1 minute).
The summary stage needs only the Python standard library and takes about 10 minutes, because the API returns 50 shots per page.
The EFIT stage needs Python 3.11 or newer with `xarray`, `zarr>=3`, `s3fs` and `netCDF4`, and took about 55 minutes with 32 processes.
It is resumable: existing files in `raw/efit/` are skipped.

## Data notes

- `plasma_max_current` in the summary table is in kA and negative for reversed-current shots.
- Some summary fields hold corrupt values (for example a minor radius of 11,384 m), so filter before computing ranges.
- The API's numeric filters return server errors, so the script downloads the full table and filters locally.
- Level-2 shots are Zarr v3 without consolidated metadata, and the archive is anonymous S3 at `https://s3.echo.stfc.ac.uk`, bucket `mast`.
- The script reads each shot with one process and concurrent requests, because fsspec routes all threads of a process through one I/O loop.
