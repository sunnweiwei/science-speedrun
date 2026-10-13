# Materializing the virtual-cell-zeroshot release (operators)

Controller-side. The agent never runs these scripts; only `README.md` of this directory is copied
into the image.

## Prepared release (default)

Release v1 is on Hugging Face: `sunweiwei/ai4sci-virtual-cell-zeroshot-data` (public; `dev/` about
169 GB without SE-600M, `private/` about 22 GB; the revision is pinned in `download-source.json`
under `prepared_release`). SE-600M comes from `arcinstitute/SE-600M` at `5a9a80f`.

```bash
bash fetch_release.sh DATA_ROOT            # dev/ + private/ + manifest, then SE-600M into dev/se600m/, then size check
bash fetch_release.sh DATA_ROOT --dev-only # development machines that never grade
python verify_release.py --root DATA_ROOT --hash   # optional full SHA-256 check
```

Then:

```bash
export TASK_DATA_DIR=DATA_ROOT/dev        # entries mounted read-only under /workspace/data
export TASK_PRIVATE_DIR=DATA_ROOT/private # verifier grader only
```

```text
DATA_ROOT/
  manifest.json      operator manifest: which public screen each test context comes from (not mounted)
  dev/               train/ test/ priors/ se600m/ manifest.json splits.json gene_panel.csv gene_panel_intersection.json
  private/           responses of the test contexts, request subsets and strata, cached no-effect and DE tables
```

## Rebuilding from the sources

```bash
bash materialize.sh SOURCE_CACHE DATA_ROOT      # download (if missing), verify, harmonise, build, embed, priors, baselines
```

Stages are idempotent and can be re-run individually with `python -m materialize.cli STAGE ...`;
the `embed` and `baseline` stages run inside the dev/verifier image (GPU, arc-state, cell-eval).
Environment knobs: `VCZ_DEV_IMAGE`, `VCZ_VERIFIER_IMAGE`, `VCZ_DOCKER`, `VCZ_GPUS`, `VCZ_WORKERS`,
`VCZ_SKIP_DOWNLOAD`, `VCZ_STRICT_VERIFY`, `VCZ_MATERIALIZE_PYTHON`. The per-target cell caps are in
`materialize/config.py`; removing them is a one-line change but multiplies the release size.
