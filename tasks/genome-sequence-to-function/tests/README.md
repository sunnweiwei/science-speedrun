# Verifier

Two services, both offline (`docker-compose.yaml`).

**`main`** runs the submission as uid 10001. It sees:
- the pinned FASTA;
- `public/test.parquet` and `public/junction_tracks.parquet`;
- `public/perturbations.parquet` (the variant queries).

It runs `infer.sh` and then `perturb.sh` within one shared time budget. It then validates both
outputs: example IDs and order, fixed shapes, dtypes, finite values, sparse-list limits, no
symlinks, and the 80-GiB prediction limit. Finally it copies them into a sealed snapshot with
SHA-256 manifests. An invalid package gets `reward` 0 with `status: invalid_submission`.

**`grader`** never executes submission code. It waits for the sealed snapshot and checks its
hashes. It then scores:
- tracks with `environment/evaluation/scoring.py`: per-track Pearson in `asinh` space, family
  means, and the equal mean across families, which is `reward`;
- variant effects with `environment/evaluation/variant_scoring.py`, reported under
  `variant_effects` with anchors.

Test targets, junction scoring queries, splice positions and variant labels are mounted only here.

It writes `report.json` and `reward.json` (`reward`, `overall_mean_family_pearson`).

Run the source tests with `bash tests/run_unit_tests.sh`; the baseline's tests are in
`environment/baseline/alphagenome-reference/reference/alphagenome_50m/tests`.

`python tests/measure_prediction_size.py` measures the v3 dense output size. That size motivated
the sparse v4 contract: about 636 MB per example, against about 11.5 MB per example under v4.
