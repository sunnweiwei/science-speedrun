# Isolated final verification

CrystalGen final evaluation uses two trust domains.

`main` receives the submission and input-only requests. It validates and copies
the package into a root-owned read-only replay tree, runs `generate.sh` from
that snapshot as UID 10001 with no network, kills remaining submission processes,
and seals a hash-bound copy of the output. It does not mount MP-20, LeMat-Bulk,
hull tables, or MLIP weights.

`grader` receives the sealed snapshot, requests, exact pinned LeMat scorer, and
operator-prepared scoring assets. It never mounts, imports, or executes the
submission. The authoritative orchestration lives in `grader.py`; shared metric
code under `scoring/` is byte-identical to the copy exposed for development.
The shared pipeline first relaxes a copy using frozen UMA `uma-s-1` (`omat`),
BFGSLineSearch + FrechetCellFilter, `fmax=0.02` eV/Å, 200 steps and a 300-second
per-candidate limit. It then scores those geometries with the unchanged LeMat
ensemble. The sealed raw outputs are never modified. Final grading has no
submission-controlled opt-out or recipe override.

Failed refinements keep original bytes; finite unconverged iterates are retained
and flagged. Relaxed CIFs, hashes and per-sample diagnostics are saved under
`results/relaxation/`; `report.json` v2 records the recipe, counts and runtimes.
Fixture-only unit tests explicitly report that relaxation was not run.

Final production suites contain exactly one 2,500-sample unconditional request.
The generation seed is prepared outside Git. No test CSV or conditional targets
are needed. The v1 suite manifest rejects old mixed suites. See `requests/README.md`.

Missing or invalid predictions remain in the denominator. Infrastructure or
asset failures exit nonzero rather than masquerading as a model score.

## Local unit tests

From the repository root:

```bash
python3 -m unittest discover -s crystalgen/tests -p 'test_*.py' -v
```

These use synthetic files and fixture scorer results. They do not download
MP-20, load an MLIP, train OMatG, or establish a scientific score.

## Full verifier

Set the operator root prepared as in `environment/data/MATERIALIZE.md`:

```bash
export TASK_PRIVATE_DIR=/secure/crystalgen   # final-suite/ is mounted at /benchmark
export TASK_DATA_DIR=/secure/crystalgen      # scorer-assets/ is mounted into the grader only
```

Then invoke the Harbor verifier or the Compose definition in this directory.
A package that fails validation, exits nonzero or exceeds the inference time
limit gets `{"reward": 0.0, ...}` with the reason; the grader then exits
without scoring. Otherwise `reward` equals `unconditional_sun_metasun`.
