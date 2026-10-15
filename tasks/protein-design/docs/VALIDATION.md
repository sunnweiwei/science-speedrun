# Integration validation

Integrator record for PR #52. The task author's own validation is in
[README.md](../README.md) ("Validation") and the measured results are in
[environment/evaluation/RESULTS.md](../environment/evaluation/RESULTS.md); this page
records only what was checked on the integration host, and what was fixed there.

## 2026-09-30, one host with 8 × H100 80 GB and 208 cores

### Checked

1. **Evaluation view.** `environment/data/ingest_targets.py` was run from a clean checkout:
   `enzyme 41 / ppi_binder 5 / sm_binder 4 / unconditional 1` = **51 targets, nothing
   skipped**, and `unconditional.jsonl` carries `uncond_L100` at length 100 with 100
   designs. Every track therefore has requests, so the reward is not capped.
2. **Both images build.** `environment/Dockerfile` 32/32 and `tests/Dockerfile` 16/16,
   from the task directory with no local modification.
3. **Development image contents.** `/workspace` holds `instruction.md`, `contract.md`,
   `data`, `evaluation`, `references`, `starter`, `research`, `submission` and **no
   `tests/`**. The designability weights are baked in — `/assets/rf3` 2.9 GB,
   `/assets/ligandmpnn_params` 17 MB — so the agent can score the real metric offline.
   `research/` and `submission/` are writable.
4. **Verifier image contents.** `/tests` complete, the unprivileged `agent` user at uid
   10001, and `prody` / `gemmi` / `numpy` 2.3.2 import.
5. **Unit tests.** 26 tests pass from a repository checkout, and 26 pass with 1 skipped
   inside the verifier image (the skip is the repository-only invariant in item 2 of
   "Fixed" below).
6. **End to end, with the real scorer.** On a reduced view — one target per track, two
   designs each — a deliberately trivial submission (a straight extended backbone per
   request, declared as a `trained_generator`) was run through `tests/test.sh` and then the
   grader, both networkless on one H100:
   - `main_verifier.py` validated the package, ran it as the unprivileged user, and sealed
     `sealed/run` with `snapshot.json` and a `READY` marker;
   - the grader designed sequences with LigandMPNN and folded them with RoseTTAFold3,
     reporting all eight designs as returned, valid and conditioning-satisfied, and **0
     designable on every track** — `pass_rate` 0.0 for all four. Grading took about 9
     minutes for 8 designs on one GPU.
   - Failures carry their measurements, e.g. `not designable (scRMSD 102.51, ipTM 0.22,
     PAE 15.28, motif 8.78, no sequence satisfied the criterion)`, so the scorer is
     computing every criterion rather than short-circuiting.

   This is a **partial null-system row**, one of the author's open items: a generator that
   models nothing collects exactly zero on all four tracks at this scale, so the reported
   pass rates are not floor artefacts. It is not the full row — that needs the whole
   51-target view.

   The harness contract was also exercised negatively: an earlier version of the same
   package declaring `kind: predictor_inversion` with no pretrained asset was rejected with
   `kind 'predictor_inversion' requires at least one declared pretrained asset`.

   The privilege drop ran: the verifier process is root (uid 0) and the submission's child
   is uid 10001, checked directly in the image. Sealed output and the grader's results stay
   root-owned.
7. **Mounts.** Both compose files validate with the standard variables. In a real
   container the development corpus lands at `/workspace/data/{fit,tune}`, `/data`
   resolves to `/workspace/data`, `/workspace/research` is writable, and the image's own
   `data/README.md` is not shadowed by the mounts.

### Fixed here

1. **Neither image could be built** (blocking). `prody>=2.4` has no cp311 wheel for
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` — `pip download --only-binary :all:`
   resolves to nothing — so pip falls back to the 2.6.1 sdist and compiles C extensions,
   while the apt line installed only `git curl ca-certificates`:
   `error: [Errno 2] No such file or directory: 'gcc'`. Both Dockerfiles hit this.
   `gcc g++` added to both.
2. **`run_unit_tests.sh` failed 1/26 inside the verifier image.**
   `test_starter_and_grader_share_one_atom14_table` compares `tests/residues.py` with
   `environment/baseline/rfd3-small/residues.py`, which only coexist in the repository; the image
   copies `tests/` alone. It now skips when the starter copy is absent, so the suite is
   green both from a checkout and from the image.
3. **The development image had no `/workspace/instruction.md`.** Now copied.
4. **TASK_STANDARD layout adopted.** `TASK_DATA_DIR` / `TASK_PRIVATE_DIR` /
   `TASK_RESEARCH_DIR` replace `PROTEIN_DESIGN_DATA_DIR`; the development corpus is
   mounted at `/workspace/data/{fit,tune}` with `/data` kept as a symlink, because 69
   references across 20+ files use that path and renaming them could not be re-verified
   without the ~100 GB corpus download; `/workspace/research` is now a
   `TASK_RESEARCH_DIR` bind mount; GPU reservations are out of the development compose
   and `network_mode = "no-network"` is declared in `task.toml` for both environments.
5. **`reward.json` has no composite, and that is now declared.** The task reports four
   per-track `pass_rate`s and no aggregate, on the grounds that the tracks carry different
   published criteria and `unconditional` is synthetic to this task. `TASK_STANDARD.md`
   names its exceptions explicitly, so `protein-design` was added to that list.

### Not checked

- **No run on the real 51-target protocol, and no known-answer replay.** The pipeline is
  exercised end to end above, but reproducing the author's numbers needs
  `epoch-0100.ckpt`, which is on the author's cluster. The graded reference row in
  RESULTS.md — enzyme 0.0624, sm_binder 0.0275, ppi_binder 0.002, unconditional 0.420,
  cross-checked against the operator harness and taking 9 h 33 min on 8 × L40S — is
  therefore **accepted from the author's report rather than reproduced here**, as was done
  for virtual-cell-zeroshot, tokamak-magnetic-control and hydrogym. `tests/README.md`
  carries the table a future build can be checked against.
- The author's remaining open items stand, with two narrowed: thresholds uncalibrated for
  RoseTTAFold3; the null-system row is now partial rather than absent (item 6 above), and
  `os.setuid` is now exercised, since the smoke ran the submission as the `agent` user; no
  `predictor_inversion` reference system has been run.
- The audit behind "0 problems over all 50 targets against `input.yaml`" is not committed,
  so that claim is not reproducible from the repository.

## 2026-09-30, format unification

Brought to the TASK_STANDARD document layout after the merge, and re-checked:

- `environment/starter/` renamed to `environment/baseline/rfd3-small/` (15 path references
  updated, including the Dockerfiles' COPY and PYTHONPATH), with a `RESULTS.md` next to its
  README and a `publish.py` that stages and swaps the package in one rename — the verifier can
  snapshot `/workspace/submission` at any moment and the task had no atomic replace.
- `instruction.md` restructured into the five standard sections, `README.md` into the eight.
- `environment/data/MATERIALIZE.md` added.
- `task.toml` authors changed from a bare email to `[{ name = "YanjingLiLi" }]`.

Re-verified after the changes: both images build; the development image carries
`/workspace/baseline/rfd3-small` with the vendored RFdiffusion3 and a resolvable PYTHONPATH,
and no `/workspace/starter`; `publish.py` publishes and republishes over an existing package
with the verifier's own `validate_harness`; unit tests are 26 from a checkout and 26 with 1
skipped from the image.

