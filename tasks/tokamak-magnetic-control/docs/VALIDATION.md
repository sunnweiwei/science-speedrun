# Validation records

The results tables (sealed references, deployable PID, RL starter, learning curves) are in the
README's section 6 and in `environment/baseline/*/RESULTS.md`.

## Layout check after TASK_STANDARD.md (2026-09-27)

The starter, the PID reference and package, and the hold-current anchor moved to
`environment/baseline/{starter,pid,hold_current}/`; the Nature 2022 notes stayed in
`environment/references/`; a writable `/workspace/research` mount was added and the network
setting moved to `task.toml`. Checked on one host with rebuilt images and the task's own compose
files:

- the development container started healthy; the environment smoke stepped the nonlinear plant
  (shape RMS 1.9 cm over ten steps of `mastu_vertical_shift`);
- `init_random.py` published into the image-layer `/workspace/submission` and then again over it
  (atomic replacement, no leftovers); the moved PID modules and `train.py` import from their new
  places;
- the verifier (`tests/docker-compose.yaml`, `bash /tests/test.sh`, CPU) replayed that random-init
  package on the sealed seeds of all six cases in 27 min: `reward.json` has feasible fraction 0.0 on
  every case, as expected for an untrained actor;
- 11 unit tests pass.

## v0.2 checks (2026-10-03, the verifier's review)

Run on an 8-core laptop with FreeGSNKE 3.1.0 installed by pip (Docker was not available, so the
images were not rebuilt):

- `python -m unittest discover -s tests` passes 19 tests (metrics, success tolerances, steady-state
  window, void rule, verifier aggregation, protocol); `scripts/audit_tasks.py tokamak-magnetic-control`
  conforms.
- All seven cases build (including `mastu_sequence` and its second shape) and 21 sealed resets
  (seeds 1001-1003) succeed at the first attempt, with no fallback to nominal parameters.
- Every plant mechanism was stepped with extreme settings (`STRUCTURAL_DEFAULTS` on), and the
  beta_p-constrained profile, crashes, drifts, dead channels and pre-roll were seen to act.
- `harbor_verify.py` replayed the hold-current package on all seven cases, sealed seeds 1001-1002:
  14/14 `plasma_lost`, `reward` 0.0, `policy_compute_ms` 0.13 (`baseline/hold_current/RESULTS.md`).
- The privileged PID was run on the sealed plant three times while fixing the task:
  - with the ±10 kA solenoid limit, every episode ended at `coil_limit` after about 0.13 s, which led
    to MAST-U's ±55 kA range;
  - without X-point-height feedback it kept the plasma but missed 1 cm on the X-point height
    (several cm) and the 32-point RMS; on visible seed 0 of `mastu_diverted_hold` adding the
    feedback took the shape error from 5.63 to 0.61 cm (a success);
  - with that feedback, seeds 1001-1002 of all cases: `baseline/pid/RESULTS.md`.
- A seed-1002 diagnosis showed the earlier silent kills were control failures: the plasma drifted
  away (6.5 cm by 280 ms) until the solver slowed and stalled, not a reset or reuse problem.
- FreeGSNKE's "object has no attribute 'ny'" failures come from `Jtor_universal.copy` dropping `ny`;
  `tokagym.env.patch_freegsnke` adds it.
