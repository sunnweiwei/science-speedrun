# Independent closed-loop evaluation

`harbor_verify.py` is both the development evaluator and the Harbor verifier
entry (`tests/test.sh`). It starts the candidate's resident `infer.sh`, builds each
case's perturbed plant, sends observations, validates voltages, advances FreeGSNKE
with the nonlinear solver and computes the fixed metrics in [METRICS.md](METRICS.md).
It never loads candidate-reported scores.

## What is fixed

- Machine description, grid and limits: `tokagym/machine.py` (vendored FreeGSNKE
  MAST-U-like description, commit in `data/machine_configs/FREEGSNKE_COMMIT`).
- Cases, horizons (700 x 0.5 ms; 1200 for `mastu_sequence`), targets and waveforms:
  `tokagym/cases.py`.
- Visible randomization and plant mechanisms: `tokagym/env.py`.
- Observation/action definition: `contract.md`.
- Success tolerances, steady-state window, outcomes, latency gate: `METRICS.md`.
- Sealed perturbations and seeds: `tests/sealed/perturbations.json` (verifier image only).

## Development commands

Inside the development image (`PYTHONPATH=/workspace`):

```bash
# No simulator: does the package speak the protocol? (one case, one seed, visible draws)
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/check-001 --cases mastu_diverted_hold --seeds 0 --workers 1

# Full visible-randomization suite, 3 seeds: all 21 episodes in parallel when CPUs allow
# (default workers = available CPUs / OMP_NUM_THREADS; override with --workers or TOKAGYM_WORKERS)
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/full-001

# Environment/plumbing smoke without a policy
python /workspace/evaluation/smoke_env.py --cases mastu_diverted_hold --steps 20
```

Each worker holds one FreeGSNKE simulator, one resident policy process and one episode
(one case and seed). A nonlinear step costs ~0.3-0.5 s on four CPU threads (up to about
1 s with the time-varying sealed plasma), so one 700-step episode is about four minutes;
the seven-case, three-seed development suite takes ~20 minutes with eight workers. Use a
fresh output directory each time.

## Clean-container verification

```bash
docker build --target verifier -t tokamak-verifier -f environment/Dockerfile .
docker run --rm --network none --cpus 24 \
  --mount type=bind,source=/abs/published-model,target=/workspace/submission,readonly \
  --mount type=bind,source=/abs/output,target=/logs/verifier \
  --entrypoint bash tokamak-verifier /tests/test.sh
```

## Outputs and failure semantics

- `reward.json`: `reward` (success fraction), `policy_compute_ms` and flat numeric
  `CASE/metric` values with the directions of METRICS.md.
- `results.json`: every case and episode, per-episode summaries, setup draws,
  latency diagnostics, randomization name, seeds, versions.
- `grading.json`: `scope` (`full-suite` or `development-subset`), case status and
  outcome counts, `aggregate_score` (= `reward`).
- `error.json`: present only for infrastructure errors (including more than 20% void
  episodes); then `reward.json` is withheld and the exit code is 2.

Each episode runs in a supervised child process (`supervise.py`); an episode that exceeds
`TOKAGYM_EPISODE_TIMEOUT` (default 5400 s, for the 1200-step sequence case), or a process
that completes no step for `TOKAGYM_STEP_SILENCE` seconds (default 600 s; the child sends a
heartbeat with the current shape and current errors after every step), is killed and
recorded as `solver_timeout`. The heartbeats decide whether it is a `solver_void` (the
plasma was within tolerance) or a `solver_failure` (METRICS.md). FreeGSNKE can spend unbounded time in a degenerate
Newton-Krylov solve and swallows in-process timeouts, so this external guard is what
keeps a bad policy from hanging the verifier. Both limits are generous relative to the
~0.6-3 s a nonlinear step takes on a lightly loaded machine; on an overloaded host
(we saw >24 s per step with ~200 busy threads on 208 cores) the silence guard can fire
on a healthy episode, which is why the reference tables were re-run for episodes killed this way.
Size workers to the CPUs you actually hold: the default (available CPUs / `OMP_NUM_THREADS`) does.
`tests/test.sh` runs the 35 sealed episodes with 35 workers on the verifier's 40 CPUs: one wave in
53 min, no void or silence-guard episodes, the documented 17/35 for `shape_vc`.

On every step the child also sends the same observation to the null policy (`null_policy/`),
next to the submission; the median of the per-step differences of the two round trips is the
episode's `policy_compute_ms` for the latency gate. Timing the null policy only before the episode,
on an idle process, charged the submission for the solver's load on the machine: a hold-current
package (twelve multiplications) measured 0.30-0.63 ms on a loaded 8-worker host. The order of the
two requests alternates every step: whichever process answers first after a solver step pays the
wake-up and cold-cache cost (with the submission always first, the hold-current package still
measured 0.22 ms and a 0.06 ms NumPy controller 0.45 ms).

A plasma lost by the policy, a coil-limit termination, a solver failure or timeout during
a policy-driven episode, or a protocol/bounds violation is a **measured outcome**, not an
infrastructure error. A `policy_error` in any case, or a median policy computation above
0.5 ms, makes the package invalid: `reward` is 0.
