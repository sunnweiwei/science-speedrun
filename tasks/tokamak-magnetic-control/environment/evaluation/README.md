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
# (default workers = (CPUs of the container quota - 2) / OMP_NUM_THREADS, at most one per episode;
#  override with --workers or TOKAGYM_WORKERS)
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/full-001

# Environment/plumbing smoke without a policy
python /workspace/evaluation/smoke_env.py --cases mastu_diverted_hold --steps 20
```

Each worker holds one FreeGSNKE simulator, one resident policy process and one episode
(one case and seed). A nonlinear step costs ~0.3-0.5 s on four CPU threads (up to about
1 s with the time-varying sealed plasma), so one 700-step episode is about four minutes;
the seven-case, three-seed development suite takes ~20 minutes with eight workers. Use a
fresh output directory each time. The default suite is a starting point: with 21 episodes one episode is about five
points, so any seeds can be added (`--seeds`) when two candidates are close.

## Clean-container verification

```bash
docker build --target verifier -t tokamak-verifier -f environment/Dockerfile .
docker run --rm --network none --cpus 112 \
  --mount type=bind,source=/abs/published-model,target=/workspace/submission,readonly \
  --mount type=bind,source=/abs/output,target=/logs/verifier \
  --entrypoint bash tokamak-verifier /tests/test.sh
```

## Outputs and failure semantics

- `reward.json`: `reward` (success fraction over all episodes), `tolerance_time_fraction`,
  `policy_compute_ms`, retry counts and flat numeric `CASE/metric` values with the directions of
  METRICS.md.
- `results.json`: every case and episode, per-episode summaries (including each fallback-ladder
  retry and rerun), setup draws, latency measurements, randomization name, seeds, versions.
- `grading.json`: `scope` (`full-suite` or `development-subset`), case status and
  outcome counts, `aggregate_score` (= `reward`).
- `error.json`: present only for infrastructure errors (a crash or evaluator error that
  repeated on the rerun, or a replay that was not deterministic); then `reward.json` is withheld
  and the exit code is 2.

Each episode runs in a supervised child process (`supervise.py`), single-threaded in the verifier
image (`OMP_NUM_THREADS=1`), one per CPU, so episodes never compete for cores and replays are
deterministic. METRICS.md ("Numerical failures and reruns") defines the bounded retries:

- every solve has a work budget (FreeGSNKE residual evaluations, not seconds), so whether it fails
  does not depend on the host;
- a step whose solver raises, returns a non-finite state or exhausts its work budget is retried
  after an exact replay of the episode, with more iterations and work, then once through the
  linearised model; at most 3 such steps per episode, and replays totalling at most four times the episode's horizon;
- a worker that exceeds the 1,800 s per-solve hang guard, is silent for `TOKAGYM_STEP_SILENCE`
  (default 2,400 s; the child sends a heartbeat after every step and setup stage), runs past
  `TOKAGYM_EPISODE_TIMEOUT` (default 6 h) or crashes is rerun once with doubled wall-clock limits.
  A repeated stall is a `solver_failure`; a repeated crash withholds the score.

FreeGSNKE can spend unbounded time in a degenerate Newton-Krylov solve and swallows in-process
timeouts, so the external guard is what keeps a bad policy from hanging the verifier. Until v3 a
step that took 180 s ended the episode (or voided it) at once, so an overloaded host changed scores:
on 35 workers with two threads each on 40 CPUs, six of 35 sealed episodes of one package ended on
that budget.

The latency gate is measured after the replay (METRICS.md): the first 600 observations of one
episode per case are replayed through a fresh package process next to the null policy
(`null_policy/`), which parses the request and answers zero volts, alternating which answers first.
The per-step difference of the two round trips is the package's computation. During the replay the
same difference is still taken on every step and reported as `policy_compute_ms_under_load`: there,
timing the null policy only before the episode, on an idle process, charged the submission for the
solver's load (a hold-current package, twelve multiplications, measured 0.30-0.63 ms on a loaded
8-worker host), and whichever process answers first after a solver step pays the wake-up and
cold-cache cost.

A plasma lost by the policy, a coil-limit termination, a solver failure the ladder could not
recover, or a protocol/bounds violation is a **measured outcome**, not an infrastructure error. A `policy_error` in any case, or a median policy computation above
0.5 ms, makes the package invalid: `reward` is 0.
