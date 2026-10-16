# Baseline results under earlier task versions (maintainers)

Results recorded under `tokagym-mastu/v0.1`/`v0.2` and their metric versions, moved out of the agent
image: they were scored on an earlier plant and score and are not comparable with v0.3.

## From `environment/baseline/hold_current/RESULTS.md`

## v0.2 (sealed-v0.2, `tokamak-shape-metrics/v2`)

Full `harbor_verify.py` replay of this package on all seven cases, sealed seeds 1001-1002, nonlinear plant
(the verifier's review, 2026-10-03, local FreeGSNKE 3.1.0 with two workers, 777 s):

| case | outcome | plasma lost after |
| --- | --- | --- |
| all seven cases | `plasma_lost` in 14/14 episodes | 29-117 steps (15-58 ms) |

`reward` 0.0; median `policy_compute_ms` 0.13 (the null policy's round trip was 0.8 ms), within the
0.5 ms gate. The plasma is lost sooner than in v0.1 because the sealed plant adds supply offsets,
sticking supplies and the handover pre-roll.

## v0.1 history

Sealed verifier, seeds 1001–1003, nonlinear plant (the task author's run; artifacts checked by the
integrator, not re-run).

| case | feasible | plasma lost after |
| --- | --- | --- |
| all six cases | 0/3 each, 0/18 in total | 48–362 steps (24–181 ms); the seed-1001 draws survive longest |

A randomly initialized actor (`starter/init_random.py`) loses the plasma after 15–52 steps and
also scores feasible fraction 0 on every case; the 2026-09-27 layout check reproduced 0 on all six
cases for a random-init package (`docs/VALIDATION.md`).
