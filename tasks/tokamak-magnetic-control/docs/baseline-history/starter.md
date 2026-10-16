# Baseline results under earlier task versions (maintainers)

Results recorded under `tokagym-mastu/v0.1`/`v0.2` and their metric versions, moved out of the agent
image: they were scored on an earlier plant and score and are not comparable with v0.3.

## From `environment/baseline/starter/RESULTS.md`

## v0.1 history

These are the task author's runs (run 015) on a shared 208-core host. The integrator checked the
run artifacts (`reward.json`, exit 0) against these tables but did not re-run them.

Recipe:

- 32 privileged PID demonstrations;
- 6,000 behaviour-cloning steps;
- then SAC with the normalized TD3+BC objective, on the nonlinear plant with 36 workers, for 5 h.

The publish-if-better rule kept the behaviour-cloned (BC) actor. The SAC-updated actor lost the plasma
within 20–45 ms in every episode, and the RL improvement step is open.

The published BC package through the trusted verifier (`tests/test.sh`, all five sealed seeds):

| Case | Feasible | Shape RMS [cm] | X-point [cm] | Ip error [%] |
| --- | --- | --- | --- | --- |
| mastu_diverted_hold | 3/5 | 2.36 | 3.18 | 0.93 |
| mastu_high_ip_hold | 3/5 | 1.95 | 3.17 | 7.48 |
| mastu_ip_ramp | 2/5 | 0.98 | 1.35 | 12.43 |
| mastu_p4_shape_hold | 1/5 | 2.90 | 4.30 | 5.11 |
| mastu_pressure_drop | 3/5 | 2.11 | 2.47 | 1.14 |
| mastu_vertical_shift | 3/5 | 2.82 | 4.55 | 1.12 |

That is 15/30 episodes kept, against 6/18 for the deployable PID on the first three seeds.

- Two sealed seeds fail across the board: seed 1004 stalls the solver in every case, and seed 1005
  loses the plasma within 25 ms in five cases. Both are low-pressure draws the demonstrations barely
  covered.
- The weak spot is the plasma current: 7.5 % error at 750 kA and 12.4 % during the ramp.
- Inference takes about 50 µs per step.
