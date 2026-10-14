# Seed 0 autoregressive continuation experiment

Branch from the completed AR2 checkpoint at update 556, including Adam moments.
Keep the original 562-update model as the control. Increase AR3–AR8 from one to
four updates per stage, producing a 580-update model. Preserve architecture,
training windows (through 2020), batch 64, optimizer, peak learning rates and the
normalized warmup/cosine rule. The longer schedules stretch that same rule over
four updates, so the total optimizer exposure increases.

The explicit schedule-branch option in `train_stage.py` validates the parent
schedule, checkpoint counters, completed AR2 state and unchanged earlier stages.
It retains parameter and optimizer-state compatibility checks and records the
branch. It does not rewrite the parent checkpoint or overwrite the control.

Measured continuation cost: 11 h 35 min 32 s on four RTX PRO 6000 GPUs
(46.37 GPU-h), with a separate evaluation allocation. This is an
additional diagnostic experiment outside the original 17-hour training budget.

After training, automatically compare control and extended weights using four
fixed 2021 development initializations (January 5, April 5, July 5, October 5,
all 00 UTC), eight members and shared random keys. Score leads 24, 72, 120 and
240 hours with identical final HRES preprocessing and training-only statistics.
The primary endpoint is 10-day 10m wind-speed RMSE. Report CRPS and RMSE for all
five fields: 2m temperature, sea-level pressure, 500 hPa geopotential, 10m wind
speed and six-hour precipitation ending at each lead. Four dates do not support
a statistical significance claim. The diagnostic stops at ten days because that
is the existing development truth coverage.

A positive result supports additional AR-stage training as a useful intervention;
it does not distinguish the benefit of longer-rollout optimization from the
benefit of additional training compute. The completed results follow.

## Results

Completed a paired comparison of the original 562-update seed 0 and a 580-update continuation. Resume from AR2 (update 556), retaining optimizer state, and increase AR3–AR8 from one to four updates each. Both checkpoints use the same four held-out 2021 development dates, eight members, inputs, normalization and random keys. Positive percentages mean lower error.

| Variable | 1 d RMSE reduction | 3 d | 5 d | 10 d | 10 d CRPS reduction |
|---|---|---|---|---|---|
| 2 m temperature | 0.61% | 0.75% | 1.36% | 2.73% | 4.46% |
| Sea-level pressure | 1.08% | 4.52% | 5.44% | 6.40% | 4.53% |
| 500 hPa geopotential | 0.95% | 3.09% | 3.84% | 3.50% | 3.62% |
| 10 m wind speed | 0.34% | 0.94% | 1.71% | 2.47% | 2.84% |
| 6 h precipitation ending at lead | 0.06% | 0.02% | 0.04% | 0.42% | 1.24% |

**Additional AR-stage training improves all 20 aggregate variable/lead combinations in both RMSE and CRPS.** Temperature, pressure, geopotential and wind RMSE improve on all four dates at every reported lead. Individual-date precipitation results are mixed at some short leads.

The primary endpoint, 10-day wind-speed RMSE, decreases from **9.927 to 9.682 m/s (2.47%)**. Ten-day temperature RMSE decreases from **4.913 to 4.779°C**, and pressure from **15.31 to 14.33 hPa**. The gains support additional AR training as a useful intervention, but long-range errors remain large. This experiment does not establish that the continuation beats persistence or GenCast: those comparisons use different 2022 dates and the original checkpoint.

Continuation training consumed **46.37 RTX PRO 6000 GPU-hours** (four GPUs × 11 h 35 min 32 s), outside the original training budget. Four development dates do not establish statistical significance or full-test generalization. The experiment changes both rollout-training exposure and total optimization compute, so it does not isolate an AR-specific causal explanation.
