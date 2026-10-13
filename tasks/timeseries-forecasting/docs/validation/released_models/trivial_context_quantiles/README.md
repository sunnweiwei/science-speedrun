# Trivial control

Forecasts the context's own distribution: the mean as the point forecast and the context's empirical
deciles as the nine quantiles, flat over the horizon. No temporal structure, no seasonality, no trend.

Its purpose is the control `CONTRIBUTING.md` asks for: if a trained package cannot beat this, the metric
is measuring interface compliance rather than forecast quality. Result on the task test split is in
`docs/validation/trivial_control.json`; the leaderboard's own `Naive` (repeat the last value) is on the
comparison table in the task README at reward 0.000.

    bash infer.sh INPUT_DIR OUTPUT_DIR      # CPU only, a few minutes for the full split
