# Baseline: Seasonal Naive

The statistical reference that defines the reward's normalizer. The forecast repeats the last
observed season, at the season length GIFT-Eval assigns each frequency. Quantiles come from the
in-sample seasonal-difference errors, which is how statsforecast's `SeasonalNaive` computes
them. It has no parameters and no training. The package follows the contract as it stands:

```bash
bash /workspace/baseline/seasonal_naive/infer.sh /workspace/data/valid/input /workspace/research/sn-valid
python3 /workspace/evaluation/score.py --truth /workspace/data/valid/truth --output /workspace/research/sn-valid \
  --baseline /workspace/evaluation/seasonal_naive_reference_valid.json --report /workspace/research/sn-valid.json
```

`predict_statsforecast.py` is the same forecaster through statsforecast itself. It was used to
cross-check the numpy port, and it needs statsforecast, which this image does not install. Replayed on the test configurations it reproduces the board's
Seasonal Naive row (`RESULTS.md`), so its reward is 0 up to rounding.
