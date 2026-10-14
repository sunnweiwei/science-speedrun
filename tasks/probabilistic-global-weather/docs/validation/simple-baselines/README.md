# Initial persistence comparison

This table extends the previously reported temperature comparison using exactly
the same three test initializations: 2022-01-05 00 UTC, 2022-03-01 06 UTC and
2022-04-27 12 UTC. Persistence holds the initialization 2m temperature unchanged.
All models use native-grid global area weights; RMSE is the square root of mean
per-date MSE. Temperature differences have identical numerical values in K and °C.

[Results](summary.csv) and [per-date metrics](results.json) are the initial
three-date diagnostic, not completion of the new all-variable baseline suite.
Seed 0 improves RMSE by 7.0% at one day, but is worse than persistence at the
reported 3, 5, 10 and 15 day leads. GenCast improves on persistence at all five.

The ongoing all-date workflow is documented in
[the comparison guide](../../../paper-comparison/README.md#simple-references-added-by-this-track).
