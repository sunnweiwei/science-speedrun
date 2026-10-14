# Starter on the 2022 evaluation panel, by variable, level and lead

Copied from `docs/validation/results-summary/` so the breakdown is in the workspace:

- `main-evaluation.md`: the summary tables (reward, normalized CRPS, paired comparisons, cyclone tracks)
  for the extended Seed 0 starter (580 updates, reward 0.680651), the original Seed 0, GenCast and
  persistence / dry, on the eight 2022 dates with 56 members through 15 days. Figures link to the repository.
- `main-evaluation-by-lead.csv`: one row per variable, pressure level and lead with CRPS, RMSE and spread
  for each of those models (`extended_*` is the starter).

These are the reported evaluation results, not the verifier's truth data. The starter's score on the 2021
development dates was not recorded: compare your own candidates on the 2021 dates, and use this breakdown to
see where the starter is weak (variables, levels, leads) relative to GenCast.
