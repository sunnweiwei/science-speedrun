# Maintainer-only experiments

This directory preserves statistical, tree, and deprecated neural baselines for audits and historical reproduction. It is outside the development Docker build context and is not copied into the agent workspace.

- `baselines/` contains the empirical-Bayes and LightGBM comparison scripts.
- `legacy-neural/` contains the superseded mean-pooling neural baseline used by synthetic verifier tests and old reports.

These files are not candidate submission templates for the neural architecture development track. Maintainer reports may use them as controls without exposing reference scores or preferred configurations to an agent run.
