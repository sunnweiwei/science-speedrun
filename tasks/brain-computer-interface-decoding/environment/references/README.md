# Sources and attribution

CBraMod: https://github.com/wjq-learning/CBraMod
Fixed source commit b9e961003214326972c567eff390e75b0287e32a, MIT;
license retained in data/cbramod_reference/LICENSE for the two preprocessing
scripts. Baseline initialization and published checkpoint provenance are
documented in `baseline/braindecode/README.md` and `PRETRAINED_AUDIT.md`.
Paper scores are not task results.

AD: https://openneuro.org/datasets/ds004504 (CC0), official derivatives.
Longitudinal: https://openneuro.org/datasets/ds007176 (CC0), raw EEG.
OpenBMI: https://gigadb.org/dataset/100542 and
https://doi.org/10.1093/gigascience/giz002.
FACED and TUEV are user-provided processed data, not redistributed in Git.
Original dataset terms/access restrictions continue to apply.

Braindecode 1.8.1 shared baselines live in `baseline/braindecode/`, including
its `CBraMod` class. The deterministic RSI mock
checks control flow only; a real RSI controller run is not yet included.
Braindecode release: https://pypi.org/project/braindecode/1.8.1/ . Model source:
https://github.com/braindecode/braindecode/tree/v1.8.1/braindecode/models .
