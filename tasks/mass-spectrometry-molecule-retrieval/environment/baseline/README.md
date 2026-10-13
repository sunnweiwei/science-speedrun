# Baselines

| baseline | what it is | Hit@1 (reward) |
| --- | --- | ---: |
| [`starter/`](starter/) | Fourier-DeepSets fingerprint regressor, trained from scratch; the contract and plumbing starter and the all-condition fallback | 4.00 % |
| [`starter/glacier/`](starter/glacier/) | native 15.1M-parameter GLACIER forward simulator from random initialization, with the DeepSets fallback | 40.45 % |

`starter/RESULTS.md` has the reproduced numbers (eight GLACIER seeds). The reward is Hit@1; the
official-weight GLACIER/MIST hybrid reaches 49.39 % (`/workspace/references/`). No weights are
shipped.
