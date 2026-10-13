# Seasonal Naive: reproduced results

This package was replayed by the task's independent verifier, in the task's images, on all 97
GIFT-Eval configurations (326,490 instances):

| | relative MASE | relative WQL | reward |
| --- | ---: | ---: | ---: |
| Seasonal Naive (this directory) | 1.000 | 0.985 | 0.0075 |

Relative MASE is exactly 1 by construction. The task author's replay gave the same 0.0075. A trivial package that forecasts the context's empirical quantiles scores
0.0.
