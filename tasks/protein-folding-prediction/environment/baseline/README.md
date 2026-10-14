# Baselines

| baseline | what it is | reward |
| --- | --- | ---: |
| [`starter/`](starter/) | approximately-50M ESMFold2 graph trained from scratch on the 100K pool (one epoch, 3.7 h on six H100s) | 0.037 |

`starter/RESULTS.md` lists every track next to the frozen ESMFold2-Fast reference (0.545), which
is the model to beat on every track. The reference's checkpoint is data under
`/workspace/data/models`: you may run it for analysis, but never copy its parameters. No weights of
the starter are shipped.
