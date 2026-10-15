# Baselines

Reference implementations that run in this workspace and publish a package that follows
`/workspace/contract.md`. `RESULTS.md` in each directory holds the measured numbers. No weights
are shipped.

| baseline | what it is | reward |
| --- | --- | ---: |
| [`starter/`](starter/) | ESM-2 `esm2_t6_8M` from scratch: masked-language pretraining on the UniRef50 subset, joint fine-tuning on all eleven tasks, `tune` selection; 11 h on one H100 | 0.6695 / 0.6774 (seeds 0 / 1) |
| [`trivial/`](trivial/) | per-task constants, no training | 0.2874 |

`/workspace/references/measured_runs.md` records further measured configurations (a 37M generic
transformer, throughput per model size).
