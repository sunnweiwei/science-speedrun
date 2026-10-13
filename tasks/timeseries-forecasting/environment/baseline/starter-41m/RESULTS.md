# starter-41m: reproduced results

The commands in `README.md` were run in the task's development image on eight H100s, data-parallel
with global batch 256. The run was stopped at step 190,000 of 2,000,000 to free the GPUs for the main
baseline. The best validation checkpoint (step 70,000; validation skill +0.026 on 15 stratified
configs) was published and replayed by the task's verifier on all 97 configurations
(326,490 instances).

| run | relative MASE | relative WQL | reward | time |
| --- | ---: | ---: | ---: | ---: |
| this directory, 8 × H100, best at step 70,000 | 0.940 | 0.681 | **0.200** | packing 2 min; 70,000 steps in 28 min; inference 7 min |
| task author, 1 × A100-40GB, best at step 560,000 of 2,000,000 | 0.937 | 0.671 | 0.207 | 28 GPU-hours for the full schedule |

Data-parallel training runs at about 43 steps/s on eight H100s against about 20 on one A100. The
model is small and each rank's batch is 32, so per-step overhead, not compute, bounds the speed.
In the author's single-GPU run, validation peaked at 28 % of the schedule. In this run it peaked at
step 70,000 and had not improved by step 190,000.
