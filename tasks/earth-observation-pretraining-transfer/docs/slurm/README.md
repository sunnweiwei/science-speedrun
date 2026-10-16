# Optional Slurm operator templates

Train the current two-phase baseline with the commands in the
[baseline guide](../../environment/baseline/starter/README.md), inside an
operator-provided GPU allocation. The generic native-training and downstream
launchers in this directory are developer utilities; their defaults are not the
current WorldCover/moisture baseline recipe.

Set `EARTH_REPO_ROOT`, `EARTH_DATA_ROOT`, `EARTH_RUNTIME_ROOT` and
`EARTH_SLURM_PARTITION` for the deployment. Supply partition, account and hardware
selection at submission time. Never run CUDA work directly on a login node.
Data preparation and frozen-probe fitting are CPU-only.

After selecting and publishing a package, `container_verify.sbatch` can run the
independent Apptainer verifier. It accepts `OLMO_MODEL`, `OLMO_EVAL_DATA` and a
fresh `OLMO_OUTPUT`, and isolates model inference from labels. With
`OLMO_VERIFIER_PHASE=inference`, it records a label-free replay manifest;
`container_score.sbatch` scores those exact artifacts afterward on CPU. Final
LFMC scoring must follow candidate selection. Do not use final scores to select
an encoder.

Build the verifier from the repository root using `tests/Apptainer.def` and the
runtime prepared by `environment/setup.sh`. The build argument
`EARTH_RUNTIME_ROOT` identifies that runtime. Use operator-owned temporary
storage and preserve scheduler-provided GPU visibility. See
[portable execution](../PORTABLE_EXECUTION.md) for path conventions and
[delivery readiness](../READINESS.md) for validation limits.
