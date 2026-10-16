# Portable execution and private run records

This repository publishes model/data code, scientific results and protocol
limitations. It does not publish operator identities, cluster names, partitions,
node names, scheduler job IDs, absolute site paths or launch receipts.
Generate fresh receipts when reproducing a run and retain them privately.
The current baseline result summary is not a machine-verifiable execution manifest.

Set these variables before using the Slurm templates:

```sh
export EARTH_REPO_ROOT="$PWD"
export EARTH_DATA_ROOT="/data/earth"
export EARTH_RUNTIME_ROOT="/opt/earth-runtime"
export EARTH_SLURM_PARTITION="your-gpu-partition"
sbatch --partition="$EARTH_SLURM_PARTITION" earth-observation-pretraining-transfer/docs/slurm/container_verify.sbatch
```

The example roots are generic conventions, not paths from an actual deployment.
Set `OLMO_MODEL`, `OLMO_EVAL_DATA` and `OLMO_OUTPUT` to the published package,
private evaluation data and a fresh output directory before submitting final
evaluation. Select any required scheduler
account with a command-line `--account` argument. GPU type/count and resource
requests can be overridden at submission time. Python execution optionally
checks `EARTH_SLURM_PARTITION` when supplied; local deployment restrictions must
be supplied by the operator.

For the verifier Apptainer build, supply the runtime root using
`--build-arg EARTH_RUNTIME_ROOT=/opt/earth-runtime`. The installed, pinned Python
packages must already exist under that root as described by the setup script.
Do not publish generated scheduler logs, receipts, environments or full paths
without reviewing them for private deployment information.
