# Public provenance and native execution

Historical receipts are publication copies. Personal filesystem roots, login and
compute node names, scheduler account names, and allocation identifiers have been
replaced with consistent aliases. Numeric allocation aliases begin at 90000001;
they are not live scheduler IDs. Paths under `/scratch/researcher` are illustrative
locations, not downloadable resources. Original unsanitized receipts and the
identifier mapping are retained privately by the maintainers.

Scientific measurements, hardware models, durations, costs, model identifiers,
and original artifact SHA-256 digests are retained. Those digests identify the
original artifacts, not the sanitized receipt bytes; historical source hashes do
not describe the current portable source. These records document past execution,
not a claim that historical paths can be replayed verbatim.

For native Slurm execution, configure `PATHFINDER_GPU_PARTITION`,
`PATHFINDER_CPU_PARTITION`, `PATHFINDER_SLURM_ACCOUNT`, and
`PATHFINDER_SCRATCH_ROOT` for the authorized local environment. Their portable
example defaults are `gpu`, `cpu`, `research`, and `/scratch`. The GPU allocation,
ledger, and scratch containment checks remain enabled. Example `.sbatch` files
also require local partition/account settings. The isolated verifier retains its
fixed `/workspace/research/run` scratch contract.

Removing identifying details from the current tree does not remove them from
previous Git commits or previously downloaded copies.
