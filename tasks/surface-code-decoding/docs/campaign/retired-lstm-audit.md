# Retired LSTM run (maintainers)

Moved out of the agent image from reproduction/alphaqubit-source-audit.md.

## From `environment/baseline/alphaqubit/reproduction/alphaqubit-source-audit.md`

## Retired LSTM run: stop audit

The user pivoted the primary baseline to AlphaQubit. Root stopped active LSTM
jobs and cancelled the pending chain. Read-only verification found eight
previously active jobs ended with Slurm signal10/SIGUSR1 (`FAILED`, `0:10`),
because the signal reached the batch shell before Python could publish its
final snapshot. Their progress files remain `training` and their last sessions
have no end timestamp; this was **not a graceful training completion**.

All16 `best.keras` / `last.keras` archives were verified readable with finite
model/optimizer tensors and optimizer iterations matching embedded data
cursors. Prior atomic checkpoints remain usable. Exact SHA256 values,
recoverable epoch/batch cursors, job accounting and lower bounds on lost work
are retained outside Git in
`native-v1/pivot-stop.json` under the artifact root. No old weights, metadata,
progress files, or Slurm jobs were modified by this audit.
