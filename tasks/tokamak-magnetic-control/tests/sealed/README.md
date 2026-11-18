# Sealed verifier files

The files listed in `SHA256SUMS` fix this task's hidden evaluation conditions (seeds, rosters).
The verifier reads them from `/tests/sealed/`; the agent's development image never contains them, and agent
sessions run offline, so publishing them here does not expose them to the agent.

Before every verification the runner (`stage_sealed.sh`, called by `run_verify.sh`) checks them against
`SHA256SUMS` and stops if any file is missing or differs; `tests/test.sh` repeats the check inside the
verifier, so substitute seeds or rosters can never be scored. A checkout without them (an older public
copy) can take them from `SEALED_DIR`, a directory laid out as `SEALED_DIR/<task>/tests/sealed/<file>`.
