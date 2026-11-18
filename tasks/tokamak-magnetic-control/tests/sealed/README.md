# Sealed verifier files

The files listed in `SHA256SUMS` fix this task's hidden evaluation conditions (seeds, rosters).
The verifier reads them from `/tests/sealed/`; the agent's development image never contains them.

They are not published with the public task repository. `SHA256SUMS` names the exact files the
verifier needs. Before every verification the runner (`stage_sealed.sh`, called by `run_verify.sh`)
copies them from `SEALED_DIR` (a checkout of the private task repository, laid out as
`SEALED_DIR/<task>/tests/sealed/<file>`) and stops if any file is missing or differs. `tests/test.sh`
repeats the check inside the verifier, so substitute seeds or rosters can never be scored.
