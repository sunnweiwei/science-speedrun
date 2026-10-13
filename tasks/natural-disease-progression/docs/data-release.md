# Controlled data release

The task is derived from MIMIC-IV 3.1 and is not an open dataset. Every person granted access must independently be a PhysioNet credentialed user, complete the required **CITI Data or Specimens Only Research** training, and sign the current MIMIC-IV/PhysioNet Credentialed Health Data Use Agreement. Team membership or another user's credentials do not confer access. Repository administrators must verify this before granting private-repository access.

The controlled repository is `ScienceRSI/natural-disease-progression`. The release must contain only the SHA-256-inventoried `runtime/development` snapshot: train inputs/truth, public validation inputs, validation scorer truth, the train-only normalization artifact, and public metadata. It must not contain `public/test`, `trusted/test`, stable controller patient keys, raw MIMIC tables, raw source IDs, or `working.duckdb`. The latter is a disposable build intermediate, not a task artifact.

The previous private, non-gated revision `6b9c2cd9c92a0dfd795b51e04451a2b9507a027a` failed this rule: its dry-run inventory included test inputs/truth, raw-ID artifacts, and `working.duckdb`. It is superseded and must not be used. The clean development-only `main` revision and inventory are recorded in `docs/release.json`. The superseded commit remains addressable through repository history until an administrator explicitly authorizes the irreversible history purge; access must not be granted while that remains true.

Test inputs and truth live only in separately controlled verifier storage. The track itself is offline at runtime; it never downloads from Hugging Face.
