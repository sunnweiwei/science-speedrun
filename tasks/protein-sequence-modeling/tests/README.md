# Independent verifier

Two-service replay, mirroring `clinical-trial-modeling`:

- **main** (`test.sh` → `main_verifier.py`): validates the submission package
  (requires `infer.sh` and `README.md`, rejects escaping symlinks), validates
  the verifier inputs against their manifest hashes, runs
  `bash infer.sh /benchmark OUTPUT` as a demoted `agent` user with no network,
  then copies only the eleven bounded `predictions/<task>.jsonl` files into a
  read-only sealed snapshot. Labels are **not** mounted in this service.
- **grader** (`grader.py`): waits for the sealed snapshot, cross-checks
  snapshot and private-label manifests, scores every task with
  `metrics.py`, and writes `report.json` plus a flat `reward.json` to
  `/sealed/results`, which `main_verifier.py` copies to `/logs/verifier`.

Failure handling: a malformed, missing, duplicate, or unknown prediction file
zeros only that task and records `invalid_reason`; setup failures (bad
submission package, input hash mismatch, snapshot mismatch) abort the run and
are therefore distinguishable from measured model performance.

Isolation boundary: inference runs in a container without the private labels
and cannot modify inputs (read-only mount) or the sealed snapshot (created
after inference exits, hash-pinned). This is independent replay, not a
cryptographic blind test; the operator supplies the private label directory
only to the grader service.

**The privilege drop is verified, not assumed.** `main_verifier.py` runs the
submission as the unprivileged `agent` user, and after `setuid` it checks that
the drop actually took effect and that root cannot be regained, aborting if
either fails. This is not theoretical: under a sandbox that emulates root with
an `LD_PRELOAD` shim rather than a real uid mapping — apptainer's fallback
`--fakeroot` when the user has no `/etc/subuid` allocation, for example —
`os.setuid()` returns success and `geteuid()` reports the new uid while the
process retains full privileges. Without the post-condition check the verifier
would run untrusted submission code as root and look perfectly healthy doing
it.

The verifier also refuses to start unless it is actually root, rather than
failing obscurely later, so there are only two outcomes: a real privilege drop,
or a clear refusal. It never silently degrades.

### Reproducing the privilege-separation check

This path has **not** been exercised, because it needs genuine multi-uid
mapping. What that requires, and what was verified present on the host where
this task was built:

| Requirement | Status when tested |
| --- | --- |
| Unprivileged user namespaces | available (`max_user_namespaces` = 1030679) |
| `newuidmap` / `newgidmap` with `cap_setuid` | present, capability set |
| A container runtime | apptainer 1.4.0 and enroot 4.1.2 |
| **An `/etc/subuid` range for the operator** | **absent — the only thing missing** |

Without that range `newuidmap` refuses (`uid range … not allowed`), so only one
uid can be mapped and the container cannot hold both root and `agent`. Adding
one line to `/etc/subuid` and `/etc/subgid` (`<user>:200000:65536`) is enough
for `apptainer build --fakeroot` to use a real mapping. A normal Docker daemon
needs none of this and is the intended environment.

Everything else about the container was exercised: the image builds from
`Dockerfile` (the `agent` user is created, `test.sh` lands at 0755, the base
image supplies Python 3.11 with torch), network egress is blocked, the
inference stage sees inputs carrying no labels, and manifest and submission
validation both pass against real data.

`metrics.py` is kept byte-identical with
`../environment/evaluation/metrics.py` so development self-scores match.

Unit tests (synthetic, no data needed): `bash run_unit_tests.sh`.
