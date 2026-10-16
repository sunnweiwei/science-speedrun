# Verification and regression tests

## Harbor entrypoint

`tests/test.sh` is bundled into the separate verifier image. It invokes the
trusted `harbor_verify.py` on the **26-case verifier suite** (every 2D case, one 3D case per family;
`case_spec.VERIFIER_CASES`) using the artifact at
`/workspace/submission` and read-only data at `/data`. It publishes
`/logs/verifier/reward.json` only when all requested physical metrics are valid.
The detailed replay includes explicit policy/infrastructure errors; no failed
case is silently omitted or filled with a favorable zero.

The native Compose definitions use source/profile preflight, not the retired
normalized-return suite gate. Set `TASK_DATA_DIR` before building/launching.
See [evaluation](../environment/evaluation/README.md) and [contract](../environment/contract.md).

## Fast CPU tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s hydrogym/tests -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s fluid-flow-control/environment/references/specialists -p 'test_*.py' -q
python3 fluid-flow-control/environment/data/public_release.py --check
```

The full pinned task runtime supplies PyTorch/SB3, TorchRL and NumPy; a host
without those dependencies may skip ML-specific tests. Report skips honestly.
Tests include synthetic fixtures for protocol, snapshot/relocation, common
shape capacity, shared training/resume, malformed actions, source/hash changes,
physical extraction/windows, MAIA GPU binding and the Harbor metric bridge.
Synthetic numbers are never published as CFD reference scores.

[docs/VALIDATION.md](../docs/VALIDATION.md) records the actual container and native checks
performed for this PR. Real full-episode specialist evidence lives under
`environment/references/specialists/verified`; CPU tests alone do not establish CFD performance.

The older normalized-R unit fixtures remain regression coverage for retained
historical helpers. They do not define the current benchmark metric or require
a production `suite.json`.
