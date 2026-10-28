# Verification and frozen-package replay

Run commands from `brain-computer-interface-decoding/` with prepared `TASK_*_DIR`
paths outside Git. [Data acquisition and pinned releases](../environment/data/MATERIALIZE.md)
describe staging. [Baseline results](../environment/baseline/braindecode/RESULTS.md) and the
[acceptance record](../docs/acceptance-2026-10.md) hold the author-run and maintainer-replayed scores.

## Build and protocol checks

```bash
docker build -f environment/Dockerfile -t bci-development:0.2 .
docker build -f tests/Dockerfile -t bci-verifier:0.2 .
docker run --rm --network none -v "$PWD:/task:ro" -w /task \
  bci-development:0.2 \
  python -m unittest discover -s tests -p 'test_*.py' -v
python tests/container_probe.py --image bci-verifier:0.2
```

When build-time package access is unavailable, provide pinned dependency wheels
under `$BCI_DATA_ROOT/wheels`, then run `python tests/build_offline.py --root
"$BCI_DATA_ROOT"`. This builds the task Dockerfiles with a local wheelhouse;
`wheel-checksums.json` records wheel hashes. Keep image IDs, source commit and
new verification reports with the operator's run. Container probes need a
host-permitted temporary directory; set `TMPDIR` when bind-mount policy requires it.

## Prepared-data and initialization checks

Only authorized operators run these checks with access to private evaluator assets:

```bash
docker run --rm --network none -v "$PWD:/task:ro" \
  -v "$TASK_DATA_DIR:/runtime/development:ro" \
  -v "$TASK_PRIVATE_DIR:/runtime/private:ro" \
  bci-development:0.2 \
  python /task/environment/data/validate_runtime.py /runtime
docker run --rm --network none -v "$PWD:/task:ro" \
  -v "$TASK_DATA_DIR/pretrained:/pretrained:ro" \
  bci-development:0.2 \
  python /task/environment/baseline/braindecode/validate_pretrained.py /pretrained
```

Prepare this task with the task-local commands above, then export
`TASK_DATA_DIR`, `TASK_PRIVATE_DIR` and `TASK_RESEARCH_DIR`
for the compose files and reproduction scripts. These checks do not depend on
a task-specific recipe in the shared agent-eval harness. Keep source roots,
whole FACED LMDBs, private labels and controller records outside agent mounts.

## Replay a frozen package

Find the model/track/seed in [RESULTS.json](../environment/baseline/braindecode/RESULTS.json).
Set `PACKAGE_DIR` to its `hf_package_path` under the pinned model download and
choose a fresh writable `SCORE_DIR`. For a specialist set `TRACK` to its track;
for a shared or RSI package leave it unset to evaluate all five tracks.

```bash
mkdir -p "$SCORE_DIR"
track_args=()
if [[ -n "${TRACK:-}" ]]; then track_args=(--tracks "$TRACK"); fi
docker run --rm --network none --cpus 4 --memory 20g \
  -v "$TASK_PRIVATE_DIR:/private/data:ro" \
  -v "$PACKAGE_DIR:/workspace/submission:ro" \
  -v "$SCORE_DIR:/logs/verifier" \
  bci-verifier:0.2 bash /tests/test.sh "${track_args[@]}"
```

No GPU is required for this inference replay. Omit `--smoke-limit`. Confirm
`scope=full`, every requested track has status `ok`, expected data/package
identities and complete coverage, and `reward.json` agrees with `report.json`.
The verifier drops UID, groups and capabilities for inference; public inputs
are staged separately from root-only labels, scoring code and reward output.
It probes label readability before running, kills leftover runner processes,
and validates the emitted scores. This boundary is not a hostile multi-tenant sandbox.

## Additional training and smoke checks

In the pinned development environment, `python tests/test_shared_matrix.py
--checkpoint-dir "$TASK_DATA_DIR/pretrained"` checks shared forward/backward and audited
checkpoint loading across the five families and tracks. `test_braindecode_matrix.py`
checks the single-task specialists. Real-data shared smoke/export/scoring uses
`tests/run_braindecode_matrix.sh`; `tests/rsi_mock.py` runs a deterministic
two-candidate controller mock. Full shared runs use `tests/campaign/run_shared_full.sh FAMILY SEED`
(or `tests/campaign/run_shared_campaign.sh` for all 25) and strict receipt validation in
`tests/campaign/summarize_shared.py --root "$TASK_RESEARCH_DIR/shared-full" --require-complete`;
add `--models biot --expected-epochs 100` for the extended 100-epoch BIOT campaign. Set
`BCI_DEVELOPMENT_IMAGE` and `BCI_VERIFIER_IMAGE` to images built from the reviewed source.
Check each run's `train.exit`, `score.exit`, full-run receipt and verifier report rather than
checkpoint existence alone. See the [baseline guide](../environment/baseline/braindecode/README.md)
for the training commands.

Smoke scope selects at most 16 examples per observed class for event tracks,
or whole subjects covering observed classes for grouped tracks. It checks
execution and packaging. Infrastructure missing-data errors are fatal;
malformed packages score zero. Smoke scores do not establish benchmark quality.

## Trivial controls

`python tests/anchors.py TRACK TRUTH_JSONL OUTPUT_JSON` computes constant-class
and seeded-random references using the same scoring implementation. Run it only
in the operator environment with access to the chosen private track's
`truth.jsonl`; the reported full-test controls are recorded in
[RESULTS.md](../environment/baseline/braindecode/RESULTS.md#fixed-controls-on-the-randomized-test-layout).
`tests/controls/faced-position/` is a position-only negative control package: it predicts FACED
labels from public shard positions alone and must score at chance on the randomized layout.
