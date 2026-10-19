# Relocatable prediction-result contract

`pathfinder-prediction/v1` packages an existing prediction volume and physical coordinates in three fixed files:

- `segmentation.npy`: nonempty ZYX uint32/uint64 instance labels; 0 means unknown/omitted.
- `metadata.json`: axes, voxel_outer_corner, shape, origin_nm, voxel_size_nm.
- `prediction.json`: schema, data types, SHA values for those files, and physical coordinates; nonzero labels alone do not establish complete predictions.

Package coordinates must match expected metadata supplied separately by the scorer. Reject mismatched file hashes, array shapes, or coordinates/grids; submitter metadata cannot change the scoring ROI. Contract checks do not establish compliant training, absence of GT use, or biological quality; those still require execution isolation and independent evaluation.

Add a small sidecar beside the original prediction, without copying the large volume or changing its pixels:

```bash
"$PYTHON" "$CODE/prediction_contract.py" --mode seal \
  --prediction "$ROOT/continuous-evaluation-v1/reconstruction" \
  --metadata "$ROOT/continuous-assembled-v1/metadata.json" \
  --report "$ROOT/continuous-evaluation-v1/prediction-seal.json"
```

This command requires the sidecar to be absent; the complete directory is relocatable afterward. For independent scoring:

```bash
"$PYTHON" "$CODE/evaluate_erl.py" \
  --skeletons "$ROOT/continuous-sparse-dev-v1/skeletons.json" \
  --metadata "$ROOT/continuous-sparse-dev-v1/metadata.json" \
  --segmentation "$ROOT/continuous-evaluation-v1/reconstruction/segmentation.npy" \
  --prediction-contract "$ROOT/continuous-evaluation-v1/reconstruction" \
  --evaluator "$ROOT/vendor/funlib.evaluate/funlib/evaluate/run_length.py" \
  --boundary-margin-voxels 0 0 0 --output "$ROOT/continuous-evaluation-v1/contract-metrics.json"
```

Matching data and ROI are prerequisites. Trusted metadata, skeletons, and evaluator are supplied by the scorer, rather than read from the model package. Compare canonical coordinates and pinned provenance across protocols; unrelated differences in JSON fields do not establish a changed physical ROI.

**Measured evidence:** Historical real 64³ development ROI 0 predictions were copied to a new directory, packaged, relocated again, and rescored with the original independent skeletons/metadata/funlib. nERL 0.523882198 and all primary diagnostics/input hashes matched exactly; see `receipts/prediction-contract-replay.json`. Coordinate shifts and pixel modifications are rejected. This ROI is a historical identity-merge result; the check does not establish continuous-volume completion or relocated model-inference replay.

Native PATHFINDER task/container integration and isolated model execution remain incomplete; this contract only defines their result-input boundary.

The scoring entrypoint first losslessly renumbers GT instance IDs and positive predicted IDs into separate contiguous ranges, retaining prediction 0 as omission. Mixed integer arrays inside the pinned funlib version may collapse floating-point precision for very large uint64 labels. In a synthetic counterexample, two perfect independent segments incorrectly scored 0.5 before adaptation and correctly scored 1 afterward. Ordinary small IDs still score 1, while all omissions and entirely incorrect merges score 0; see `receipts/erl-id-audit.json`. Renumbering does not change nodes, edges, lengths, or label equality.

New results also pin `adapter_sha256`; local before/after comparisons reject mixed adapters. Current continuous FFN global labels are uint32 and do not trigger this large-ID issue. Pending CPU evaluation uses the frozen `code-continuous-evaluate-v2` snapshot, with the same corrected adapter for before/after predictions. This correction is not an improvement in model quality.

## Frozen trusted scoring suite and reward interface

The scorer runs `freeze_evaluation.py --skeletons ... --metadata ... --evaluator ... --output SUITE`. It copies small trusted skeletons/coordinates/pinned funlib implementation; `suite.json` records input and scoring-code hashes without model assets. Submitters must not supply or modify this suite. Independent scoring entrypoint:

```bash
"$PYTHON" "$CODE/verify_prediction.py" --prediction "$PREDICTION" \
  --suite "$SUITE" --output "$RESULT"
```

Successful execution produces `assessment/metrics.json`, `verification.json`, and scalar `reward.txt` (local nERL). Integrity/geometry/scoring errors exit nonzero; `reward.txt` is first reset to zero to prevent stale scores. Completed assessment directories are not silently overwritten. Mount code, suite, and predictions read-only and isolate network/host paths; this entrypoint does not execute submitted models or establish training compliance.

The packaged historical real ROI 0 prediction scored the same 0.5238821980488535 through this entrypoint. A negative case with tampered trusted metadata was rejected, resetting a preexisting reward=1 to 0; see `receipts/frozen-suite-verifier.json`. These are actual CPU scoring checks; native task/container integration and independent GPU model replay still require acceptance.

For isolated execution of the same reward entrypoint, use `replay_prediction_container.py --root ROOT --image IMAGE.sif --prediction PREDICTION --suite SUITE --output OUTPUT`, requiring a CPU cpu allocation. `--suite` is mutually exclusive with the old three trusted-input arguments. Suite mode retains verification reports, reward, complete scoring logs, and isolation evidence. Actual job 90000020 was verified; see RUNTIME.md.
