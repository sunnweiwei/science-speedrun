# Single-checkpoint model contract

Keep a `pathfinder-model-package/v1` at `/workspace/submission`. Its required
files are `package.json` and exactly three selected checkpoints:
`weights/ffn.pt`, `weights/sense.pt`, `weights/shape.pt`. The manifest records
SHA-256 hashes, model architectures and `inference.search_threshold: 0.8`.
Export with `bash /workspace/baseline/pathfinder-fast/run.sh export` (which runs
`export_weights.py`). Stage a new version outside the published path, validate
it, then publish the complete directory with `run.sh publish`.

The verifier loads checkpoints as data with `torch.load(weights_only=True)`
into trusted model classes. Submitted Python, shared libraries and cached
predictions are never executed or used. Legacy packages may contain such extra
files; only the three checkpoint files and their declared configurations are
staged. Each checkpoint is at most 1GiB. Malformed weights or unsupported model
configurations are invalid submissions. Do not include test data or predictions.

The architecture is FFN width32/20 residual blocks/33³, SENSE
width8/factors[1,2,4,8,8]/128³ and SHAPE8192 points/10µm. The evaluator uses one
FFN checkpoint, ascending seeds, overlapping windows and stitching, with no
consensus. It derives the instance context, processes all eligible SENSE centers,
evaluates all shared pairs with FFN and two-node SHAPE, and applies deterministic
disjoint SHAPE merges at 0.8. Higher-order motifs and graph search are omitted.
Each FFN window grows from its seeds until they are used up or the window reaches
1,000,000 FFN calls; at that cap the segments grown so far are kept and the
remaining seeds stay unlabelled, which the score counts as omissions. A model
whose growth needs fewer calls therefore also loses less to the cap.

Inputs are raw EM and its observation mask on the declared ZYX, voxel-outer-
corner, nanometre, 16nm grid. The formal entry starts from these image inputs
and model weights; no prepared prediction/context is accepted as a shortcut.
Trusted runtime dependencies and FPS are operator-provided, never loaded from
the submission. Predictions run in a namespace with no private suite, original
submission directory, experiment history or network. Scoring occurs afterward
against the private suite.

Reward is the disjoint-pair SHAPE prediction's local sparse nERL in [0,1], higher
is better. `/logs/verifier/reward.json` includes reward and all three methods'
raw metrics; `comparison.json` retains FFN-only and shared-proposal FFN controls.
Missing or incomplete stages do not receive a diagnostic partial reward.
Invalid submissions score zero. Infrastructure, namespace, resource and runtime
failures produce `status.json` and no reward.

The verifier exposes eight GPUs, 64 CPUs and 512GiB memory with a 12-hour outer
timeout (11.5 hours for prediction). Stage caps are FFN 8h, SENSE 4h, FFN-pair
2h and SHAPE 2h; a stage that reaches its cap produces no reward. A complete run
with a baseline-recipe FFN took about 7 hours on four H100s (FFN 4h46m, SENSE
67 minutes) and is about 4 hours on eight. FFN time grows with the number of
calls the checkpoint's growth needs.
