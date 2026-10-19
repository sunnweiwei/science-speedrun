# FFN corrective training and admission

On October 5 the user selected one checkpoint and no consensus. The active
experiment uses the corrected checkpoint's completed ascending reconstruction
directly, with the same weight for the shared-proposal FFN ablation. Its fixed
validation nERL is highest among the retained weights in both directions;
its higher validation merge count remains disclosed. The original complete
ROI and model/context sizes are retained. Prior six-run outputs and costs
are historical evidence, not inputs to the active base segmentation.

The retained BCE snapshots from the original local training run are an
unverified checkpoint selection. The 7250-step snapshot increased full-region
model windows by 3.52 times and pre-stitch connected components by 6.63 times
relative to the 6750-step snapshot, combining ascending and descending runs.
Do not use low four-step BCE as sufficient admission for another full replay.

`train_volume.py` now defaults to production movement and disconnectedness
bias, with eight detached recurrent updates. Exhausted examples have zero
supervision masks; dummy forwards retain the fixed DDP backward count. The
old four-step scalar validation is retained as a diagnostic. `--ffn-policy
legacy` explicitly opts into the former training policy.

Production training requires `--ffn-validation-reference`: generate this with
`ffn_growth_validation.py` from the frozen reference weight on validation.
Reference and candidate must share manifest hash, fixed seed, crop size,
sample count, precision and call cap. Labels only score completed predictions.
Reject incomplete/empty runs, lower Dice or recall, precision loss >0.01,
increased components per positive voxel, or calls per correctly predicted
foreground voxel above reference*1.2. Seed growth can require more calls when
it reconstructs more of the target; total region calls are checked separately.

Only eligible checkpoints enter `top-checkpoints.json`; rank these by actual
production seed-growth Dice. `last.pt` is retained for diagnosis/warm starts
even if it fails admission. No eligible snapshot means no selected checkpoint.
Training supports a pinned warm start and an explicit positive-class weight;
changing that weight is a new experiment requiring reconstruction validation.

`ffn_reconstruction_admission.py` checks complete, scored, fixed-validation
reconstructions before replacing members 2/3. It requires the other four
members and the geometry/policies to remain unchanged, three distinct weight
hashes, replacement nERL at least the weaker first-weight member, omission
no worse than the first-weight maximum+0.02, each replacement's total model
windows <= the matching first-weight member*1.2, and consensus nERL at least
10% above the original consensus. Score and input hashes must be verified by
the caller. Passing seed growth alone does not authorize full reconstruction.

The training unroll count is preserved in each checkpoint identity, but is
excluded from inference-architecture comparison: FFN15 inference uses the
same parameters regardless of the training unroll. All other architecture
fields and full tensor states remain checked.

Corrective experiments retain prior outputs and charges and explicitly record
that test results had already been observed. Preserve the complete test region,
three distinct weights, six ordered runs, three-checkpoint baseline, and all
official inference thresholds. Reuse unchanged member outputs only after
their checkpoint, raw-data, inference-policy and artifact hashes are checked.
Regenerate consensus and downstream inputs when a member weight changes.
