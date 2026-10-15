# Operator benchmark: scoring an upstream-RFD3 checkpoint

Scores a checkpoint from an upstream reduced-width RFD3 training run on the task's four
tracks, using the task's own grader. **Operator-side only** - nothing here is copied into
the agent image, and it is not part of the starter.

    cd "$RFD3_EVAL_WORK/logs" && sbatch <repo>/protein-design/environment/evaluation/benchmark/run_eval.sh \
        <run-name> <ckpt> <view> "<tracks>" [timesteps] [subset]

    # the benchmark view: 41 enzyme, 4 sm_binder, 5 ppi_binder. It has no unconditional
    # targets; nothing aggregates across tracks, so that simply means three tracks report.
    sbatch run_eval.sh ep60-evaluation "$CKPT" "$DATA/data/evaluation" \
        "sm_binder ppi_binder enzyme"

    # unconditional, which only the development views have
    sbatch run_eval.sh ep60-uncond "$CKPT" "$DATA/data-full/development/tune" unconditional

`RFD3_VENV` and `RFD3_EVAL_WORK` override the host paths; the defaults are this cluster's.

## Why this is separate from the starter's `rfd3_infer.sh`

They solve different problems and neither replaces the other:

| | `benchmark/make_inputs.py` | `starter/rfd3_inputs.py` |
| --- | --- | --- |
| spec source | **upstream's published benchmark specs**, vendored at `environment/data/targets/*/input.yaml` | rebuilt from the task's `.npz` conditioning context |
| works on | the **evaluation** view only | any view - fit, tune, evaluation |
| who runs it | the operator, comparing against upstream's own numbers | the agent, which only ever mounts `development` |

The evaluation view is never mounted into the agent container (`docker-compose.yaml`
binds `${TASK_DATA_DIR}/development` and nothing else), so an agent could not
use this path even if it wanted to. Conversely, reusing upstream's exact `unindex`,
`select_fixed_atoms` and `select_hotspots` strings is the only way to say "this is the
same benchmark upstream published", which is what makes an operator number comparable.

Both paths converge on the same `collect.py` logic and the same grader.

## The three stages

1. `make_inputs.py` - view -> `InputSpecification` JSON per track. One thing changes from
   the published spec: the design length is pinned to what the request asks for, because
   the grader rejects a design whose residue count is not `design_length`. The vendored
   CIFs are in the *same frame* as the task's conditioning contexts (checked: max distance
   from a context atom to the nearest CIF atom is 0.0000 A for both a ppi target and an
   enzyme ligand), so designs are directly comparable with what the grader measures.
2. `rfd3 design` - upstream inference, unmodified. The reduced widths come from the
   checkpoint itself (`train_cfg` inside the ckpt), so no config is repeated here.
3. `collect.py` - output mmCIF -> the task's submission format. For enzyme it recovers
   `motif_map` by matching the motif's fixed atoms in the output, which is the honest
   reading of an unindexed design: the model chose where to put the motif, and we report
   where it put it.

Then `score.py`, which scores the real criterion.

## Reading a number from this

- **Real designability.** `run_eval.sh` calls `score.py` with no shortcut flag:
  LigandMPNN/ProteinMPNN sequences refolded by RoseTTAFold3, judged on the track's
  `success.txt` criterion. The geometry-only upper bound this section used to describe was
  a consequence of AlphaFold 3's parameters being unshippable; RoseTTAFold3's are public,
  they are on this host, and the flag was deleted on 2026-09-26. Numbers taken before that
  date are upper bounds and do **not** belong in a column with these.
- **100 designs per target**, generated as `n_batches=25 diffusion_batch_size=4`, which is
  what every view requests.
- **Never `--repair`.** There is no repair path here at all; this is pure model output.
- **The enzyme motif criterion is now judged where upstream judges it** - on the refolded
  models, all-atom RMSD <= 1.5 A after motif-backbone alignment, over the atoms the
  *context* carries. Until 2026-09-26 the task additionally applied that RMSD to the raw
  design, which was stricter than anything upstream publishes; that check is deleted. What
  remains of the difference is only which atoms are compared: upstream's spec fixes a few
  tip atoms per motif residue, and RFD3 is trained for exactly that (tipatom is 53% of
  pretraining).

## Three bugs this cost, kept so they are not rediscovered

1. A pinned contig of `150` parses as **residue index 150**, not a 150-residue design.
   It must be `150-150`. This failed the whole sm_binder track once.
2. `diffused_index_map` keys come back **truncated** - motif residue `B1029` appears as
   key `B102` - so exact lookup silently dropped 8 designs. Matched on unique prefix now.
   For `M0110_1c0p` the truncation is genuinely lossy (`A1335` and `A1339` both give
   `A133`), so upstream's own output cannot be disambiguated there and the coordinate
   fallback takes over.
3. The coordinate fallback originally demanded all four backbone atoms. These contexts
   carry only the 2-3 fixed **tip** atoms per motif residue, so it could never fire; it
   needs >= 2 shared atoms.
