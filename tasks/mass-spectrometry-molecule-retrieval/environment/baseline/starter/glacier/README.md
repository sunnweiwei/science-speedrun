# Native GLACIER research starter

This is the pinned native 15.1M-parameter GLACIER architecture, initialized
randomly. It learns fragment masks and spectrum intensities from the released
training fold. Its fixed package routes the published `[M+H]+`, finite-energy,
Orbitrap/QTOF domain through forward simulation and entropy ranking, and uses
a separately trained DeepSets fallback elsewhere. Every candidate and every
query remains in the evaluation. The fallback is editable and replaceable.
The starter also uses that fallback when the observation has no positive peak
inside the native scorer's precursor/binning range, or when every candidate is
outside GLACIER's structure support (for example, all exceed 100 heavy atoms).
These public-input checks preserve complete output on future conforming inputs;
they do not change the frozen official-weight reference's routing policy.

The official-weight GLACIER/MIST result, 49.391% test Hit@1, is an external
performance reference. It is not the expected score of this scratch recipe:
its initialization, fallback and compute differ. Use identical validation
queries to compare your own experiments. No pretrained checkpoint is supplied.

The completed task-owned run obtained **40.322397% full-test Hit@1** and
35.987441% full-validation Hit@1. Native training charged 20,988.4 seconds on
one H200; the separately trained fallback took 504.8 seconds. Test prediction
used CS A5000s followed by CPU ranking, so this is not a single-H100 timing
measurement. See the aggregate [training and test record](../../../references/task_trained_baseline.json)
and [SOTA/protocol audit](../../../references/SOTA_AUDIT.md).

New publications use native **inference batch size 64**, with two CPU workers.
A [reviewer replay](https://github.com/T0-RSI/ai4sci-tasks/pull/27#issuecomment-5750791449)
reported 3.4–3.9 hours for all 17,556 test queries on one H100 at batch 64,
compared with about five hours at batch 8. This is reviewer timing evidence;
acceptance of the current native Docker images with freshly downloaded HF data
is a separate deployment check. Training still defaults to batch 8.

## Available resources

- `/opt/glacier/bin/python`: pinned Python 3.11.14, Torch 2.6.0/CUDA 12.4,
  DGL 2.5.0, Lightning 2.6.5 and native graph dependencies. The verifier has
  the same runtime. Use this interpreter for both parts of this hybrid.
- `/opt/ms-pred/src/ms_pred`: editable-copy source from commit
  `ed8311f22958cb37f055b663b5f56c5c77a2ee33`, without upstream weights.
- `/workspace/data/glacier`: MAGMa fragment targets and train/validation row indices,
  generated only from the corresponding public development folds. The
  manifest reports selected, annotated and excluded spectra. This preparation
  is done before the agent clock starts. Training never reads a test fold.
- `/workspace/data/synthetic_augmentation`: optional existing held-out-clean simulations.
  The supplied native recipe does not consume them automatically. Designing a
  compatible pretraining/augmentation method is part of the research space.

## Establish a complete package

Copy the complete starter so its relative reference-script path stays valid.
Keep optimizer state outside the submission; only the published model package
is transferred to the verifier.

```bash
cp -a /workspace/baseline/starter /workspace/research/starter
cp -a /opt/ms-pred /workspace/research/ms-pred
/opt/glacier/bin/python /workspace/research/starter/train.py \
  --input-root /workspace/data --output-dir /workspace/submission \
  --work-dir /workspace/research/fallback-fit --epochs 50 --workers 4
cp -a /workspace/submission /workspace/research/fallback
```

The first completed fallback epoch already publishes a usable package. The
full default fallback is a short contract baseline; its earlier commissioning
time is in `../README.md`. For a setup smoke, use one epoch and explicit subset
limits, and label its score as a smoke result.

The publication adapter needs the five public contract adapters in
`/workspace/references/scripts`. When working from a copied starter, set up its
reference sibling once:

```bash
ln -s /workspace/references /workspace/research/reference
```

## Native training and recovery

```bash
/opt/glacier/bin/python /workspace/research/starter/glacier/train.py \
  --data-root /workspace/data/glacier --source-root /workspace/research/ms-pred \
  --work-dir /workspace/research/glacier-fit \
  --fallback /workspace/research/fallback \
  --submission-dir /workspace/submission \
  --batch-size 8 --workers 2 --accumulate-grad-batches 4 \
  --validate-every 1000 --checkpoint-every 500 --max-seconds 21600
```

Six hours is a configurable starting allocation for this recipe, not a measured
H100 reference duration or the whole 48-hour task budget. Time an initial
`--max-steps 128` run in a separate work directory before committing to a longer
run. Do not shrink the model for this timing probe. Batch size, workers, native
configuration, validation cadence and method are all research choices. The
provided reproducible wrapper uses one GPU; the remaining GPUs can run
independent trials. A multi-GPU trainer can be implemented in the editable code.

For recovery, use the same arguments and add:

```bash
--resume /workspace/research/glacier-fit/last.ckpt
```

During a running or preempted job the latest periodic file is
`/workspace/research/glacier-fit/resume/last.ckpt`. Recovery restores model,
optimizer, scheduler, sampler position, RNG and native MAGMa loss-schedule
counter. `--max-seconds` and `--max-steps` are cumulative limits, not extra time
or steps added at each restart. The wrapper rejects changed model/data/batch
identities on resume; use a new experiment directory for a new recipe.
Completed validation/export time in `result.json` remains charged when resuming
fit state saved earlier. An operator may additionally record previously spent
or conservatively charged interruption time as `elapsed_seconds` in
`budget-ledger.json`; keep that ledger with the experiment. Checkpoint, completed
result and ledger usage are combined using their maximum, never added as fresh
time. The runner's hard campaign deadline remains authoritative, including
setup, interrupted work and any final validation that outlasts the trainer's
cooperative timer.

Validation improvements publish a complete immutable generation and atomically
switch `model_manifest.json`. This works on NFS as well as local filesystems.
The previous generation remains usable if export is interrupted. Do not delete
`.versions` while inference or a snapshot is running. Freeze the selected
package before final transfer; copy the entire directory, including dotfiles.

For a long run, generation history can consume storage. After stopping training
and all inference readers, make a compact frozen copy with
`python /workspace/research/starter/publication.py --source /workspace/submission
--output /workspace/frozen-submission`. It retains the selected complete
generation and research records. Move the old submission aside and put this
completed copy at `/workspace/submission` before finishing. Never prune a live
package whose older generation may still be in use.

Logs, `recipe.json`, full training source, resumable checkpoints and
`metrics.jsonl` remain in the work directory. Preserve them in the experiment
archive. The submission includes model source, training wrapper, configuration,
weights and provenance; inference performs no fitting or downloading.

## Publish for inference

Training automatically publishes with inference batch 64. For a GPU with less
memory, republish the selected checkpoint with a smaller positive integer:

```bash
/opt/glacier/bin/python /workspace/research/starter/glacier/publish.py \
  --checkpoint /workspace/research/glacier-fit/checkpoints/best.ckpt \
  --fallback /workspace/research/fallback \
  --source-root /workspace/research/ms-pred \
  --reference-scripts /workspace/references/scripts \
  --output-dir /workspace/submission \
  --inference-batch-size 8
```

Use the selected checkpoint path recorded by training if it differs from the
example. Omit `--inference-batch-size` for 64. The value is stored in the
published `model_config.json`; it changes only inference batching. Model
parameters, native training batch size, two inference workers, routing,
fallback and training data stay fixed. Stop training before this manual export
so a later automatic publication does not replace the chosen inference setting.
Existing packages retain their saved configuration until explicitly republished.
Measure memory use and runtime on the deployment GPU after lowering the batch;
the reviewer's H100 batch-64 timing does not apply to every override or device.

## Validate a research change

Use `/workspace/evaluation/stage_validation.py` and `score.py` as documented in
`/workspace/evaluation/README.md`. Start with a fixed, diverse validation subset
and keep its query IDs unchanged across comparisons. Then evaluate the complete
validation fold before choosing a final model. Spectrum loss is a cheap training
signal; candidate retrieval Hit@1 is the actual task objective.

Keep a record of the hypothesis, baseline, one changed factor, training data,
seed, validation query-set hash, GPU model/count, measured wall/GPU time and
checkpoint hash. Include failures and negative results. Never choose a model
using the hidden test report. Verify relocated inference while there is still
time to fix packaging and runtime problems.
