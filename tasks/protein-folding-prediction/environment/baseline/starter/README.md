# Baseline: starter (approximately-50M ESMFold2 graph, trained from scratch)

This directory starts from a serious, executable folding architecture rather
than a toy network. `esmfold2_50m/` ports the validated Biohub ESMFold2 graph to
the task's final-hidden feature contract:

- one all-atom model for every FoldBench track;
- 45M--55M learned parameters (the exact audited count is printed by
  `python -m esmfold2_50m.audit`);
- evaluator-supplied `[L, 2560]` ESMC-6B final normalized hidden states;
- a learned-from-scratch input projection, all-atom input embedder, pair trunk,
  PARCAE/CODA recurrence, diffusion structure module, and confidence heads;
- no MSA, templates, ESMC runtime, or inherited folding-model parameters.

This is the primary development baseline (reproduced results: `RESULTS.md`). It is intentionally trainable and
modifiable: researchers should improve the architecture, data curriculum,
objectives, optimization, inference, and evaluation while retaining the one
45M--55M candidate contract.

## First executable checks

From `/workspace/baseline/starter` inside the development container:

```bash
python -m esmfold2_50m.audit
python -m reference.build_reference_index \
  --manifest /workspace/data/manifests/foldingtrain-100k-v1/foldingtrain_100k.jsonl \
  --output-dir /workspace/research/esmfold2-50m/index
MAX_STEPS=1 CHECKPOINT_INTERVAL=1 NPROC_PER_NODE=8 ./train.sh
```

`train.sh` starts the checkpoint publisher before training. Step 1 and every
subsequent completed checkpoint are converted into one atomic, independently
replayable `/workspace/submission`. The submission contains one inference
package and one current weight file; native checkpoints, optimizer state,
metrics, ablations, and older candidates stay under `/workspace/research`.

For a longer run, omit `MAX_STEPS`; set `RESUME` to a native checkpoint when
continuing an interrupted training process. The starter defaults are an
executable baseline, not a prescribed final research strategy.

## Replay memory behaviour

`predict.py` serves every request independently: a request the model cannot
complete (in practice CUDA out of memory on the largest assemblies) is logged
to `OUTPUT/failures.jsonl` and the batch continues, so the scorer counts that
target as a missing prediction instead of the whole replay failing. Above
`low_memory_tokens` (2,480 tokens after chain copies; `model_config.json`)
the runtime switches from Biohub's fused triangle kernels, which materialize
whole L x L buffers, to the reference kernels with a row-chunked triangle
update, and it retries once on that path after an out-of-memory error. The
fused path replays a 2,468-token assembly in about 13 s on an H100 80 GB; the
low-memory path takes minutes per request and reaches about 2,600 tokens.
Requests above `replay_max_tokens` (2,700; override with
`FOLDBENCH_REPLAY_MAX_TOKENS`) are recorded as failures without a forward
pass — roughly twenty suite assemblies (up to 4,693 tokens) exceed 80 GB
elsewhere in the trunk whatever the kernels — since each doomed attempt costs
minutes; raise both limits on larger GPUs. Reducing the footprint itself is
part of the research space, not something the evaluator works around.

## Layout

- `esmfold2_50m/`: model, final-hidden adapter, training recipe, DDP trainer,
  offline runtime, and parameter audit.
- `train.sh`, `publish_latest.py`, `publish.py`: continuous native checkpoint to
  submission publication path.
- `infer.sh`, `predict.py`, `model.py`: evaluator entry point and loader.
- `reference/`: compact loaders, losses, training plumbing, and model smoke
  fixtures from the earlier project. These remain useful for fast diagnostics,
  but their small pairformer model is not the main 45M--55M candidate.
- `folding_contract.py`, `structure_to_mmcif.py`: request validation and generic
  all-atom mmCIF helpers.

The primary architecture uses the installed pinned Biohub ESM and Transformers
sources and the pinned CCD asset in `/opt/esmfold2-assets`. Additional runnable
and study references are documented in `/workspace/references/README.md`.

## Baseline-weight boundary

`/workspace/data/models/esmfold2-fast-b28d8ace` is available for inspection, supplied
baseline results, and compatible data analysis only. It expects an ESMC layer
mixture plus its released projection, so it cannot be faithfully replayed from
the task's final-hidden-only tensors. Do not initialize the candidate from it,
call it during candidate inference, or publish any of its parameters.
