# Baseline: starter (ESM-2, pretrained and fine-tuned from scratch)

A **runnable** starting point, reproduced at reward 0.6695 / 0.6774 (seeds 0 / 1; `RESULTS.md`): `train.py` pretrains an ESM-2 encoder on the
UniRef50 corpus, fine-tunes it jointly across all eleven tasks, selects a
checkpoint on the labeled `tune` split, and writes a package that satisfies
`/workspace/contract.md`. It is deliberately plain — the point is that
the tedious parts already work, so research can start from something that runs.

```bash
# quick pass, minutes, to see the whole path work
python3 train.py --data /workspace/data --out /workspace/research/try \
    --pretrain-hours 0.05 --finetune-hours 0.05 --corpus-limit 20000

# the recorded run (RESULTS.md), then publish it: publish.py checks the package the way the verifier
# will and swaps it into /workspace/submission with one rename
python3 train.py --data /workspace/data --out /workspace/research/seed-0/submission \
    --model esm2_t6_8M --pretrain-hours 8 --finetune-hours 3
python3 publish.py /workspace/research/seed-0/submission --to /workspace/submission

# a second seed of the same recipe (default --seed 0; recorded in the package)
python3 train.py --data /workspace/data --out /workspace/research/seed-1/submission \
    --model esm2_t6_8M --pretrain-hours 8 --finetune-hours 3 --seed 1
```

## What is here

| File | Purpose |
| --- | --- |
| `esm2.py` | The ESM-2 architecture — rotary position embeddings, pre-LN blocks, tied LM head, the ESM alphabet, and the published size ladder in `CONFIGS`. Run it directly to print parameter counts. |
| `train.py` | MLM pretraining, joint multi-task fine-tuning, `tune`-split model selection, and packaging. |
| `predict.py` | Inference for a trained package; copied into the submission by `train.py`. |
| `publish.py` | Atomically swap a candidate into `/workspace/submission`, after checking it the way the verifier will. |
| `../trivial/` | A fixed-statistic predictor that trains nothing: the output-format demo and the control to beat (a separate baseline). |

Available sizes (`--model`), backbone parameters excluding heads:

| name | layers | dim | backbone | 1-GPU throughput | one epoch |
| --- | --- | --- | --- | --- | --- |
| `esm2_t6_8M` | 6 | 320 | 7.41M | 495K tok/s | 5.6 h |
| `esm2_t12_35M` | 12 | 480 | 33.27M | 198K tok/s | 14.0 h |
| `esm2_t30_150M` | 30 | 640 | 147.73M | 68K tok/s | 41.0 h |
| `esm2_t33_650M` | 33 | 1280 | 649.40M | — | — |

Throughput measured on one A100-80GB with the default token budget.

## Resuming across experiments

A research loop should not repay for pretraining every time it wants to try a
different downstream recipe. `--checkpoint-dir` writes resumable checkpoints
(model, optimizer, elapsed budget, step, tokens seen); `--resume` continues from
one, detecting from the checkpoint whether it was mid-pretrain or mid-fine-tune.

```bash
# spend the pretraining budget once, keeping checkpoints
python3 train.py --data /workspace/data --out /workspace/research/run-1/submission \
    --checkpoint-dir /workspace/research/run-1/ck --pretrain-hours 8 --finetune-hours 3

# reuse that encoder for a different fine-tuning recipe, no pretraining repeat
python3 train.py --data /workspace/data --out /workspace/research/run-2/submission \
    --resume /workspace/research/run-1/ck/pretrain.pt --skip-pretrain \
    --checkpoint-dir /workspace/research/run-2/ck --finetune-hours 3

# continue a run that was interrupted
python3 train.py --data /workspace/data --out /workspace/research/run-1/submission \
    --resume /workspace/research/run-1/ck/pretrain.pt \
    --checkpoint-dir /workspace/research/run-1/ck --pretrain-hours 8 --finetune-hours 3
```

Both phases always leave a checkpoint when they end, not only on the periodic
cadence, so a cleanly finished pretraining run is always reusable. Checkpoints
are written atomically — a preempted job never leaves one that looks loadable
and is not.

Resuming a checkpoint from a *different* architecture is allowed and reports
what happened rather than failing: tensors whose shapes match are transferred,
the rest stay at their initialization, and optimizer state is dropped because
its momentum means nothing for different parameters.

**The learning-rate schedule follows elapsed budget, not a step count.** That is
deliberate: both phases stop on wall clock, so a schedule keyed to a guessed
step count desynchronizes — overshoot the guess and the cosine pins at zero,
and training silently stops improving while still consuming the budget. If you
rewrite the loop, keep the schedule on the same clock that ends it.

## Publishing in a running loop

The submission directory is read at unpredictable times while training
continues, so never copy a candidate into it file by file — a verifier reading
mid-copy would see weights without code, or the reverse.

```bash
python3 publish.py /workspace/research/run-17/submission --check  # validate only
python3 publish.py /workspace/research/run-17/submission          # atomic swap
```

It rejects packages the verifier would reject (missing `infer.sh` or
`README.md`, symlinks escaping the package, or an experiment archive rather
than a single candidate), and also rejects training *state* that has no role in
replay — `optimizer.pt`, `pretrain.pt`, `finetune.pt` belong under
`/workspace/research`. Training *provenance* is kept: `training_history.json`
is a few KB and records the model-selection trace the contract asks you to
document. It then stages and renames in a single step.

## Things worth knowing before you tune

**Batch by tokens, not by sequences.** Over this length distribution (mean 265,
p99 1659) a fixed 64-sequence batch pads to its longest member and wastes 3.4×
of the compute. `build_batches` sorts by length and fills a token budget
instead, which measured 4× faster end to end. ESM-2 does the same.

**The token budget is already near its useful ceiling.** Raising
`--token-budget` from 32,768 to 524,288 grows peak memory from 2.9 GB to
44.8 GB but throughput only from 493K to 568K tok/s — 16× the batch for 15%
more speed, saturating around 65,536. At 8M scale the GPU is compute-bound, not
occupancy-bound, so spend tuning effort elsewhere. Larger models shift this
tradeoff.

**You cannot exhaust the corpus, and you do not need to.** Pretraining samples
minibatches; an incomplete pass over a large corpus is better than repeated
passes over a small one. The real decision is how to split the budget between
pretraining and supervised training, and at what model size — measure it.

**Model selection is part of the contract.** `train.py` scores `tune`
periodically with the grader's own `metrics.py` and keeps the best checkpoint.
If you change the training loop, keep an explicit selection policy and document
it in the submission README.

## Limitations of the starter

One training run per `--seed`, no ensembling, no hyper-parameter search, no gradient accumulation,
no distributed training. Sequences are truncated to 1024 tokens and
`secondary_structure` predictions beyond that are filled with a uniform score
vector — which is a real accuracy loss on long chains and an obvious thing to
improve. The heads are minimal MLPs over a mean-pooled representation; pair
tasks simply concatenate two pooled sequences, which throws away any
cross-sequence attention.
