# References

Measured runs and model specifications from the task author. Only the starter recipe has been
reproduced in this setup (`/workspace/baseline/starter/RESULTS.md`); treat the rest as
context, not as checked results. Read the executable starter first, then use the measured runs
here to decide what to keep, change, or replace.

## Recommended starting point

`/workspace/baseline/starter/train.py` is runnable and produces a contract-compliant
package end to end: masked-language pretraining on the provided UniRef50
corpus, joint fine-tuning across all eleven tasks, model selection on the
labeled `tune` split against the grader's own metric, and packaging. The
architecture in `starter/esm2.py` is the published ESM-2 design — rotary
position embeddings, pre-LN blocks, tied LM head, the ESM alphabet — with the
released size ladder (`esm2_t6_8M` through `esm2_t33_650M`) available via
`--model`.

Nothing about it is privileged. Replace any part of it.

## Measured runs

[measured_runs.md](measured_runs.md) records what has actually been trained and scored
under this task's budget, with the provenance needed to reproduce or beat it:
configuration, tokens seen, throughput, wall clock, model-selection trace, and
per-task rewards on the held-out split.

Two things to take from it rather than re-deriving:

- **The trivial control scores 0.2874.** A constant predictor collects most of
  the accuracy-based tasks, so a reward near 0.3 means nothing has been learned.
  Always read a score next to that number.
- **`fold` is where the headroom is.** The reference reaches 0.1212 over 1,195
  classes while most other tasks are past 0.65. It is the task most likely to
  separate serious attempts.

## What is deliberately absent

No pretrained protein weights, no ESM/ProtT5/Ankh checkpoints, and no
embeddings computed by another model are supplied, because the contract forbids
them. Every learned parameter in a submission must originate from random
initialization in the session. `esm2.py` is an *architecture*, not a
checkpoint — building an ESM-2-shaped model is allowed and expected; loading
Meta's released ESM-2 weights is not.
