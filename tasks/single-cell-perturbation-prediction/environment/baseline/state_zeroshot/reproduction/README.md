# Reproduction record: B2 STATE zero-shot

The runs behind `../RESULTS.md`. STATE has already been trained and scored, so there is no need to
train it again to see what it does.

| file | content |
| --- | --- |
| `full-train.log`, `full-train2.log` | training on all four training contexts (20,000 steps, `state_sm`): data panel, model summary (58.4M trainable parameters), trainer settings, completion. The first attempt is `full-train.log`. |
| `loco-k562-train.log`, `loco-jurkat-train.log` | the same recipe with K562 or Jurkat held out |
| `loco.jsonl` | the LOCO score and every normalised cell-eval metric for the two held-out contexts; HCT116 and HEK293T were not run |

The final-subset numbers in `../RESULTS.md` come from the verifier; no test predictions are included.
