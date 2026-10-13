# Delphi training reference

This is a pinned, offline-runnable copy of the core training implementation from [`gerstung-lab/Delphi`](https://github.com/gerstung-lab/Delphi) at commit `fb72166be6b29d8db819227a59487e51c1df1454`.

Delphi is a modified GPT-2 model for longitudinal disease-event and event-time modeling. The included synthetic trajectories are the upstream coding-test dataset, not UK Biobank data and not benchmark data.

## Quick smoke test

Inside the task image:

```bash
bash /workspace/references/delphi/smoke_train.sh
```

This runs one optimization step on CPU with a small model and writes output under `/tmp`, proving that the data loader, forward/backward pass, optimizer, and checkpoint path are usable.

## Demo training

```bash
cd /workspace/references/delphi
bash train_demo.sh --device=cuda --out_dir=/workspace/research/Delphi-demo-2M
```

Any additional `train.py` overrides can be appended. For example:

```bash
bash train_demo.sh --device=cuda --max_iters=100 --eval_interval=25
```

The original demo config uses 12 layers, 12 heads, embedding size 120, vocabulary size 1,270, and 5,000 training iterations.

## Data format

`train.bin` and `val.bin` contain flat `uint32` triples `(patient_id, age_in_days, token_id)`. The scripts resolve paths relative to this directory, so run them through the wrappers or change into this directory first.

See `PROVENANCE.md` and `LICENSE.upstream` for origin and licensing. This upstream reference is not an adapter for the benchmark probability interface and includes no benchmark-trained checkpoint.
