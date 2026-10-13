# Baseline: starter (a from-scratch patched transformer)

A minimal but complete pipeline, about 6.4M parameters at the default size. Reproduced reward
**0.161** (`RESULTS.md`). The corpus pack and every run go under `/workspace/research`.

```
python3 data.py pack --data /workspace/data/gifteval_pretrain --out /workspace/research/packed [--tier fast]
python3 train.py --packed /workspace/research/packed --out /workspace/research/runs/r0 --steps 30000 --batch 64 --lr 5e-4 --context 1024 --horizon 128 --patch 32 --synthetic-fraction 0.1 --valid /workspace/data/valid --seed 0   # the recorded run: 6 min on an H100, 16 on an A100-40GB
python3 publish.py --weights /workspace/research/runs/r0/best.pt --out /workspace/submission --sources-from /workspace/research/packed [--synthetic kernelsynth]
bash /workspace/submission/infer.sh INPUT_DIR OUTPUT_DIR
```

- `data.py` packs the Arrow corpus into a float32 memmap (multivariate series
  flattened per variate; series shorter than 64 points dropped, which removes
  every yearly series in the allow-list) and samples
  random (context, horizon) windows uniformly over series.
- `model.py`: contexts standardized by their finite mean / std, patches of 32
  observations embedded with their missing-value mask, a causal transformer
  (d 256, 6 layers), and a head that emits the next 128 steps as nine
  quantiles plus a mean at every position; longer horizons feed the median
  back autoregressively. Quantile crossing is removed by sorting.
- `train.py`: pinball loss on the nine levels plus 0.5 × Huber loss on the mean (standardized targets clipped to ±20), in
  standardized units, at every patch position; AdamW, cosine schedule, bf16.
  Every `--eval-every` steps it forecasts a subset of the validation configs
  and scores them with the task grader against the validation Seasonal Naive
  reference; `best.pt` is the best validation skill (the recorded run's best
  was step 25,000 of 30,000). `--valid` must point at a staged validation
  split; a missing one is an error, not a silent run without selection.
- `synth.py`: a KernelSynth-style Gaussian-process generator (Chronos §4.2),
  pre-generated as a bank of `--synthetic-bank` series at start-up (a few minutes on 8 cores) and mixed in with `--synthetic-fraction`. Allowed by the rules as self-generated
  data; declare it as `synthetic:kernelsynth`.
- `predict.py` / `infer.sh` / `publish.py`: contract inference (instances
  flattened per variate, batched by context length) and packaging with a
  manifest that records the config, step and weights hash.

Validation during training uses a small stratified subset of the validation
configs and the task's *validation-split* Seasonal Naive reference (the
board's numbers exist only for the test split), so validation skill is a
selection signal on a different scale from the reward: the shipped baseline
reached validation skill −0.08 on 15 stratified configs and reward 0.153 on
the 97 test configs, where the 480–900-step horizons dominate.

Obvious next steps for an agent: larger model and context, a proper output
head for long horizons instead of autoregression, better sampling across
frequencies, multivariate attention, and the synthetic mix.
