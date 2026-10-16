# Starter: reproduced results

Reproduced in this setup on eight H100s (2026-09-19/20) with the task's development
image, the pinned MP-20 materialization and prepared scorer assets. Each row is one unconditional
request of 2,500 structures (seed 20260919), relaxed with the fixed UMA-200 recipe and scored by
the unchanged LeMat ensemble, exactly as `/workspace/evaluation/score.sh` does.

| generator | SUN + MetaSUN (reward) | SUN | MetaSUN | validity | uniqueness | novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke generator (aggregate statistics, no learning) | 0.106 | 0.0088 | 0.0972 | 0.8724 | 0.7864 | 0.6748 |
| OMatG starter, 2,000 epochs from scratch | **0.208** | 0.0060 | 0.2020 | 0.9552 | 0.9504 | 0.6852 |

Every requested structure is in the denominator. About half of the trained model's score is
reached by the no-learning smoke generator, because the fixed UMA relaxation moves many plausible
compositions onto the metastable side of the hull. Improvements should be judged against both rows.

## OMatG training run

- Recipe: `omg_port/config.yaml` unchanged except the device count, `--trainer.devices=6
  --data.batch_size=85` (effective batch 510, close to the recipe's 8 × 64 = 512), 2,000 epochs,
  seed 20260903.
- Wall time 5.4 h on six H100s (about 10 s per epoch). The best `val_loss_total` (≈ 0.504) came
  near epoch 700; that checkpoint was published.
- Package: 12,403,912 parameters, `model.safetensors` SHA-256
  `e298e7a6fb8e8c044cacff70efa48dddb5fb2f7215069e446edb787457ae940b`.
- Generation of 2,500 structures from the relocated package passed the contract validator with 0
  errors.

## Evaluation cost (2,500 structures)

The OMatG relaxation ran on six GPUs, every other stage on eight.

| stage | OMatG starter | smoke generator |
| --- | --- | --- |
| UMA-200 relaxation | 2,482 relaxed (2,471 converged, 11 unconverged kept), 18 original fallbacks; 74 min | 2,462 relaxed (2,381 converged, 81 unconverged kept), 38 original fallbacks; 79 min |
| LeMat scoring | about 1 h 43 min | about 1 h 26 min |

A 100-structure development batch (`/workspace/evaluation/examples/input`) takes minutes and is
far noisier; uniqueness in particular depends on the batch size.
