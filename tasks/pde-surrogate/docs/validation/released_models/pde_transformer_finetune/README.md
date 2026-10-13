# Reference: PDE-Transformer fine-tuned on the transfer datasets (the paper's recipe)

The paper's headline transfer result fine-tunes the pretrained separate-channel
PDE-S model on each Well dataset. Its subsets were never published (see
`../../data/AUDIT.md`), so this reference runs the same recipe on the task's
own split:

1. `convert_pools.py` writes the task's train/valid/held-out trajectories as
   PBDL files (`<name>_train/_valid/_test.hdf5`) with upstream's metadata.
2. `make_lightning_ckpt.py` wraps the released `sc-s` weights (MIT, pinned) as
   a Lightning checkpoint for upstream's `ckpt_path`.
3. `finetune_trl2d.yaml` + `env_local.yaml` drive upstream's `main.py`
   (tum-pbs/pde-transformer @ `850e09d3`, MIT): one-step supervised MSE,
   AdamW 4e-5, EMA 0.999, batch 8, bf16, non-periodic (the layer is open in y).
4. The fine-tuned model is replayed on the task's held-out cases through
   upstream's own rollout (`Supervised.predict`), exported into the contract's
   case files, and scored with the task grader.

Epoch count is this task's choice (the paper does not state its fine-tuning
schedule); it is recorded with the numbers. `run_finetune_trl2d.sh` is the
launcher used on the contributor's A100.

### Per-dataset deviations from the recipe

Each is forced by the hardware or by the dataset, and each is repeated in the
validation JSON for the run it affects.

| dataset | deviation | why |
| --- | --- | --- |
| turbulent radiative layer | none | the recipe runs as written |
| active matter | batch 2 with 8 accumulation steps (same effective batch 16); EMA callback removed | 11 channels at 256² do not fit a 40 GB card at batch 8, and upstream's EMA raises `Tensor list must have at least one tensor` at the first optimizer step once the batch is reduced |
| PDEArena Navier-Stokes | `"buoyancy"` appended to upstream's constant vocabulary; `test_unrolling_steps` 29 → 12 | the vocabulary has no buoyancy entry, so `convert_constants` raises; appending is what upstream's own note sanctions, the class embedding table has 1000 rows so the released weights load unchanged, and index 36 is simply an untrained row. Trajectories are 14 frames long, so a 30-frame test window cannot be cut; training and validation use `unrolling_steps`, and upstream's test loop is unused here |

## Exporting a fine-tune as a package

`export_package.py` turns a Lightning checkpoint from the run above into a
`pde-surrogate-rollout/v1` package:

```
python3 export_package.py --ckpt finetune/logs/ft-trl2d/checkpoint/epoch=000036.ckpt \
    --config finetune_trl2d.yaml --dataset-name well/turbulent_radiative_layer_2D --out pkg_ft_trl2d
bash pkg_ft_trl2d/infer.sh INPUT_DIR OUTPUT_DIR
```

The package keeps upstream's model and rollout: it stores the checkpoint's
weights and the resolved `model:` block of the training config, rebuilds
`pdetransformer.core.separate_channels.Supervised` with
`instantiate_from_config`, and calls its own `predict()`. Only the batch dict
that upstream's dataset would supply is assembled here, from the task's case
files and the same metadata table `convert_pools.py` uses.

Four details cost time and are worth knowing before touching this file. The
training config interpolates top-level values (`${patch_size}`), so it has to
be resolved with OmegaConf after the training-only blocks are dropped. `Fields`
must carry a batch dimension: a one-dimensional tensor is read as one channel
id per sample, which silently sets the batch size to the channel count. And
`Supervised.predict` returns `(frames, reference)` with frames shaped
`(batch, num_frames, channels, x, y)`, of which the last K are the forecast.

The fourth is the one that actually moved a number. The `Fields` and
`Constants` entries of `physical_metadata` are indices into two fixed
vocabularies in `pdetransformer/data/metadata_remapping.py`, and the training
pipeline encodes them with `convert_fields` / `convert_constants` — they are
not positional. Turbulent-radiative-layer fields are classes `[5, 6, 1, 2]`,
not `[1, 2, 3, 4]`, and its one constant is class `21`, not `1`. An export that
numbers them positionally loads the right weights and runs without error while
conditioning the model on embeddings the fine-tune never saw. It cost 0.53
VRMSE on the active-matter group (2.1983 positional against 1.6693 correct) and
was caught only by comparing the two. `export_package.py` now calls upstream's
converters, and `setup.json` records `constant_names` beside the classes so the
mapping is visible in the package.

Plumbing verified on 2026-09-18 with an interim epoch-36 checkpoint on two
turbulent-layer cases through the sealed verifier: 0 failures, a finite score.
That is a wiring check, not a result; the reported numbers come from the
completed fine-tunes.
