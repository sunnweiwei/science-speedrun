# Model-visible data (V4)

Source acquisition, preprocessing, the task-side TUEV split adaptation, and public dataset
download/materialization code are documented in `MATERIALIZE.md`. This directory describes the
released development data interface mounted inside `/workspace/data`.

Four active EEG tracks:
- `tuev`: 6 classes, shape `[16, 5, 200]`
- `ad65`: 3 classes, shape `[19, 10, 200]`
- `openbmi`: 2 classes, shape `[62, 4, 200]`
- `longitudinal`: 2 classes, shape `[58, 2, 200]`

Each track directory contains `train/` and `val/` with `manifest.json` and NumPy shards.
Only these development directories are mounted here; raw sources and held-out test data are absent.
`pretrained/` holds only the two allowlisted initialization checkpoints
(`bci-cbramod.safetensors` and `bci-codebrain.safetensors`), pinned in
`/workspace/baseline/braindecode/pretrained.lock.json` (`BIOT` is excluded in V4 due to `TUEV`
pretraining overlap).

A development manifest sample gives `id`, `path`, `index`, `label`, `subject`, `session` and
`provenance`. Arrays are stored in microvolts; the supplied baseline divides by `100` before
casting to `torch.float32` (do not divide twice). Training labels may be used to fit models;
validation splits are for model selection only. Joint training on all four supplied training
splits is required.

- `ad65` labels are patient diagnoses: `0` Alzheimer's, `1` frontotemporal dementia, `2` healthy control.
- `openbmi` labels: `0` left hand, `1` right hand motor imagery.
- `longitudinal` labels: `0` eyes closed, `1` eyes open.
- `tuev` labels: `0..5` (`SPSW`, `GPED`, `PLED`, `EYEM`, `ARTF`, `BCKG`). Training and validation
  are partitioned by subject (`80/20` over 290 development subjects: `68,445` train / `15,487` val),
  while `test` (`29,421` windows) uses the provider's official patient-disjoint evaluation split.
