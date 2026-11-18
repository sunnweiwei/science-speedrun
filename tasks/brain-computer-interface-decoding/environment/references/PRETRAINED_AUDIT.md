# Pretrained initialization audit (V4)

The active allowlist is `baseline/braindecode/pretrained.lock.json`: immutable Hub
revisions, filenames, SHA-256 values, published license labels and source
Braindecode versions. Only allowlisted files may be mounted read-only for the agent,
and `tests/main_verifier.py` enforces this policy at evaluation time.

## V4 De-Contamination: Removal of `BIOT`

In SSI Benchmark V4, **`BIOT` (`braindecode/biot-pretrained-six-datasets-18chs`, SHA-256
`a218f4054ae29962c6fd9d23d94c446d354c4f53e805a6771c85749ccc20baaa`) is removed from
authorized pretrained weights and explicitly blocked** (`disallowed_models` in
`pretrained.lock.json`, enforced in `pretrained.py`, `validate_pretrained.py`, and
`tests/main_verifier.py`). Upstream documentation for the six-dataset BIOT checkpoint
([ycq091044/BIOT](https://github.com/ycq091044/BIOT#2-eeg-pre-trained-models-all-re-sampled-to-200hz))
states that it was pretrained directly on labeled/unlabeled splits of **TUEV** alongside
MGH/PREST, SHHS, TUAB, CHB-MIT and IIIC, without a verifiable recording-level manifest
proving zero patient/recording overlap with our frozen `TUEV` evaluation cohort.

| Family | Status in V4 | Initialization / published pretraining source | Overlap audit note |
| --- | --- | --- | --- |
| EEGNet | **Authorized** | Random initialization | Zero external pretraining overlap |
| ContraWR | **Authorized** | Random initialization | Zero external pretraining overlap |
| CBraMod | **Authorized** | Self-supervised TUEG pretraining ([Wang et al., ICLR 2025](https://openreview.net/pdf?id=NPNUHgHF2w)) | Zero overlap with AD65, OpenBMI, and Longitudinal; unlabeled TUEG corpus provenance noted below |
| CodeBrain | **Authorized** | Self-supervised TUEG pretraining ([Ma et al., ICLR 2026](https://arxiv.org/html/2506.09110v2)) | Zero overlap with AD65, OpenBMI, and Longitudinal; unlabeled TUEG corpus provenance noted below |
| BIOT | **DISALLOWED (Removed in V4)** | Six-dataset checkpoint including TUEV | **Prohibited**: direct downstream `TUEV` pretraining corpus overlap |

The Braindecode Hub cards do not include per-recording pretraining manifests for
unlabeled TUEG; private task test labels are never used in joint training or
checkpoint selection.

The pinned [CBraMod](https://huggingface.co/braindecode/cbramod-pretrained/tree/584cdc415913739a05d84bf0c1cb3db397764507)
and [CodeBrain](https://huggingface.co/braindecode/codebrain-pretrained/tree/1ba45893685e6d5d2d51967cb48b1992c61668ef)
Hub releases declare BSD-3-Clause (original repositories: CBraMod MIT, CodeBrain Apache-2.0).
No checkpoint is redistributed in Git.

## Tensor compatibility and adaptation

`pretrained.py` validates the complete file hash against `pretrained.lock.json`, rejects
any disallowed family (`biot`), and requires every backbone state tensor by exact key and
shape with `strict=True`. Missing, mismatched or unexpected core tensors stop the run.
The published `final_layer.*` is removed and replaced with an identity before loading.
Successfully loaded core tensor counts are CBraMod 211 and CodeBrain 269, including buffers.

All four task-specific channel projections (`tuev`, `ad65`, `openbmi`, `longitudinal`) and
four classifier heads are newly initialized. The shared input representation uses 18 channels
(19 for CodeBrain) at 200 Hz in chronological 2-second chunks. `training.json` and
`config.json` retain the loaded revision, hash, counts and initialization mode.

## Operator reproduction

Before starting an agent, stage ONLY the two allowlisted `model.safetensors` files from
`pretrained.lock.json` into `$TASK_DATA_DIR/pretrained/` (mounted at
`/workspace/data/pretrained/`) as `bci-cbramod.safetensors` and
`bci-codebrain.safetensors` (ensure `bci-biot.safetensors` is NOT present), then run:

```bash
python environment/baseline/braindecode/validate_pretrained.py "$TASK_DATA_DIR/pretrained"
python tests/test_shared_matrix.py --checkpoint-dir "$TASK_DATA_DIR/pretrained"
```

Use the task's pinned Braindecode 1.8.1 dependencies. The agent has no network
checkpoint retrieval permission. Keep downloaded assets outside the repository.
