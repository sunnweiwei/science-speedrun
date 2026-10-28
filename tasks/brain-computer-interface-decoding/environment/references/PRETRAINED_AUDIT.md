# Pretrained initialization audit

The allowlist is `baseline/braindecode/pretrained.lock.json`: immutable Hub
revisions, filenames, SHA-256 values, published license labels and source
Braindecode versions. Only these files may be mounted read-only for the agent.
Weight integrity and core compatibility have been checked; this is not a
claim that pretraining records were independently reconstructed.

| Family | Initialization / published pretraining source | Overlap limitation |
| --- | --- | --- |
| EEGNet | Random initialization | No external pretraining |
| ContraWR | Random initialization | No external pretraining |
| CBraMod | TUEG, as described in the [original paper](https://openreview.net/pdf?id=NPNUHgHF2w) | TUEG and downstream TUH subsets can overlap; exact records for the rehosted weights are not supplied |
| BIOT | MGH/PREST, SHHS, and training splits of TUAB, TUEV, CHB-MIT and IIIC, as described for the [six-dataset checkpoint](https://github.com/ycq091044/BIOT#2-eeg-pre-trained-models-all-re-sampled-to-200hz) | Explicit TUEV pretraining exposure; a training-split claim does not prove disjointness from this task's frozen split |
| CodeBrain | TUEG, as described in the [original paper](https://arxiv.org/html/2506.09110v2) | No per-record pretraining manifest for the rehosted weights; downstream TUH overlap cannot be excluded |

The Braindecode Hub cards do not include detailed conversion or pretraining
manifests. The corpus descriptions above are author-reported provenance, not
an independently verified mapping of every Hub tensor to a pretraining run.
We do not claim strictly unseen test signals for any pretrained family.
Private task test labels are never used in joint training or checkpoint selection.

The pinned [CBraMod](https://huggingface.co/braindecode/cbramod-pretrained/tree/584cdc415913739a05d84bf0c1cb3db397764507),
[BIOT](https://huggingface.co/braindecode/biot-pretrained-six-datasets-18chs/tree/7e9aabf302870012f73773276803aab97993951c),
and [CodeBrain](https://huggingface.co/braindecode/codebrain-pretrained/tree/1ba45893685e6d5d2d51967cb48b1992c61668ef)
Hub releases declare BSD-3-Clause. Original implementation repositories carry
their own licenses (CBraMod and BIOT: MIT; CodeBrain: Apache-2.0); the Hub license
label does not change dataset access terms. No checkpoint is redistributed in Git.

## Tensor compatibility and adaptation

`pretrained.py` validates the complete file hash, then requires every backbone
state tensor by exact key and shape with `strict=True`. Missing, mismatched or
unexpected core tensors stop the run. The published `final_layer.*` is removed;
BIOT's obsolete `encoder.index` buffer is also excluded. The selected classifier
is replaced with an identity before loading. Successfully loaded core tensor
counts are CBraMod 211, BIOT 56 and CodeBrain 269, including buffers.

All task-specific channel projections and five classifier heads are newly
initialized. The shared input representation uses 18 channels (19 for CodeBrain)
at 200 Hz in chronological 2-second chunks, retaining all channels within each
chunk. Pooling over chunks and replacing the published classifier are task-side
adaptations. This differs from the published downstream training recipes and
is not a reproduction of their reported accuracy. `training.json` and
`config.json` retain the loaded revision, hash, counts and initialization mode.

## Operator reproduction

Before starting an agent, download the three pinned `model.safetensors` files
from the revisions in the lock file into `$TASK_DATA_DIR/pretrained/` (mounted at
`/workspace/data/pretrained/`) as
`bci-cbramod.safetensors`, `bci-biot.safetensors`, and
`bci-codebrain.safetensors`, then run:

```bash
python environment/baseline/braindecode/validate_pretrained.py "$TASK_DATA_DIR/pretrained"
python tests/test_shared_matrix.py --checkpoint-dir "$TASK_DATA_DIR/pretrained"
```

Use the task's pinned Braindecode 1.8.1 dependencies. The agent has no network
checkpoint retrieval permission. Keep downloaded assets outside the repository.
