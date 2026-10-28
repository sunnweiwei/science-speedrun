# BCI five-track shared submission v2

Keep one complete package under /workspace/submission, including infer.sh,
architecture configuration, code and weights. One trainable EEG backbone must
serve all five tracks; input adapters and classification heads may be task-specific.
The backbone must be trained jointly on all five training splits. Initialization
may be random or from the task-provided audited checkpoint allowlist. Package
contents must not contain symlinks. The package must run after relocation.

Invocation: bash infer.sh INPUT_DIR OUTPUT_DIR.
INPUT_DIR/manifest.json contains schema=bci-input-v1, track, config, and samples.
Each sample is {"id":"opaque","path":"shard-000000.npy","index":0}.
NumPy shards have shape [N,C,S,200], in microvolts. Track C/S:
faced 32/10, tuev 16/5, ad65 19/10, openbmi 62/4, longitudinal 58/2.
The baseline scales input by 1/100. No labels, subjects or source filenames are
part of test inputs. Support unseen IDs, arbitrary order and sample counts.

Write OUTPUT_DIR/predictions.jsonl, exactly one row per ID:
{"id":"opaque","probabilities":[0.2,0.8]}.
Class counts: 9/6/3/2/2 respectively. Values must be finite in [0,1],
sum to 1 within 1e-5, and use the documented class order. Missing, duplicate,
unknown IDs or invalid probabilities invalidate that track (reward 0).

Runtime: PyTorch 2.8.0, dependencies in requirements.txt, one GPU, 64 GiB RAM,
two hours maximum per track, no network, unprivileged user, package/input
read-only and output writable. Inference cannot use labels, external caches or
workspace files. No training/adaptation on evaluation batches.

Each track is independently scored in [0,1]; no scalar aggregate.
FACED/TUEV: balanced accuracy, with weighted F1 and Cohen kappa.
AD: arithmetic mean of window probabilities per subject, then subject-level
balanced accuracy (window metrics additionally reported).
OpenBMI: mean of subject-level balanced accuracies over all each subject's trials.
Longitudinal: balanced accuracy per subject/session, mean across sessions within
subject, then mean across subjects. Infrastructure failures produce no reward.
