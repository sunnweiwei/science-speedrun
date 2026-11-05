# DrugCLIP agent starting point

This directory contains the maintained code, the short training launcher and the
[historical training and inference recipes](TRAINING.md). It is available at
`/workspace/baseline/drugclip/` in the offline development container.

The reference contains 95,305,418 parameters and supports the supplied
generic Uni-Mol molecule initialization, ProFSA pretraining, BioLiP fine-tuning,
dev BEDROC checkpoint selection, resumable optimization, and standalone export.

For a short wiring check in a fresh workspace:

    bash /workspace/baseline/drugclip/train.sh
    python /workspace/publish.py /workspace/research/candidate

This is eight records and one epoch per stage, not a performance baseline.
The training process writes recipe.json, metrics.jsonl, last.pt, and
training_summary.json under its work directory. train.py --help documents
full-data settings, separate stages, torchrun, and resume. Use fresh work
directories for changed recipes. Use BioLiP dev for checkpoint selection.

Development retrieval metrics are computed by validation.py during training;
screening EF is reserved for the separate hidden tests. An exported package
can also be evaluated without retraining using `/workspace/evaluation/dev.py`;
see the [development evaluation instructions](../../evaluation/README.md). The published package's
infer.sh consumes the label-free test-shaped LMDB view in contract.md.
For a local interface replay, create a synthetic test-shaped LMDB or derive
one exclusively from BioLiP dev. Do not obtain the hidden test data.

RESULTS.md records the GPU-trained reference's current full-suite reward
0.166385 (dataset-balanced BEDROC) and raw mean EF1% 16.108
(epoch 14, selected by dev BEDROC 0.4288), its 34-epoch development curve and
reported A100 recipe under reproduction/. This is separate from the historical
tiny-data pilot and benchmark-trained author-checkpoint control. The short
launcher above is a wiring check, not the complete-data recipe. The released epoch-14 weights have also been replayed on all 117 targets
through the independent verifier. See RESULTS.md for the verified score.
