# Screening model delivery contract

## Package

Keep the current selected package at /workspace/submission. It contains
infer.sh, owned inference code and weights, model configuration,
model_manifest.json, and README.md. The manifest declares method, positive
parameter_count, model_config (package-relative path), and checkpoint
(package-relative path and SHA256). The supplied starters use model/v1 or
model/v2 schemas; other methods may extend the manifest with these same
required fields. No manifest paths may escape the package. Symlinks and
special files are forbidden.

Use one shared learned parameter set on both benchmarks and all targets.
There is no parameter-count cap. The allowed generic Uni-Mol initialization
becomes owned candidate weights; its original files are unavailable at replay.
Author benchmark-trained weights are not permitted. Exclude optimizer states,
research history, test answers, and references to the live trainer.

README.md states the model, exact learned parameter count, initialization,
training provenance, data use, selection method, and limitations.

## Entry point and inputs

The verifier runs:

    bash <package>/infer.sh <input-root> <output-root>

The working directory is the package. Output starts empty and is writable.
Package and inputs are read-only. No network, research directory, original
pretrained checkpoint, or labeled benchmark is accessible to inference.

    <input-root>/test/dude/<target>/mols.lmdb
    <input-root>/test/dude/<target>/pocket.lmdb
    <input-root>/test/lit_pcba/<target>/mols.lmdb
    <input-root>/test/lit_pcba/<target>/pockets.lmdb

Each is a single-file LMDB with contiguous ASCII integer keys from zero and
pickled dictionaries. Molecules contain only atoms, coordinates (a list of
conformers), and smi. Pockets contain only pocket_atoms and pocket_coordinates.
Coordinates are in Ångströms. Pocket atom names may be PDB/MOL2 names such as
CA or N.am; preprocessing must distinguish atom names from elements.
Labels, source paths, compound IDs, and benchmark metadata are removed.
Molecules are privately shuffled independently per target and assigned fresh
indices. No original index or permutation seed is supplied. Pocket order
is preserved; scoring maps these view indices to private labels.

Handle all molecules and all supplied pockets. The reference max-pools raw
similarities over LIT-PCBA pockets and median/MAD-normalizes each DUD-E pocket
before max pooling. Other general scoring methods are allowed. Do not change
learned parameters during replay.

## Outputs

Write a regular UTF-8 scores.csv with exactly these columns:

    benchmark,target_id,ligand_index,score

benchmark is dude or lit_pcba; target_id is the input directory name;
ligand_index is the numeric index in the supplied shuffled LMDB; score is a finite real number, higher
meaning more likely active. Keep each target's rows contiguous, with indices
0 through N-1 in order. Score every ligand in every supplied target exactly
once. An optional prediction_manifest.json is not trusted as proof of coverage.

## Resources and scoring

Replay has one H100 GPU, 16 CPUs, 64 GiB host RAM, and a 22-hour inference
timeout, independent of the development budget. All 102 DUD-E and 15 LIT-PCBA
targets are evaluated. Coverage comes from the frozen LMDBs.

Scoring protocol: drugclip-dataset-bedroc/v2. For each target, calculate
BEDROC(alpha=80.5). Average the 102 DUD-E values and the 15 LIT-PCBA values
separately, then average those two numbers with equal weight:

    score = reward = 0.5 * mean_DUDE(BEDROC_80.5) + 0.5 * mean_LIT_PCBA(BEDROC_80.5)

Both `score` and `reward` are written identically in `reward.json` in [0, 1],
higher is better. Both datasets must be complete.
There is no further nonlinear mapping. Random BEDROC depends on class
prevalence; it is not universally 0 or 0.5. ROC-AUC, BEDROC(80.5), EF at
0.5%, 1%, 2%, and 5%, per-target values, and benchmark means are retained.
The six overall_mean_* diagnostics remain equal-target means and are not
the primary dataset-balanced score. The verifier may also invoke `infer.sh` on
additional label-free staged views (such as pocket-swapped counterfactual
views) with a fresh empty `<output-root>` strictly for operator diagnostics in
`report.json` without changing `score` / `reward`. Reports include the scoring
protocol.
Exactly tied scores receive the expected metric over all orders within each
tied block; a constant score cannot exploit actives-first input order.

Malformed packages, inference errors/timeouts, incomplete coverage, duplicate
targets or indices, and nonfinite scores receive reward 0 with an error.
Missing/corrupt benchmark data or inability to establish isolation are
infrastructure failures and produce no reward.

Publish with /workspace/publish.py. It validates a staged copy and atomically
exchanges complete directories on Linux. The controller must snapshot a stable
published package before starting the separate verifier. The verifier does
not read the live trainer.

