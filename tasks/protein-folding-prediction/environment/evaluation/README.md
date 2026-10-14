# Self-service development evaluation

You have the complete benchmark at `/workspace/data/foldbench/suite`, native mmCIFs at
`/workspace/data/foldbench/ground_truths`, and official scoring source at `/opt/foldbench`.
Its installed Python is `/opt/foldbench-env/bin/python`. Read, copy, run and
analyze this code directly. There is **no required evaluation service or wrapper**.

Each suite request is a coordinate-free JSON referenced by `suite_manifest.json`.
Create an input directory with a `requests.jsonl` containing any desired subset,
plus `features.esmc6b_final_int8.jsonl` and its referenced input feature files as
specified in `contract.md`, then invoke your model. Suite feature sources are
indexed by `/workspace/data/foldbench/features.esmc6b_final_int8.jsonl`; select rows by the
SHA-256 of each protein sequence and stage their files under the input directory.
The optional delivery adapter uses this same contract for development and
external inference. Do not load ESMC-6B to regenerate them.

For a quick contract replay, stage the first request or explicit request IDs:

```bash
python /workspace/evaluation/stage_inputs.py --output /workspace/research/my-input
python /workspace/evaluation/stage_inputs.py --output /workspace/research/my-input \
  --request-id 5sbj-assembly1
bash /workspace/submission/infer.sh /workspace/research/my-input /workspace/research/my-output
```

FoldBench expects a `prediction_reference.csv` under
`EVALUATION_DIRECTORY/ALGORITHM_NAME/`. It uses the same five columns as the
delivery CSV. Before running the native scorer, resolve `prediction_path` entries
to absolute paths in the current container (delivery paths themselves are
relative to the inference output directory).

For example, after producing that reference:

```bash
/opt/foldbench-env/bin/python /opt/foldbench/evaluate.py \
  --targets_dir /workspace/data/foldbench/suite/targets \
  --ground_truth_dir /workspace/data/foldbench/ground_truths \
  --evaluation_dir /workspace/research/my-evaluation \
  --algorithm_name candidate \
  --targets monomer_protein

/opt/foldbench-env/bin/python /opt/foldbench/task_score_summary.py \
  --target_dir /workspace/data/foldbench/suite/targets \
  --evaluation_dir /workspace/research/my-evaluation \
  --algorithm_names candidate --targets monomer_protein \
  --metric_type rank --output_path /workspace/research/my-evaluation/summary.csv
```

Run the native scripts from `/opt/foldbench`; prepend
`/opt/foldbench-env/bin` to PATH if invoking their subprocess tools. Choose your
own subsets, schedule, analyses and visualization. Use target tables filtered to
the chosen subset for cheap diagnostics; these are not full-benchmark scores.

The external verifier rejoins raw scores to **all original target rows** and
counts missing/nonfinite values as failures, whereas an upstream summary may
drop missing values. It reports nine scores on a 0–1 scale:

- Protein/RNA/DNA monomers: mean lDDT, missing values zero.
- Five protein/protein-or-nucleic-acid interface tracks: fraction with DockQ >= 0.23 and CAPRI
  acceptable geometry (ligand RMSD <= 10 Å or interface RMSD <= 4 Å). The upstream summary uses
  DockQ alone; `python3 /workspace/evaluation/track_scores.py SCORING_TARGETS_DIR RAW_DIR` applies
  the verifier's rules to the raw scorer CSVs.
- Protein-ligand: fraction with RMSD < 2 Å and lDDT-PLI > 0.8.

It retains full denominators, selects by model ranking rather than native score,
and does not introduce a new cross-track aggregate. Keep coverage and failure
reports alongside scores. These are benchmark-visible development results;
independent replay does not turn them into a blind holdout evaluation.
