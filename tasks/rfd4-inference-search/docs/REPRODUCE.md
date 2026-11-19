# Reproduce the public comparison

1. Run the materializer on the controller and build the development image. Mount the `development/assets` directory at `/workspace/data/assets` read-only and a fresh writable directory at `/workspace/research`.
2. Use a separate empty run directory per policy. Keep the GPU exclusive during each measurement. Run the command in the baseline README once with the default size and, for the Table S15 sample count, once with `--paper-size`.
   For independently scheduled length cohorts, use `--paper-size --length 100`, then the same command for 150 and 250 in fresh directories. The equal-weight mean of the three scores gives the full reward. Sum the allocated device-seconds to report throughput and include each AF3 startup in that sum.
3. The public seeds are 727100, 727150 and 727250 for the generator and 727000 for AF3. Run both policies on the same seeds. Different branching or batch shapes can change the mapping between slots and initial noise.
4. Inspect `monomer_run.yaml` and `generate.log` for each length. Native `per_example_metrics.csv` identifies the exact output files. The independent evaluator writes AF3 logs, Foldseek assignments and per-slot measurements under `evaluation/`.
5. To recompute a cohort's score, select rows with scRMSD ≤ 1.5 Å, count their distinct Foldseek representative IDs and divide by all allocated slots. Average the three cohort rates. Compare the ordinary and clustered pass rates to see where duplicates affect the result.

Run `python -m unittest discover -s tests -p test_reproduction.py` from the task directory to audit the bundled raw records using only the Python standard library.

The frozen native checkpoint is the public release used by this task. The authors' printed Table S15 numbers remain historical references. AF3 output-derived measurements are subject to the [AF3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).
