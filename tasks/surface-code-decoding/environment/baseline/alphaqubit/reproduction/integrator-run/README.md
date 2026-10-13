# Integrator run: AlphaQubit recipe from scratch on H100s

The run behind the integrator row of `../../RESULTS.md` (P 0.75569). The baseline has already been
trained and scored; these logs are its record, so there is no need to train it again to see what it
does. The task author's own run is documented in `/workspace/data/reference/results`.

One gzipped log per training group (`d3_*` and `d5_*`, the geometry, basis and centre), each with
pretraining and fine-tuning:
- `validation` records: complete-pass shots and, per condition (rounds), errors, error rate and loss;
- `training_result` records: stage, status and timestamps.

Read them with `zcat <group>.log.gz | grep '^{'`.
