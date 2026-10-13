# B1 / B1g: cross-context mean delta

Per target, the mean of the training contexts' pseudobulk deltas (log1p CP10k), added to resampled
basal cells; unknown targets get no effect. `mode.json` selects the variant:

- `{"gated": false}`: B1;
- `{"gated": true}`: B1g, no effect if the target gene is not detected in the basal state, and
  deltas only on genes detected in at least 1 % of basal cells.

The package ships without `delta_table.npz`; `build_table.py` builds it (1.4 GB) next to the
script from the mounted release in under two minutes. Work in a copy under `/workspace/research`:

```bash
cp -r /workspace/baseline/meandelta /workspace/research/b1 && cd /workspace/research/b1
python build_table.py --data /workspace/data --contexts k562 jurkat hct116 hek293t
python /workspace/baseline/publish.py /workspace/research/b1 \
  --check-input /workspace/data/test/ctx_far/public
```

Leave-one-context-out: `python /workspace/evaluation/run_loco.py --package /workspace/research/b1
--outdir /workspace/research/loco/meandelta --train-cmd "python build_table.py --data {data}
--contexts {train_contexts}"`. Results: [RESULTS.md](RESULTS.md).
