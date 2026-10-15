# CrystalGen starter

Two verified generators that publish a contract-conforming package. Reproduced scores are in
[RESULTS.md](RESULTS.md).

- **`omg_port/`, the research baseline.** Pinned OMatG's unconditional de-novo model for species,
  fractional positions and cells, trained from random initialization on MP-20 train.
- **`train.py` / `generate.py`, the smoke generator.** Aggregate element, atom-count and density
  statistics with no learning. It exists so contract changes can be tested in seconds.

## Unconditional OMatG path

[`omg_port/config.yaml`](omg_port/config.yaml) selects upstream `Model`, `CSPNetFull` with
`prop: false`, and `StructureDataset` with no property keys. There are no condition adapters,
property targets, condition dropout or request-driven geometry projections.
[`omg_port/runtime.py`](omg_port/runtime.py) supplies only a CLI compatibility shim: it accepts
OMGCLI's linked batch-size argument and delegates sampling to the unchanged upstream
`IndependentSampler`. Trainer precision is linked to all datasets by OMGCLI.

The starting Linear-SDE-Gamma recipe uses 2,000 epochs and eight GPUs with batch size 64 each (an
effective global batch of 512) and 12,403,912 parameters. It is an editable baseline, not a claim
to reproduce the published OMatG result. The fit helper copies the read-only CSVs into the run
directory, because OMatG creates a local LMDB cache beside its input:

```bash
bash /workspace/baseline/starter/omg_port/fit.sh \
  /workspace/data/mp20/train.csv \
  /workspace/data/mp20/val.csv \
  /workspace/research/omatg-run
```

Extra arguments are passed to `omg fit`, for example `--trainer.devices=6 --data.batch_size=85`
or, for a one-step check, `--trainer.max_epochs=1 --trainer.limit_train_batches=1
--trainer.limit_val_batches=1`.

Publish a selected checkpoint as the single-safetensors offline package:

```bash
python /workspace/baseline/starter/omg_port/publish.py \
  --checkpoint /workspace/research/omatg-run/lightning_logs/version_0/checkpoints/best_val_loss_total.ckpt \
  --sampler-stats /workspace/research/omatg-run/sampler_stats.json \
  --output /workspace/submission
```

`publish.py` assembles the package next to the output and swaps it in with one rename, so
`/workspace/submission` is never half-written. The published entry point shards inference over
every visible GPU, reconstructs request-only OMatG seed records from the learned MP-20 atom-count
prior, and writes the exact manifest/CIF contract. It needs neither MP-20 nor scorer assets at
inference. `config.yaml` is a starting recipe, not a frozen research choice.

Published OMatG checkpoints are forbidden. This path instantiates the architecture at random and
trains only on the mounted MP-20 train split. Historical v0 checkpoints include conditional
adapters and do not load under this starter's strict state-dict loading; start a fresh run.

## Smoke path

```bash
bash /workspace/baseline/starter/train.sh \
  /workspace/data/mp20/train.csv \
  /workspace/research/smoke/submission

bash /workspace/research/smoke/submission/generate.sh \
  /workspace/baseline/starter/examples/input \
  /workspace/research/smoke/output
```

`train.py` publishes a complete JSON-checkpoint package to the chosen directory, with the same
stage-and-swap publication. It stores aggregate element, atom-count and density statistics, not
CIFs, material IDs or a structure index. It is intentionally noncompetitive.

Score either output with `/workspace/evaluation/score.sh` (see `/workspace/evaluation/README.md`).
