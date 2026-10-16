# Operator preparation for the agent workspace

Prepare data before starting the agent. Run these commands from the repository
root with network access and enough disk space. The agent consumes prepared,
read-only data and does not run downloads or runtime setup.

`sources.json` pins public revisions or object generations, sizes, SHA-256 hashes
and licenses. No credentials or Earth Engine account are required. The selected
two-shard recipe downloads about 148 GB before extraction; allow additional
space for a self-contained development copy, caches, runtime and checkpoints.

## Prepare the pinned source release

Install `uv` 0.8.22 and the system dependencies documented by the environment
Dockerfile. Replace `/chosen/...` with operator-owned locations.

```bash
set -euo pipefail
OLMO_RUNTIME=/chosen/runtime
OLMO_PREPARED=/chosen/prepared-release
TASK_DATA_DIR=/chosen/agent-data
TASK_PRIVATE_DIR=/chosen/private-evaluation
TASK_RESEARCH_DIR=/chosen/research

bash earth-observation-pretraining-transfer/environment/setup.sh "$OLMO_RUNTIME"
OLMO_PY="$OLMO_RUNTIME/olmoearth-upstream/.venv/bin/python"
CUDA_VISIBLE_DEVICES='' "$OLMO_PY" \
  earth-observation-pretraining-transfer/environment/data/materialize.py \
  --pretrain-source two-shards --output "$OLMO_PREPARED" --workers 4
```

The materializer combines 20,015 samples from public shards 0000 and 0001.
The declared geographic exclusions retain 19,837. Its `eval/` directory contains
the complete original AWF/LFMC release, including LFMC test labels. Keep this
source release outside the agent's mounts.

## Assemble the development view

Use a fresh destination. The pretraining alias and sample files can be symlinks
into the preparation tree; dereference them when making the development copy so
that mounting only `19837/` is sufficient. This copies data and needs additional
storage. It does not grant the agent access to the original archives.

```bash
test ! -e "$TASK_DATA_DIR"
test ! -e "$TASK_PRIVATE_DIR"
test ! -e "$TASK_RESEARCH_DIR"
mkdir -p "$TASK_DATA_DIR" "$TASK_PRIVATE_DIR" "$TASK_RESEARCH_DIR"
cp -aL "$OLMO_PREPARED/19837" "$TASK_DATA_DIR/19837"
CUDA_VISIBLE_DEVICES='' "$OLMO_PY" earth-observation-pretraining-transfer/environment/evaluation/screening.py prepare \
  --data-root "$OLMO_PREPARED" --output "$TASK_DATA_DIR/eval" \
  --site-fraction 1
cp -aL "$OLMO_PREPARED/eval/." "$TASK_PRIVATE_DIR/"
```

The development evaluation view has 1,115 AWF training / 344 validation rows and
31,175 LFMC training / 8,262 validation rows, with zero test rows. Its manifest
records membership and source hashes. Do not prepare this already-filtered view
again; its `index` fields retain the canonical source-row indices.

Use an empty research directory for each agent run. Supply these mounts through
the execution provider:

| Host data | Container destination | Access |
|---|---|---|
| `TASK_DATA_DIR/19837` | Agent: `/workspace/data/19837` | Read-only |
| `TASK_DATA_DIR/eval` | Agent: `/workspace/data/eval` | Read-only |
| `TASK_RESEARCH_DIR` | Agent: `/workspace/research` | Read/write |
| `TASK_PRIVATE_DIR` | Verifier only: `/data/eval` | Read-only |

Mount only the two development subdirectories, preserving the data guide and
source manifest copied into the image. Never mount the complete prepared release
or private evaluation directory into the agent environment. The development Compose file installs the two read-only mounts and the writable
research mount; the verifier Compose file mounts only `TASK_PRIVATE_DIR`. Export
these variables before invoking the controller. Offline policy and the default
48-hour development budget are declared in `task.toml`; the controller enforces
them. GPU allocation is also controller-owned. The controller snapshots the
submission for the separate verifier and collects its output.

Public LFMC test was exposed during earlier development. Runtime separation does
not establish a new blind holdout. Keep that limitation when reporting scores.

## Existing alternative preparation paths

The materializer also supports `subset_1k` and `shard0000`; these are not the data
configuration of the selected baseline. The existing `prepare_hf.py` can consume
a separately verified prepared release using `--repo`, `--revision` and
`--components`. The documented prepared-release revision is still a placeholder,
so the commands above use the pinned original public sources.

`prepare_hf.py` verifies sizes and hashes, resumes matching existing files, and
rejects changed files. Its native downstream components and historical training
launchers are not needed for this frozen-encoder baseline.


The standard controller entry point is `environment/data/materialize.sh`, which
forwards all arguments to `materialize.py` using `OLMO_PY` (or Python on PATH).
Run it after runtime setup, then assemble the development and private views above.
It is CPU-only and does not start training or evaluation.
