#!/bin/bash
#SBATCH -p gpu-train
#SBATCH --nodes 1
#SBATCH --gres=gpu:large:8
#SBATCH -C L40|L40S
#SBATCH --mem=512g
#SBATCH -t 2-00:00:00
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -J rfd3-small-ppi
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
#SBATCH --no-kill=off

# The exact recipe that produced the checkpoint reported in
# `../evaluation/RESULTS.md` - job 25880709, 8x L40S, 48 h wall, cold start, ending on
# walltime at ~epoch 108 (~264k examples). `epoch-0100.ckpt` is the one that was scored.
#
# There is ONE experiment config and both sides run it. The agent runs
# `experiment=pretrain_small_ppi` as-is: it defaults to the container's data (/data) and
# a single GPU. This script is the same config with the CLUSTER's paths and 8 ranks
# passed on the command line. Scale and file locations differ; the recipe does not.
#
# WHAT THE EXPERIMENT CHANGES: exactly one leaf, `ppi.frequency: 0.0 -> 5.0`
# (`configs/experiment/pretrain_small_ppi.yaml`, which defaults to `pretrain_small`,
# which defaults to upstream `pretrain` and cuts layer widths and nothing else).
# Verified by composing both experiments and diffing the resolved tree.
#
# NO `ckpt_path`: a cold start, deliberately, so every condition including PPI is
# learned together from random init. Resuming an existing checkpoint with a new
# condition switched on would confound "PPI was added" with "N more epochs happened".
# The cost is real: at ~1.50 examples/s two days reaches only ~half the examples a
# 4-day run had seen, so the non-ppi tracks score BELOW the earlier benchmark. That was
# the accepted trade, and RESULTS.md reports it.
#
# `grad_accum_steps=1` is a known deviation from upstream's 3 and is kept here only
# because changing it would have confounded this run's one intended change with a
# change of effective batch size.
#
# ONE RANK PER GPU, LAUNCHED BY srun - not optional, and not what
# `trainer.devices_per_node=8` does by itself. Lightning's SLURMEnvironment reports
# `creates_processes_externally = True` and takes world size from SLURM_NTASKS, so
# plain `python` trains on 1 of 8 GPUs and says nothing about it.

# --- site paths. Override these; everything below is portable. ---------------------
FOUNDRY=${FOUNDRY:?directory holding models/rfd3, i.e. the vendored rfd3 tree}
VENV_PY=${VENV_PY:-python}          # interpreter with rc-foundry's deps importable
WORK=${WORK:?scratch dir for logs, checkpoints and the parsed-CIF cache}
PQ=${PQ:?directory holding the two metadata parquets}
# The experiment config defaults `paths/data` to the agent's container (/data/fit/...),
# so the cluster's structure mirror has to be named here or atomworks reads nothing.
PDB_MIRROR=${PDB_MIRROR:-/projects/ml/frozen_pdb_copies/2025_07_13_pdb}

cd "$FOUNDRY"

# Dataloader workers bind an AF_UNIX socket under TMPDIR and the path has a 108-char
# limit; a deep scratch path breaks inference silently rather than erroring.
export TMPDIR=${TMPDIR:-/tmp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
NWORKERS=$(( NCPU - 1 ))
echo "=== job ${SLURM_JOB_ID:-local} on $(hostname): ${SLURM_NTASKS:-1} ranks, ${NCPU} cpus/rank, ${NWORKERS} loader workers/rank ==="
echo "=== COLD START: no ckpt_path, training from random init ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$VENV_PY" models/rfd3/src/rfd3/train.py \
  experiment=pretrain_small_ppi \
  trainer.devices_per_node=8 \
  trainer.grad_accum_steps=1 \
  paths.log_dir="$WORK/logs" \
  paths.data.pdb_data_dir="$PDB_MIRROR" \
  paths.data.pdb_parquet_dir="$PQ" \
  paths.data.cif_cache_dir="$WORK/cifcache" \
  dataloader.train.dataloader_params.num_workers=$NWORKERS \
  dataloader.val.dataloader_params.num_workers=$NWORKERS

echo "=== finished $(date) ==="

# Two overrides above are not width or condition changes and are worth naming:
#
#   paths.data.pdb_data_dir / pdb_parquet_dir
#     The experiment config points at the container's /data by default, which is what
#     the agent needs. These two put it back on the cluster's mirror and split dir.
#     Both table filenames (`pn_units_df_train.parquet`, `interfaces_df_train.parquet`)
#     are set in the config and match the IPD layout, so nothing else is needed here.
#     NOTE the public atomworks archive (TRAINING_DATA.md §0) ships `interfaces_df.parquet`
#     WITHOUT the `_train` suffix - symlink it or override that leaf if you use it.
#
#   calculate_hbonds
#     Needed HBPLUS, which is installed nowhere; upstream's 0.2 raises without it. It is
#     now 0 in the config itself, so this script no longer overrides it.
