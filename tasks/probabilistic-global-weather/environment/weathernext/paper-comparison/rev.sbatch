#!/bin/bash
#SBATCH --job-name=paper-rev-climatology
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
root=${PAPER_ROOT:?}
export OPENBLAS_NUM_THREADS=2
exec /usr/project/xtmp/sc829/weather-track/wn2-pilot/eval-py312/bin/python \
  "$root/rev-source/paper-comparison/prepare_rev.py" --output "$root/rev-climatology" --workers 4
