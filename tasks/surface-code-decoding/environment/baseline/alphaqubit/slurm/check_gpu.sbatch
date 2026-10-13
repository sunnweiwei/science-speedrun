#!/usr/bin/env bash
#SBATCH --job-name=qec-native-gpu
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"${QEC_TASK_ROOT}/.venv/bin/python" - <<'PY'
import json, os, time
from pathlib import Path
import numpy as np
import tensorflow as tf
assert tf.config.list_physical_devices('GPU'), 'GPU was allocated but TensorFlow cannot see it'
model = tf.keras.Sequential([tf.keras.layers.Input((25,24)),tf.keras.layers.LSTM(253,return_sequences=True),tf.keras.layers.LSTM(253),tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy')
x=np.random.default_rng(0).random((256,25,24),dtype=np.float32)
y=np.random.default_rng(1).integers(0,2,(256,1)).astype(np.float32)
model.train_on_batch(x,y)
t=time.perf_counter()
for _ in range(20): result=model.train_on_batch(x,y)
report={'tf':tf.__version__,'gpus':[str(x) for x in tf.config.list_physical_devices('GPU')],'seconds_per_step':(time.perf_counter()-t)/20,'loss':float(result),'job_id':os.environ.get('SLURM_JOB_ID'),'scope':'runtime-only toy two-layer LSTM check; not the reference model or scientific result'}
Path(os.environ['QEC_ASSET_ROOT'],'gpu-runtime-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
PY
