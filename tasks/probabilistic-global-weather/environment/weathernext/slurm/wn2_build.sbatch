#!/usr/bin/env bash
#SBATCH --job-name=wn2-images
#SBATCH --partition=compsci
#SBATCH --account=users
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
TASK_ROOT=${WEATHER_TASK_ROOT:-${SLURM_SUBMIT_DIR}/probabilistic-global-weather/environment/weathernext}
WN2_IMAGES=${WEATHER_IMAGE_DIR:?Set a new image output directory}
mkdir -p "${WN2_IMAGES}"
python3 - "${TASK_ROOT}" "${WN2_IMAGES}/source" <<'PYCODE'
import hashlib,json,shutil,sys
from pathlib import Path
source,destination=map(Path,sys.argv[1:])
destination.mkdir(exist_ok=False)
for name in ('wn2','environment','tests','paper-comparison'):
    shutil.copytree(source/name,destination/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(destination/'slurm').mkdir()
for name in ('wn2_validation.py','wn2_distributed.py','render_apptainer.py'):
    shutil.copyfile(source/'slurm'/name,destination/'slurm'/name)
files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
(destination.parent/'source-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
PYCODE
WN2_SOURCE=${WN2_IMAGES}/source
WN2_BUILD_TMP=$(mktemp -d /tmp/wn2-images.XXXXXX)
trap 'rm -rf -- "${WN2_BUILD_TMP}"' EXIT
export APPTAINER_TMPDIR=${WN2_BUILD_TMP}
export APPTAINER_CACHEDIR=${WN2_IMAGES}/cache
for WN2_CONTEXT in environment tests; do
  python3 "${TASK_ROOT}/slurm/render_apptainer.py" \
    "${WN2_SOURCE}/${WN2_CONTEXT}/Dockerfile" "${WN2_IMAGES}/${WN2_CONTEXT}.def" --context "${WN2_SOURCE}"
  apptainer build --fakeroot --mksquashfs-args '-processors 4' \
    "${WN2_IMAGES}/${WN2_CONTEXT}.sif" "${WN2_IMAGES}/${WN2_CONTEXT}.def"
  apptainer exec --containall --cleanenv "${WN2_IMAGES}/${WN2_CONTEXT}.sif" \
    /opt/weather-python/bin/python -m pip freeze > "${WN2_IMAGES}/${WN2_CONTEXT}-model-requirements.txt"
  apptainer exec --containall --cleanenv "${WN2_IMAGES}/${WN2_CONTEXT}.sif" \
    /opt/wn2-eval/bin/python -m pip freeze > "${WN2_IMAGES}/${WN2_CONTEXT}-eval-requirements.txt"
  sha256sum "${WN2_IMAGES}/${WN2_CONTEXT}.sif" > "${WN2_IMAGES}/${WN2_CONTEXT}.sha256"
done
apptainer exec --containall --cleanenv --env PYTHONPATH=/opt/weatherbenchX --env OPENBLAS_NUM_THREADS=2 \
  "${WN2_IMAGES}/tests.sif" /opt/wn2-eval/bin/python -m pytest -q -p no:cacheprovider /tests/wn2
apptainer exec --containall --cleanenv --env PYTHONPATH=/opt/weatherbenchX --env OPENBLAS_NUM_THREADS=2 \
  "${WN2_IMAGES}/tests.sif" /opt/wn2-eval/bin/python -m pytest -q -p no:cacheprovider /tests/paper-comparison
printf 'WN2_HARBOR_DOCKERFILE_APPTAINER_BUILDS_AND_EVALUATOR_TESTS_OK\n'
