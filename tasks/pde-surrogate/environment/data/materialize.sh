#!/usr/bin/env bash
# Convenience wrapper around materialize.py: DEST is $1 (or $TASK_PRIVATE_DIR); every other
# argument is forwarded unchanged, so the documented flags work here too.
#   bash materialize.sh /large-disk/pde-surrogate                       # what the verifier needs: held-out files only (67 GB)
#   bash materialize.sh /large-disk/pde-surrogate --plan                # print the plan, download nothing
#   bash materialize.sh /large-disk/pde-surrogate --verify-only         # re-hash what is present
#   bash materialize.sh /large-disk/pde-surrogate --all                 # the four sources in full (185 GB, includes files the split never uses)
#   bash materialize.sh /large-disk/pde-surrogate --from-release --subtree train   # an agent host: the published train/ subtree
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${TASK_PRIVATE_DIR:?set TASK_PRIVATE_DIR or pass a destination}}"
shift || true
python3 -c "import huggingface_hub, h5py" 2>/dev/null || {
  echo "materialize.py needs the huggingface_hub and h5py packages (pip install huggingface-hub==0.36.2 h5py==3.16.0)" >&2
  exit 2
}
if [ "${1:-}" = "--all" ]; then
  shift
  exec python3 "$HERE/materialize.py" "$DEST" --sources ape2d-full,well-turbulent_radiative_layer_2D,well-active_matter,pdearena-ns "$@"
fi
case " $* " in
  *" --from-release "*|*" --sources "*|*" --pretrained "*)
    exec python3 "$HERE/materialize.py" "$DEST" "$@" ;;
  *)
    exec python3 "$HERE/materialize.py" "$DEST" --manifest "$HERE/split.json" --sources - "$@" ;;
esac
