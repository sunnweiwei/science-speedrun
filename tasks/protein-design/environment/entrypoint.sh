#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d /data ]]; then
    echo 'FATAL: /data is not mounted. The operator must materialize the task data first;' >&2
    echo 'see /workspace/data/README.md.' >&2
    exit 2
fi
for split in fit tune; do
    if [[ ! -s "/data/${split}/manifest.json" ]]; then
        echo "FATAL: /data/${split}/manifest.json is missing or empty." >&2
        exit 2
    fi
done
# The scoring parameters are BAKED INTO THE IMAGE at /assets by
# environment/fetch_designability_weights.sh, so designability - the only metric this
# task reports - normally just works.
#
# This used to probe /eval-assets/af3, which was wrong twice over: AlphaFold 3 was
# removed from the task (its parameters cannot be redistributed, which is why the
# predictor is RoseTTAFold3), and nothing has ever created /eval-assets. The warning
# therefore fired on EVERY container start and told the agent to use a --geometry-only
# shortcut when full scoring was available. That shortcut no longer exists.
if [[ ! -d /assets/rf3 || ! -d /assets/ligandmpnn_params ]]; then
    echo 'WARNING: /assets is missing rf3/ or ligandmpnn_params/, so designability' >&2
    echo 'cannot be measured locally. Rebuild them with:' >&2
    echo '    bash /usr/local/bin/fetch-designability-weights /assets' >&2
    echo 'Until then score.py cannot run at all: designability is the only metric, and' >&2
    echo 'it needs these parameters. The verifier scores with its own copy regardless.' >&2
fi
exec "$@"
