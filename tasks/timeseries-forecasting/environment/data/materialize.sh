#!/usr/bin/env bash
# Documented entry point for data preparation; see README.md in this directory.
#   bash materialize.sh /large-disk/timeseries                 # gifteval + the 81 allow-listed corpus directories (43 GB)
#   bash materialize.sh /large-disk/timeseries --tier fast     # gifteval + the 1.25 GB fast tier
#   bash materialize.sh /large-disk/timeseries --verify-only   # re-hash what is already there
set -euo pipefail
python3 - <<'EOF' || { echo "materialize.sh needs the huggingface_hub package: pip install huggingface-hub" >&2; exit 1; }
import huggingface_hub  # noqa: F401
EOF
exec python3 "$(dirname "$(readlink -f "$0")")/materialize.py" "$@"
