#!/bin/bash
set -euo pipefail
exec python "$(dirname "$0")/infer.py" "$1" "$2"
