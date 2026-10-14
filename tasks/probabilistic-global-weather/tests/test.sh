#!/usr/bin/env bash
set -euo pipefail
WN2_TESTS=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ -f "${WN2_TESTS}/wn2/test.sh" ]]; then
  exec bash "${WN2_TESTS}/wn2/test.sh"
fi
exec bash "${WN2_TESTS}/../environment/weathernext/wn2/test.sh"
