#!/usr/bin/env bash
# Sequential, resumable 5-family × 5-seed shared-model campaign.
set -euo pipefail
: "${TASK_RESEARCH_DIR:?Set TASK_RESEARCH_DIR}"
task=$(cd "$(dirname "$0")/../.." && pwd)
root="$TASK_RESEARCH_DIR/shared-full"
for family in eegnet contrawr cbramod biot codebrain; do
  for seed in 0 1 2 3 4; do
    if [[ -f "$root/$family-s$seed/score.exit" && "$(cat "$root/$family-s$seed/score.exit")" == 0 ]]; then
      python3 - "$task/tests/campaign" "$root" "$family" "$seed" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from summarize_shared import checked_run
root=Path(sys.argv[2]);campaign=json.loads((root/'campaign.json').read_text())
checked_run(root,campaign,sys.argv[3],int(sys.argv[4]))
PY
      echo "accepted $family seed $seed"
      continue
    fi
    echo "running $family seed $seed"
    bash "$task/tests/campaign/run_shared_full.sh" "$family" "$seed"
  done
done
python3 "$task/tests/campaign/summarize_shared.py" --root "$root" --require-complete
