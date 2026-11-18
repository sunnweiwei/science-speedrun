#!/usr/bin/env bash
# Score snapshots of a finished run to get reward over time.
#   score_snapshots.sh NAME [K]
# Picks K snapshots (default 6) evenly spaced in time, plus the last one, scores each with
# run_verify.sh (serialized, the task's verifier resources, on VERIFY_GPU_IDS as in run.sh) and writes
# $EVAL_ROOT/results/NAME/curve.tsv: hours since start, then one column per reward key.
# Versions published in a submission window (GRACE_SECONDS) are not research and are left out.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$here/config.sh"
name=$1; k=${2:-6}; out="$EVAL_ROOT/results/$name"; snaps="$out/snapshots"
task=$(cat "$out/task"); [ -f "$out/tasks_dir" ] && export TASKS_DIR=$(cat "$out/tasks_dir")
mapfile -t picks < <(python3 - "$snaps/index.tsv" "$k" "$out/grace-period.json" <<'PY'
import calendar, csv, json, os, sys, time
window = json.load(open(sys.argv[3]))["grace_start_epoch"] if os.path.exists(sys.argv[3]) else None
def research(row):  # published before the submission window started
    return window is None or calendar.timegm(time.strptime(row["utc"], "%Y-%m-%dT%H:%M:%SZ")) <= window
rows = [int(r["seconds"]) for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if research(r)]
k = int(sys.argv[2])
if not rows: sys.exit()
if len(rows) <= k + 1: chosen = rows
else:
    chosen = sorted({rows[round(i * (len(rows) - 1) / k)] for i in range(k + 1)})
print("\n".join(map(str, chosen)))
PY
)
for s in "${picks[@]}"; do
  [ -f "$out/snapshot-scores/$s.json" ] && continue
  echo "[curve] scoring snapshot at $(python3 -c "print(round($s/3600, 2))") h"
  GPU_IDS="${VERIFY_GPU_IDS-${GPU_IDS:-}}" "$here/run_verify.sh" "verify-$name-s$s" "$task" "$snaps/$s" > "$out/verify-s$s.log" 2>&1 || true
  mkdir -p "$out/snapshot-scores"
  $DOCKER run --rm -v "$EVAL_ROOT:$EVAL_ROOT" --entrypoint bash "$RUNNER_IMAGE" -c \
    "cp '$EVAL_ROOT/trials-verify/verify-$name-s$s/verifier/reward.json' '$out/snapshot-scores/$s.json' 2>/dev/null; chown -R $(id -u):$(id -g) '$out/snapshot-scores'" || true
done
python3 - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); rows = []
for f in sorted((out / "snapshot-scores").glob("*.json"), key=lambda p: int(p.stem)):
    rows.append((int(f.stem) / 3600, json.loads(f.read_text())))
keys = sorted({k for _, r in rows for k in r})
lines = ["\t".join(["hours"] + keys)] + ["\t".join([f"{h:.2f}"] + [str(r.get(k, "")) for k in keys]) for h, r in rows]
(out / "curve.tsv").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
PY
