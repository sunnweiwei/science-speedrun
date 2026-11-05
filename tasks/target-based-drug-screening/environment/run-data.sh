#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
user_home=$(getent passwd "$(id -u)" | cut -d: -f6)
original_mode=$(stat -c '%a' "$user_home")

restore_home_mode() {
  chmod "$original_mode" "$user_home"
}
trap restore_home_mode EXIT INT TERM

# The Docker daemon is root-squashed to nobody on this NFS host. It needs
# search permission only long enough to establish the two bind mounts.
chmod o+x "$user_home"

exec_status=0
docker compose --project-directory "$script_dir" \
  -f "$script_dir/data-compose.yaml" run --rm data "$@" || exec_status=$?
exit "$exec_status"

