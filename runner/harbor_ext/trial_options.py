"""Optional submission window shared by the continuing agents (Codex, Claude Code, Antigravity).

It is off unless the runner sets the environment variables. With it off, every agent runs exactly
the command it ran before this module existed.

HARBOR_GRACE_SECONDS (run_trial.sh: GRACE_SECONDS, with HARBOR_RESEARCH_SECONDS from HOURS)
    A separate, hard-limited, submission-only window after the research time is over:

      1. Research runs as before, but its end is enforced inside the container after
         HARBOR_RESEARCH_SECONDS, because the window must start in the same session.
      2. Every process of the agent user that the session started is stopped, so nothing started
         during research keeps running into the window.
      3. The package is left untouched for HARBOR_GRACE_SETTLE_SECONDS, so the snapshot collector
         keeps a copy of the package as it stood when research ended.
      4. The same session is continued once with SUBMIT_PROMPT, for at most HARBOR_GRACE_SECONDS.
      5. Everything is stopped again and /logs/agent/grace-period.json records the times and the
         package digest at the end of research and after the window (snapshot.sh's digest).

    The window is not research time. run.sh scores the package from the end of research as the
    result and, only if the window changed the package, scores the changed package separately.
    The agent is told not to experiment in the window, but nothing can enforce that from inside
    the container: review what it did (the trajectory after the "harbor.grace_start" line) before
    using a window result.
"""
import os
import shlex

SUBMIT_PROMPT = (
    "It is now time to submit. Stop exploring the problem. Submit your best solution without "
    "further experiments. You will be given an additional {minutes} starting now to submit.\n\n"
    "The research budget is over. This is a separate, hard-limited submission-only window.\n"
    "Use only code, weights, observations, and development results you already produced.\n"
    "Do not train, run inference/evaluations/simulations, search hyperparameters, or conduct any "
    "new experiment.\n"
    "Choose using only your existing research notes and results; no sealed evaluation results are "
    "available.\n"
    "Publish a complete portable package to /workspace/submission using the original task "
    "contract. File listing, copying, archiving, and static checks needed for packaging are "
    "allowed. Do not alter the scientific method.\n"
    "The processes left running by the research session have been stopped. Finish and stop after "
    "publishing."
)

# POSIX sh (Harbor runs agent commands with `sh -c`). @NAME@ fields are filled in by with_grace().
_PREAMBLE = r'''
__harbor_over=/tmp/.harbor-research-over.$$
rm -f "$__harbor_over"
__harbor_started() { sed -e 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20; }
__harbor_since=$(__harbor_started $$)
__harbor_stop_all() {
  # Stop every process of this user that started after this shell, except this shell, its
  # ancestors and the pids given. Processes that were already running stay: the container's
  # keep-alive process runs as the same user (root in most tasks), and stopping it ends the
  # container.
  __keep=" 1 $$ $* "; __p=$$
  while [ "${__p:-0}" -gt 1 ] 2>/dev/null; do
    __p=$(sed -e 's/^.*) //' "/proc/$__p/stat" 2>/dev/null | cut -d' ' -f2)
    __keep="$__keep${__p:-0} "
  done
  for __sig in TERM KILL; do
    for __d in /proc/[0-9]*; do
      __q=${__d#/proc/}
      case "$__keep" in *" $__q "*) continue ;; esac
      [ "$(stat -c %u "$__d" 2>/dev/null)" = "$(id -u)" ] || continue
      __s=$(__harbor_started "$__q"); [ -n "$__s" ] && [ "$__s" -ge "$__harbor_since" ] || continue
      kill -$__sig "$__q" 2>/dev/null
    done
    [ $__sig = TERM ] && sleep 5
  done
  return 0
}
__harbor_digest() {
  (cd /workspace/submission 2>/dev/null && find . -printf "%P %s %T@ %l\n" | sort | sha256sum) 2>/dev/null | cut -c1-16
}
( sleep @RESEARCH@; : > "$__harbor_over"; __harbor_stop_all "$(exec sh -c 'echo $PPID')" ) &
'''

_EPILOGUE = r'''
if [ -e "$__harbor_over" ]; then __harbor_ended=deadline; else __harbor_ended=loop; : > "$__harbor_over"; fi
__harbor_stop_all
__harbor_d0=$(__harbor_digest); __harbor_t0=$(date +%s)
echo "{\"type\":\"harbor.research_end\",\"ended_by\":\"$__harbor_ended\"}" >&2
sleep @SETTLE@
( sleep @GRACE@; __harbor_stop_all "$(exec sh -c 'echo $PPID')" ) &
__harbor_t1=$(date +%s)
echo "{\"type\":\"harbor.grace_start\",\"seconds\":@GRACE@}" >&2
@SUBMIT@
__harbor_stop_all
__harbor_t2=$(date +%s); __harbor_d1=$(__harbor_digest)
echo "{\"type\":\"harbor.grace_end\"}" >&2
mkdir -p /logs/agent 2>/dev/null
printf '{"research_seconds": %s, "grace_seconds": %s, "settle_seconds": %s, "research_ended_by": "%s", "research_end_epoch": %s, "grace_start_epoch": %s, "grace_end_epoch": %s, "package_digest_at_research_end": "%s", "package_digest_after_grace": "%s"}\n' \
  @RESEARCH@ @GRACE@ @SETTLE@ "$__harbor_ended" "$__harbor_t0" "$__harbor_t1" "$__harbor_t2" "$__harbor_d0" "$__harbor_d1" \
  > /logs/agent/grace-period.json 2>/dev/null
rm -f "$__harbor_over"
exit 0
'''


def _positive_int(name: str) -> int:
    try:
        return max(0, int(float(os.environ.get(name, "") or 0)))
    except ValueError:
        return 0


def grace() -> dict | None:
    """Window settings, or None when the window is off (the default)."""
    seconds = _positive_int("HARBOR_GRACE_SECONDS")
    research = _positive_int("HARBOR_RESEARCH_SECONDS")
    if not seconds or not research:
        return None
    return {"research": research, "grace": seconds, "settle": _positive_int("HARBOR_GRACE_SETTLE_SECONDS")}


def loop_guard() -> str:
    """Extra `while` condition for the continuation loop: stop continuing once research is over."""
    return '[ ! -e "$__harbor_over" ] && ' if grace() else ""


def submit_prompt() -> str:
    seconds = grace()["grace"]
    minutes = f"{seconds // 60} minutes" if seconds % 60 == 0 and seconds >= 120 else f"{seconds} seconds"
    return SUBMIT_PROMPT.format(minutes=minutes)


def with_grace(research_script: str, submit_command: str) -> str:
    """Wrap an agent's whole research script (first run plus continuation loop) with the window.

    `submit_command` is the agent's own continuation command with submit_prompt() as the prompt.
    Returns `research_script` unchanged when the window is off.
    """
    settings = grace()
    if settings is None:
        return research_script

    def fill(text: str) -> str:
        return (text.replace("@RESEARCH@", str(settings["research"])).replace("@GRACE@", str(settings["grace"]))
                .replace("@SETTLE@", str(settings["settle"])))

    return fill(_PREAMBLE) + research_script.rstrip("\n") + "\n" + fill(_EPILOGUE).replace("@SUBMIT@", submit_command)


def quoted_submit_prompt() -> str:
    return shlex.quote(submit_prompt())
