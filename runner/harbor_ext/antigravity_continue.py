"""Harbor's Antigravity CLI (agy) agent with a mounted binary, resumed whenever it ends its turn.

`agy --prompt` returns as soon as the model answers without a tool call, but the tasks are
continuing research with no end point. This subclass appends a loop to the run command: each time
agy ends, the most recent conversation is continued (`--continue`) with CONTINUE_PROMPT. The prompt
says nothing about time; Harbor's agent timeout still ends the run. Three consecutive continuations
that end within MIN_TURN_SECONDS stop the loop (an auth, quota or CLI error would otherwise spin).
A quick exit is often a per-minute quota error (429) on a shared Gemini project, so the loop waits
QUICK_EXIT_WAIT_SECONDS x the number of consecutive quick exits before continuing.

Harbor's installer downloads agy inside the task container, which has no network. This subclass
instead uses a host agy binary mounted read-only at /usr/local/bin/agy (the runner's AGY_BIN).

    -a harbor_ext.antigravity_continue:ContinuingAntigravity -m google/<gemini model>

Auth comes from the runner's environment: GEMINI_API_KEY (or GOOGLE_API_KEY).
"""
import re
import shlex

from harbor.agents.installed.antigravity_cli import AntigravityCli

CONTINUE_PROMPT = (
    "Continue the research. There is no end point: pick up from your research notes and the latest "
    "results, run the next experiment, and keep the submission up to date with the strongest "
    "replayable candidate."
)
MIN_TURN_SECONDS = 120
MAX_QUICK_EXITS = 3
QUICK_EXIT_WAIT_SECONDS = 90
_RUN = re.compile(r"(\$HOME/\.local/bin/agy --dangerously-skip-permissions .*?)--prompt=.*? 2>/logs/agent/", re.S)


class ContinuingAntigravity(AntigravityCli):
    @staticmethod
    def name() -> str:
        return "antigravity-cli"

    async def install(self, environment) -> None:
        await self.exec_as_agent(
            environment,
            command='mkdir -p "$HOME/.local/bin" && ln -sf /usr/local/bin/agy "$HOME/.local/bin/agy" '
                    '&& "$HOME/.local/bin/agy" --version',
        )

    def _with_resume_loop(self, command: str) -> str:
        match = _RUN.search(command)
        if not match or "--continue" in command or not command.rstrip().endswith("exit $rc"):
            return command
        flags = match.group(1)
        err, stream = self._STDERR_LOG_FILENAME, self._STREAM_LOG_FILENAME
        resume = (f"{flags}--continue --prompt={shlex.quote(CONTINUE_PROMPT)} "
                  f"2>>/logs/agent/{err} </dev/null | stdbuf -oL tee -a /logs/agent/{stream}")
        first = command.rstrip()[: -len("exit $rc")]
        return (
            f"{first}\n"
            "quick=0; turn=1\n"
            f"while [ $quick -lt {MAX_QUICK_EXITS} ]; do\n"
            "  turn=$((turn+1)); start=$(date +%s)\n"
            '  echo "{\\"type\\":\\"harbor.resume\\",\\"turn\\":$turn}" >&2\n'
            f"  {resume}\n"
            f"  if [ $(( $(date +%s) - start )) -lt {MIN_TURN_SECONDS} ]; then quick=$((quick+1)); "
            f"sleep $((quick * {QUICK_EXIT_WAIT_SECONDS})); else quick=0; fi\n"
            "done\n"
            "exit 0\n"
        )

    async def exec_as_agent(self, environment, command, *args, **kwargs):
        return await super().exec_as_agent(environment, self._with_resume_loop(command), *args, **kwargs)
