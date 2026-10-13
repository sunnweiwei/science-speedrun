"""Harbor's Codex agent, resumed whenever it ends its turn.

The tasks are continuing research with no end point, but `codex exec` ends as soon as the model
answers without a tool call (for example "research remains active ..."), and Harbor's Codex agent
runs it once. This subclass appends a loop to that one command: each time Codex ends, the same
session is resumed with CONTINUE_PROMPT. The prompt says nothing about time; Harbor's agent
timeout still ends the run. Three consecutive resumes that end within MIN_TURN_SECONDS stop the
loop (an auth, quota or CLI error would otherwise spin).

    -a harbor_ext.codex_continue:ContinuingCodex   (same kwargs as `-a codex`)
"""
import shlex

from harbor.agents.installed.codex import Codex

CONTINUE_PROMPT = (
    "Continue the research. There is no end point: pick up from your research notes and the latest "
    "results, run the next experiment, and keep the submission up to date with the strongest "
    "replayable candidate."
)
MIN_TURN_SECONDS = 120
MAX_QUICK_EXITS = 3


class ContinuingCodex(Codex):
    @staticmethod
    def name() -> str:
        return "codex"

    async def run(self, instruction, environment, context):
        self._first_instruction = shlex.quote(instruction)
        return await super().run(instruction, environment, context)

    def _with_resume_loop(self, command: str) -> str:
        quoted = getattr(self, "_first_instruction", None)
        if not quoted or quoted not in command or "codex exec resume" in command:
            return command
        resume = (command.replace("codex exec ", "codex exec resume --last ", 1)
                  .replace(quoted, shlex.quote(CONTINUE_PROMPT), 1)
                  .replace("| tee ", "| tee -a ", 1))
        return (
            f"{command}\n"
            "quick=0; turn=1\n"
            f"while [ $quick -lt {MAX_QUICK_EXITS} ]; do\n"
            "  turn=$((turn+1)); start=$(date +%s)\n"
            '  echo "{\\"type\\":\\"harbor.resume\\",\\"turn\\":$turn}" >&2\n'
            f"  {resume}\n"
            f"  if [ $(( $(date +%s) - start )) -lt {MIN_TURN_SECONDS} ]; then quick=$((quick+1)); else quick=0; fi\n"
            "done\n"
        )

    async def exec_as_agent(self, environment, command, *args, **kwargs):
        return await super().exec_as_agent(environment, self._with_resume_loop(command), *args, **kwargs)
