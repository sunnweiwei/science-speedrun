"""Harbor's Claude Code agent, resumed whenever it ends its turn.

`claude --print` returns as soon as the model answers without a tool call, but the tasks are
continuing research with no end point. This subclass appends a loop to the run command: each time
Claude Code ends, the same session is continued (`--continue`) with CONTINUE_PROMPT. The prompt says
nothing about time; Harbor's agent timeout still ends the run. Three consecutive continuations that
end within MIN_TURN_SECONDS stop the loop (an auth, quota or CLI error would otherwise spin).

    -a harbor_ext.claude_continue:ContinuingClaudeCode   (same kwargs as `-a claude-code`)

Auth comes from the runner's environment: CLAUDE_CODE_OAUTH_TOKEN (a subscription login, from
`claude setup-token`) or ANTHROPIC_API_KEY.
"""
import shlex

from harbor.agents.installed.claude_code import ClaudeCode

CONTINUE_PROMPT = (
    "Continue the research. There is no end point: pick up from your research notes and the latest "
    "results, run the next experiment, and keep the submission up to date with the strongest "
    "replayable candidate."
)
MIN_TURN_SECONDS = 120
MAX_QUICK_EXITS = 3
_MARKER = "claude --verbose --output-format=stream-json "


class ContinuingClaudeCode(ClaudeCode):
    @staticmethod
    def name() -> str:
        return "claude-code"

    def _with_resume_loop(self, command: str) -> str:
        if _MARKER not in command or "--continue " in command or "--resume " in command:
            return command
        claude = command[command.index(_MARKER):]
        if " --print " not in claude or "| tee " not in claude:
            return command
        resume = (claude.replace(" --print ", " --continue --print ", 1)
                  .replace("| tee ", "| tee -a ", 1))
        return (
            f"{command}\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            "quick=0; turn=1\n"
            f"while [ $quick -lt {MAX_QUICK_EXITS} ]; do\n"
            "  turn=$((turn+1)); start=$(date +%s)\n"
            '  echo "{\\"type\\":\\"harbor.resume\\",\\"turn\\":$turn}" >&2\n'
            f"  printf '%s' {shlex.quote(CONTINUE_PROMPT)} | {resume}\n"
            f"  if [ $(( $(date +%s) - start )) -lt {MIN_TURN_SECONDS} ]; then quick=$((quick+1)); else quick=0; fi\n"
            "done\n"
        )

    async def exec_as_agent(self, environment, command, *args, **kwargs):
        return await super().exec_as_agent(environment, self._with_resume_loop(command), *args, **kwargs)
