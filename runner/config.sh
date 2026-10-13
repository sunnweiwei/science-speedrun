# Shared settings for the runner scripts. Override any of them in the environment.
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
: "${EVAL_ROOT:?Set EVAL_ROOT to a large, persistent, writable directory (trials, results, data env files)}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/tasks}"
RUNNER_IMAGE="${RUNNER_IMAGE:-harbor-runner:0.23.0-bx}"
DOCKER="${DOCKER:-sudo -n docker}"
ENVS_DIR="${ENVS_DIR:-$EVAL_ROOT/envs}"                 # <task>.env written by prepare.sh
# Agent harness: codex (default) or claude-code. Both resume the same thread whenever the model
# ends its turn, with a prompt that says nothing about time.
AGENT_KIND="${AGENT_KIND:-codex}"
# Codex: the standalone x86_64-unknown-linux-musl package (bin/codex, bin/codex-code-mode-host,
# codex-path/rg) and an auth.json from a separate login (mode 644 inside a 700 directory).
CODEX_PKG="${CODEX_PKG:-$EVAL_ROOT/codex-pkg}"
CODEX_AUTH_JSON="${CODEX_AUTH_JSON:-$EVAL_ROOT/.authdir/auth.json}"
CODEX_MODEL="${CODEX_MODEL:-openai/gpt-6.1-sol}"
CODEX_REASONING="${CODEX_REASONING:-medium}"
CODEX_VERSION="${CODEX_VERSION:-0.160.1}"
# Claude Code: a host binary mounted read-only; auth from CLAUDE_CODE_OAUTH_TOKEN (login, default,
# made by `claude setup-token`) or ANTHROPIC_API_KEY, read from the caller's environment.
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude 2>/dev/null || true)}"
CLAUDE_MODEL="${CLAUDE_MODEL:-anthropic/claude-opus-5-5}"
# Host directories the runner container must see at the same path (space separated), for example
# the data roots in the env files. EVAL_ROOT, the tasks and this directory are always mounted.
MOUNT_ROOTS="${MOUNT_ROOTS:-}"

runner_mounts() {
  local d
  echo -n "-v /var/run/docker.sock:/var/run/docker.sock"
  for d in "$HARNESS_DIR" "$EVAL_ROOT" "$TASKS_DIR" $MOUNT_ROOTS; do echo -n " -v $d:$d"; done
}

agent_args() {
  case "$AGENT_KIND" in
    codex)
      printf '%s\n' -a harbor_ext.codex_continue:ContinuingCodex -m "$CODEX_MODEL" \
        --agent-kwarg "reasoning_effort=$CODEX_REASONING" --agent-kwarg web_search=disabled \
        --agent-kwarg "version=$CODEX_VERSION" \
        --mounts "[{\"type\":\"bind\",\"source\":\"$CODEX_PKG/bin/codex\",\"target\":\"/usr/local/bin/codex\",\"read_only\":true},{\"type\":\"bind\",\"source\":\"$CODEX_PKG/bin/codex-code-mode-host\",\"target\":\"/usr/local/bin/codex-code-mode-host\",\"read_only\":true},{\"type\":\"bind\",\"source\":\"$CODEX_PKG/codex-path/rg\",\"target\":\"/usr/local/bin/rg\",\"read_only\":true}]" \
        --allow-agent-host chatgpt.com --allow-agent-host auth.openai.com --allow-agent-host api.openai.com ;;
    claude-code)
      printf '%s\n' -a harbor_ext.claude_continue:ContinuingClaudeCode -m "$CLAUDE_MODEL" \
        --mounts "[{\"type\":\"bind\",\"source\":\"$CLAUDE_BIN\",\"target\":\"/usr/local/bin/claude\",\"read_only\":true}]" \
        --allow-agent-host api.anthropic.com ;;
    *) echo "AGENT_KIND must be codex or claude-code" >&2; return 1 ;;
  esac
}

agent_env() {
  case "$AGENT_KIND" in
    codex) printf '%s\n' -e "CODEX_AUTH_JSON_PATH=$CODEX_AUTH_JSON" ;;
    claude-code)
      [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ] || {
        echo "Set CLAUDE_CODE_OAUTH_TOKEN (claude setup-token) or ANTHROPIC_API_KEY" >&2; return 1; }
      [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && printf '%s\n' -e CLAUDE_CODE_OAUTH_TOKEN
      [ -n "${ANTHROPIC_API_KEY:-}" ] && printf '%s\n' -e ANTHROPIC_API_KEY
      return 0 ;;
  esac
}
