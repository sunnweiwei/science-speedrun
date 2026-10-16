"""Bounded, resident action-only policy protocol. No CFD or ML dependencies."""

import json
import math
import os
from pathlib import Path
import pwd
import select
import shutil
import signal
import subprocess
import tempfile
import time


PROTOCOL = "hydrogym-policy/v1"
MAX_LINE = 64 * 1024 * 1024
# When HYDROGYM_POLICY_ISOLATION=1 (exported by the Harbor verifier, tests/test.sh),
# the submitted policy runs as this unprivileged account, so it can neither rewrite
# results nor outlive its case: every process of the account is killed when the policy
# closes. Cases replayed side by side therefore each get their own account
# (HYDROGYM_POLICY_UID, set by evaluate_native.replay_cases). A missing account fails closed.
POLICY_UID = int(os.environ.get("HYDROGYM_POLICY_UID", "10001"))
ISOLATION_ENV = "HYDROGYM_POLICY_ISOLATION"


class PolicyError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


def numbers(values):
    return isinstance(values, list) and all(
        type(v) in (int, float) and math.isfinite(v) for v in values)


def unpack(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [number for part in value for number in unpack(part)]
    return [float(value)]


def policy_account():
    """Unprivileged account for the policy process, or None when isolation is off."""
    if os.environ.get(ISOLATION_ENV) != "1":
        return None
    if os.geteuid() != 0:
        raise InfrastructureError("Policy isolation requires a root-run evaluator")
    try:
        account = pwd.getpwuid(POLICY_UID)
    except KeyError as error:
        raise InfrastructureError(f"No unprivileged policy account (uid {POLICY_UID})") from error
    if account.pw_uid == 0 or account.pw_gid == 0:
        raise InfrastructureError("Policy account must not be root")
    return account


def _demote(account):
    def drop():
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    return drop


def kill_account_processes(uid):
    """SIGKILL every process owned by uid (daemonized leftovers escape the process group)."""
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if entry.stat().st_uid == uid:
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass


class Policy:
    def __init__(self, package, stderr_path, timeout=120):
        self.timeout = timeout
        self.buffer = b""
        self.sequence = 0
        self.stderr = Path(stderr_path).open("wb")
        # A predictor spawned inside MPMD must not accidentally rejoin its MPI world.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_"))}
        cwd, demote, self.home = "/tmp", None, None
        try:
            self.account = policy_account()
        except InfrastructureError:
            self.stderr.close()
            raise
        if self.account is not None:
            self.home = tempfile.mkdtemp(prefix="hydrogym-policy-")
            os.chown(self.home, self.account.pw_uid, self.account.pw_gid)
            env.update(HOME=self.home, TMPDIR=self.home, USER=self.account.pw_name,
                       LOGNAME=self.account.pw_name)
            cwd, demote = self.home, _demote(self.account)
        try:
            self.process = subprocess.Popen(
                ["bash", str(Path(package).resolve() / "infer.sh")], cwd=cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                start_new_session=True, env=env, bufsize=0, preexec_fn=demote)
        except OSError as error:
            self.stderr.close()
            if self.home:
                shutil.rmtree(self.home, ignore_errors=True)
            raise PolicyError(f"Could not launch policy: {error}") from error
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def ask(self, op, **payload):
        self.sequence += 1
        request_id = f"request-{self.sequence}"
        raw = (json.dumps(dict(protocol=PROTOCOL, op=op, request_id=request_id, **payload),
                          allow_nan=False) + "\n").encode()
        if len(raw) > MAX_LINE:
            raise InfrastructureError("Observation exceeds the declared protocol limit")
        deadline = time.monotonic() + self.timeout
        remaining = memoryview(raw)
        while remaining:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([], [self.process.stdin], [], left)[1]:
                raise PolicyError("Policy stdin timeout")
            try:
                count = os.write(self.process.stdin.fileno(), remaining)
                remaining = remaining[count:]
            except (BrokenPipeError, OSError) as error:
                raise PolicyError("Policy closed stdin") from error
        while b"\n" not in self.buffer:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.process.stdout], [], [], left)[0]:
                raise PolicyError("Policy response timeout")
            chunk = os.read(self.process.stdout.fileno(), min(65536, MAX_LINE + 1))
            if not chunk:
                raise PolicyError("Policy exited before replying")
            self.buffer += chunk
            if len(self.buffer) > MAX_LINE:
                raise PolicyError("Policy response exceeds protocol limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise PolicyError("Policy stdout is not JSON-lines") from error
        if not isinstance(response, dict) or response.get("request_id") != request_id:
            raise PolicyError("Wrong/missing response request_id")
        return response

    def reset(self, context, observation_shape, action_shape, low, high):
        self.low, self.high = low, high
        for shape in (observation_shape, action_shape):
            if not shape or any(type(n) is not int or n <= 0 for n in shape):
                raise InfrastructureError("Invalid native observation/action shape")
        self.observation_size = math.prod(observation_shape)
        if not numbers(low) or not numbers(high) or len(low) != len(high):
            raise InfrastructureError("Invalid native action bounds")
        if len(low) != math.prod(action_shape):
            raise InfrastructureError("Native action shape and bounds disagree")
        if any(a > b for a, b in zip(low, high)):
            raise InfrastructureError("Reversed native action bounds")
        reply = self.ask("reset", context=context, observation_shape=list(observation_shape),
                         action_shape=list(action_shape), action_low=low, action_high=high)
        if reply.get("ready") is not True:
            raise PolicyError("Policy did not acknowledge reset")

    def act(self, observation):
        if not numbers(observation) or len(observation) != self.observation_size:
            raise InfrastructureError("Nonfinite/invalid native observation")
        action = self.ask("act", observation=observation).get("action")
        if not numbers(action) or len(action) != len(self.low):
            raise PolicyError("Nonfinite or wrong-sized action")
        if any(x < a or x > b for x, a, b in zip(action, self.low, self.high)):
            raise PolicyError("Action outside native environment bounds")
        return action

    def close(self):
        try:
            if self.process.poll() is None:
                reply = self.ask("close")
                if reply.get("closed") is not True:
                    raise PolicyError("Policy did not acknowledge close")
                try:
                    code = self.process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as error:
                    raise PolicyError("Policy did not exit after close") from error
                if code:
                    raise PolicyError(f"Policy close exit code {code}")
            else:
                raise PolicyError("Policy exited before close")
        finally:
            self.abort()

    def abort(self):
        # Kill any descendants even if the original process has already exited.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.account is not None:
            kill_account_processes(self.account.pw_uid)
            shutil.rmtree(self.home, ignore_errors=True)
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


def replay_episode(env, policy, case, seed):
    """Only the trusted environment supplies returns; the policy supplies actions."""
    started = time.monotonic()
    observation, _ = env.reset(seed=seed)
    policy.reset({"environment": case["environment"], "family": case["family"], "seed": seed},
                 env.observation_space.shape, env.action_space.shape,
                 unpack(env.action_space.low), unpack(env.action_space.high))
    reward_sum = 0.0
    terminated = truncated = False
    info = {}
    for step in range(case["horizon"]):
        # Once the policy has acted, an invalid simulation state (nonfinite observation, force or
        # reward) is the outcome of its actions: the case fails, the rest of the suite is scored.
        # Invalid state before the first action, or a broken solver process, stays infrastructure.
        try:
            action = policy.act(unpack(observation))
        except InfrastructureError as error:
            if step == 0:
                raise
            raise PolicyError(f"Simulation diverged under the policy's actions by step {step}: {error}") from error
        try:
            observation, reward, terminated, truncated, info = env.step(action)
            if type(reward) not in (int, float) or not math.isfinite(reward):
                raise InfrastructureError("Native reward must be one finite scalar")
            reward_sum += reward
            if not math.isfinite(reward_sum):
                raise InfrastructureError("Nonfinite accumulated return")
        except InfrastructureError as error:
            raise PolicyError(f"Simulation diverged under the policy's actions at step {step + 1}: {error}") from error
        if terminated or truncated:
            break
    if not (terminated or truncated):
        raise InfrastructureError("Native episode did not finish within its reviewed horizon")
    diagnostics = {k: float(v) for k, v in info.items()
                   if type(v) in (int, float) and math.isfinite(v)}
    result = {"seed": seed, "return": reward_sum, "steps": step + 1,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "wall_time_sec": time.monotonic() - started, "diagnostics": diagnostics}
    if 'primary_metric' in case:
        result['primary_metric'] = env.evaluation_metrics()
    return result
