"""Bounded, resident action-only policy protocol (``tokamak-policy/v1``).

Wire format is identical to HydroGym's ``hydrogym-policy/v1``: UTF-8 JSON lines on
stdin/stdout, one outstanding request, echoed ``request_id``. No simulator or ML
dependencies live in this module.
"""

import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time

PROTOCOL = "tokamak-policy/v1"
MAX_LINE = 64 * 1024 * 1024


class PolicyError(RuntimeError):
    """The candidate violated the protocol, bounds or timing."""


class InfrastructureError(RuntimeError):
    """The evaluator itself could not proceed; never attributed to the candidate."""


def numbers(values):
    return isinstance(values, list) and all(
        type(v) in (int, float) and math.isfinite(v) for v in values)


def unpack(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [number for part in value for number in unpack(part)]
    return [float(value)]


class Policy:
    def __init__(self, package, stderr_path, timeout=120):
        self.timeout = timeout
        self.buffer = b""
        self.sequence = 0
        self.latencies = []
        self.stderr = Path(stderr_path).open("wb")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_"))}
        try:
            self.process = subprocess.Popen(
                ["bash", str(Path(package).resolve() / "infer.sh")], cwd="/tmp",
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                start_new_session=True, env=env, bufsize=0)
        except OSError as error:
            self.stderr.close()
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
        reply = self.ask("reset", context=context, observation_shape=list(observation_shape),
                         action_shape=list(action_shape), action_low=low, action_high=high)
        if reply.get("ready") is not True:
            raise PolicyError("Policy did not acknowledge reset")

    def act(self, observation):
        if not numbers(observation) or len(observation) != self.observation_size:
            raise InfrastructureError("Nonfinite/invalid native observation")
        started = time.perf_counter()
        action = self.ask("act", observation=observation).get("action")
        self.latencies.append(time.perf_counter() - started)
        if not numbers(action) or len(action) != len(self.low):
            raise PolicyError("Nonfinite or wrong-sized action")
        if any(x < a or x > b for x, a, b in zip(action, self.low, self.high)):
            raise PolicyError("Action outside machine voltage limits")
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
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # already exited (macOS reports EPERM)
            pass
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
