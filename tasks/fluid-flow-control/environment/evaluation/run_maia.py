"""Native MAIA launcher and trusted episode worker; no solver/reward substitutes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from evaluate import preflight, write_json
from protocol import InfrastructureError, Policy, PolicyError, replay_episode
from native_settings import apply_settings, with_episode_limit


def check_native_cache(case, data):
    # Native MAIA gives this legacy cache precedence over local_fallback_dir.
    # Never silently use another resource version left in a base image.
    cached = Path.home() / ".cache" / "maiagym" / case["environment"]
    expected = (data / "environments" / case["environment"]).resolve()
    if cached.exists() and cached.resolve() != expected:
        raise InfrastructureError(f"Legacy MAIA cache would shadow pinned data: {cached}")


def make_env(case, data):
    import numpy as np
    import hydrogym.maia as maia

    check_native_cache(case, data)
    env = maia.from_hf(case["environment"], use_clean_cache=False,
                       local_fallback_dir=str(data / "environments"), **case["env_kwargs"])
    if Path(env.env_data_path).resolve() != (data / "environments" / case["environment"]).resolve():
        raise InfrastructureError("Native environment loaded unpinned cached resources")
    if int(np.prod(env.observation_space.shape)) != case["observation_size"] or int(np.prod(env.action_space.shape)) != case["action_size"]:
        raise InfrastructureError("Native spaces differ from the reviewed release")
    # Apply organizer-reviewed paper settings equally for training and replay.
    # Never mutate the read-only source YAML or infer settings from policy files.
    env.reviewed_protocol = apply_settings(env, case)
    return with_episode_limit(env, case)


def evaluate_worker(case, suite, data, submission, output):
    import numpy as np

    env = make_env(case, data)

    class Adapter:
        observation_space = env.observation_space
        action_space = env.action_space

        def reset(self, seed):
            return env.reset(seed=seed)

        def step(self, action):
            obs, reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=env.action_space.dtype).reshape(env.action_space.shape))
            if np.asarray(reward).ndim != 0:
                raise InfrastructureError("Native multi-agent/vector reward needs an explicitly reviewed adapter")
            return obs, float(reward), terminated, truncated, info

    policy = None
    result = {"status": "ok", "episodes": [], "native_protocol": env.reviewed_protocol}
    try:
        policy = Policy(submission, output.with_suffix(".policy.log"), suite["policy_timeout_sec"])
        for seed in case["seeds"]:
            result["episodes"].append(replay_episode(Adapter(), policy, case, seed))
        policy.close()
        policy = None
    except PolicyError as error:
        result = {"status": "policy_error", "error": str(error)}
    finally:
        if policy is not None:
            policy.abort()
        env.close()
    write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["evaluate", "train"])
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--resume", type=Path, default=Path("/workspace/research/shared-ppo.zip"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.submission, args.resume = args.submission.resolve(), args.resume.resolve()
    # Training and native replay depend on fixed physics, not on a learned
    # reference already having achieved its target. The outer grader checks
    # score anchors separately when a normalized aggregate is requested.
    suite = preflight(args.data, require_reference=False)
    case = next((c for c in suite["cases"] if c["id"] == args.case), None)
    if case is None:
        parser.error("Case is not in the fixed release")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.worker:
        if args.mode == "evaluate":
            evaluate_worker(case, suite, args.data, args.submission, args.output)
        else:
            # The editable training code is not imported by the verifier.
            sys.path.insert(0, "/workspace/baseline/starter")
            from train import train
            train(make_env(case, args.data), suite, case, args.steps, args.resume, args.submission)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep solver logs/restarts beside the diagnostic output, not in source data.
    scratch = Path(tempfile.mkdtemp(prefix=f"{case['id']}-", dir=args.output.parent))
    check_native_cache(case, args.data)
    # Upstream snapshot_download omits revision. A fresh, offline Hub cache
    # forces its explicit local fallback instead of resolving a cached main.
    os.environ["HF_HUB_CACHE"] = str(scratch / "hf-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from hydrogym.maia.workspace import prepare_maia_workspace

    work, properties = prepare_maia_workspace(
        case["environment"], work_dir=str(scratch), use_clean_cache=False,
        local_fallback_dir=str(args.data / "environments"))
    solver = ([sys.executable, str(Path(__file__).resolve().with_name("maia_gpu_rank.py"))]
              if case["solver_processes"] > 1 else ["maia"])
    command = ["mpirun", "--bind-to", "none", "-np", "1", sys.executable, str(Path(__file__).resolve()),
               args.mode, "--worker", "--data", str(args.data), "--case", args.case,
               "--submission", str(args.submission), "--output", str(args.output),
               "--steps", str(args.steps), "--resume", str(args.resume),
               ":", "-np", str(case["solver_processes"]), *solver, properties, "--silent"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                       OMPI_MCA_coll_hcoll_enable="0")
    subprocess.run(command, cwd=work, env=environment, check=True)


if __name__ == "__main__":
    main()
