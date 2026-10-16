"""Organizer-only two-case CFD integration check; never a full-suite score."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def cases(bootstrap_time_limits=False):
    definitions = [
        ("Pinball_2D_Re100", "pinball", [1.5 + i for i in range(7)],
         [-1.5 + 0.5 * i for i in range(7)], 98, 3, 170, 0.25),
        ("Cylinder_2D_Re3900", "cylinder", [1 + 7 * i / 3 for i in range(4)],
         [-0.75, 0.0, 0.75], 24, 1, 600, 0.03),
    ]
    result = []
    for name, family, xs, ys, obs, actions, substeps, limit in definitions:
        result.append({"id": name, "environment": name, "family": family,
                       "horizon": 200, "observation_size": obs, "action_size": actions,
                       "seeds": [30], "env_kwargs": {
                           "probe_locations": [c for x in xs for y in ys for c in (x, y)],
                           "obs_normalization_strategy": "U_inf"},
                       "native_settings": {"max_episode_steps": 199, "substeps_per_action": substeps,
                                           "max_control": limit, "omega": 1.0}})
        if bootstrap_time_limits:
            result[-1]["episode_end"] = "time_limit"
            result[-1]["native_settings"]["max_episode_steps"] = 200
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["train", "replay"])
    parser.add_argument("--case-index", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bootstrap-time-limits", action="store_true")
    args = parser.parse_args()
    args.output, args.data = args.output.resolve(), args.data.resolve()
    sys.path.insert(0, "/workspace/evaluation")
    sys.path.insert(0, "/workspace/baseline/starter")
    from evaluate import package_hash, write_json
    from run_maia import make_env, evaluate_worker, check_native_cache
    import hydrogym.data_manager as manager
    manager.HF_AVAILABLE = False
    receipt = json.loads((args.data / "materialized.json").read_text())
    if receipt["revision"] != "cf5ef6748443ae72e9647f398487e98d421a7243":
        raise RuntimeError("Wrong resource revision")
    suite = {"cases": cases(args.bootstrap_time_limits), "policy_timeout_sec": 120,
             "status": "two-case-integration-diagnostic-not-release"}
    if any(c["environment"] not in receipt["verified_directories"] for c in suite["cases"]):
        raise RuntimeError("Unverified resources")
    case = suite["cases"][args.case_index]
    resume = args.output / "research/shared-ppo.zip"
    submission = args.output / "submission"
    if args.worker == "train":
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from train import train
        torch.set_num_threads(1)
        torch.manual_seed(30)
        np.random.seed(30)
        previous_steps = 0
        previous_hash = None
        if resume.exists():
            previous_steps = int(PPO.load(resume, device="cpu").num_timesteps)
            previous_hash = hashlib.sha256(resume.read_bytes()).hexdigest()
        env = make_env(case, args.data)
        native_protocol = env.reviewed_protocol
        train(env, suite, case, 400, resume, submission, publish_every=200)
        model = PPO.load(resume, device="cpu")
        updated_hash = hashlib.sha256(resume.read_bytes()).hexdigest()
        if int(model.num_timesteps) != previous_steps + 400 or updated_hash == previous_hash:
            raise RuntimeError("Shared training phase did not resume/update the checkpoint")
        write_json(args.output / f"train-{args.case_index}.json", {
            "status": "native-shared-training-phase", "environment": case["environment"],
            "previous_steps": previous_steps, "training_steps": int(model.num_timesteps),
            "previous_checkpoint_sha256": previous_hash, "checkpoint_sha256": updated_hash,
            "native_protocol": native_protocol, "submission_sha256": package_hash(submission)})
        return
    if args.worker == "replay":
        evaluate_worker(case, suite, args.data, args.output / "relocated-policy",
                        args.output / f"replay-{args.case_index}.json")
        return

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "source.py")
    write_json(args.output / "protocol.json", suite)
    from hydrogym.maia.workspace import prepare_maia_workspace

    def launch(mode, index):
        item = suite["cases"][index]
        check_native_cache(item, args.data)
        work, props = prepare_maia_workspace(item["environment"],
            work_dir=str(args.output / f"cfd-{mode}-{index}"), use_clean_cache=False,
            local_fallback_dir=str(args.data / "environments"))
        command = ["mpirun", "--bind-to", "none", "-np", "1", sys.executable, str(Path(__file__).resolve()),
                   "--worker", mode, "--case-index", str(index), "--data", str(args.data),
                   "--output", str(args.output)]
        if args.bootstrap_time_limits:
            command.append("--bootstrap-time-limits")
        command += [":", "-np", "1", "maia", props, "--silent"]
        environment = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                           OMPI_MCA_coll_hcoll_enable="0", HF_HUB_OFFLINE="1")
        with (args.output / f"{mode}-{index}.log").open("w") as stream:
            subprocess.run(command, cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    for index in range(2):
        launch("train", index)
    original_hash = package_hash(submission)
    shutil.copytree(submission, args.output / "relocated-policy")
    if package_hash(args.output / "relocated-policy") != original_hash:
        raise RuntimeError("Submission changed during relocation")
    for index in range(2):
        launch("replay", index)
        result = json.loads((args.output / f"replay-{index}.json").read_text())
        if result["status"] != "ok" or result["episodes"][0]["steps"] != 200:
            raise RuntimeError("Incomplete native shared-policy replay")
        if args.bootstrap_time_limits and (result["episodes"][0]["terminated"]
                                          or not result["episodes"][0]["truncated"]):
            raise RuntimeError("Shared replay lost the reviewed critic-bootstrap boundary")
        if package_hash(args.output / "relocated-policy") != original_hash:
            raise RuntimeError("Shared model mutated during replay")
    write_json(args.output / "completed.json", {
        "status": "shared-two-case-native-integration-passed-not-benchmark-score",
        "training_steps": 800, "case_count": 2, "submission_sha256": original_hash,
        "time_limit_bootstrap": args.bootstrap_time_limits})


if __name__ == "__main__":
    main()
