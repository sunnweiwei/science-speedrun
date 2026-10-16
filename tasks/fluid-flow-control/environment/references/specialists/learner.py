"""TorchRL baseline reconstruction. Assumed hyperparameters are recorded, not hidden."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import NormalParamExtractor, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate, TD3Loss
from torchrl.objectives.value import GAE


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, 256),
                         nn.ReLU(), nn.Linear(256, outputs))


class QNetwork(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        self.network = mlp(observations + actions, 1)

    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=-1))


def networks(config):
    observations, actions = config["observations"], config["actions"]
    if config["algorithm"] == "PPO":
        actor = ProbabilisticActor(
            TensorDictModule(nn.Sequential(mlp(observations, 2 * actions), NormalParamExtractor()),
                             in_keys=["observation"], out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"], distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True, log_prob_key="sample_log_prob")
        critic = ValueOperator(mlp(observations, 1), in_keys=["observation"])
    else:
        actor = TensorDictModule(nn.Sequential(mlp(observations, actions), nn.Tanh()),
                                 in_keys=["observation"], out_keys=["action"])
        critic = ValueOperator(QNetwork(observations, actions), in_keys=["observation", "action"],
                               out_keys=["state_action_value"])
    return actor, critic


def observe(observation):
    return TensorDict({"observation": torch.as_tensor(observation, dtype=torch.float32)}, batch_size=[])


def action_for(actor, observation, algorithm, deterministic=False):
    with torch.no_grad():
        td = observe(observation)
        if algorithm == "PPO":
            distribution = actor.get_dist(td)
            action = distribution.deterministic_sample if deterministic else distribution.sample()
            td["action"] = action
            td["sample_log_prob"] = distribution.log_prob(action)
        else:
            actor(td)
    return td


def transition(td, observation, reward, done, terminated):
    td["next"] = TensorDict({"observation": torch.as_tensor(observation, dtype=torch.float32),
                             "reward": torch.tensor([reward], dtype=torch.float32),
                             "done": torch.tensor([done], dtype=torch.bool),
                             "terminated": torch.tensor([terminated], dtype=torch.bool)}, batch_size=[])
    return td


def optimize(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite learner loss")
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 0.5, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def run(env, args, provenance):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not np.all(env.action_space.low == -1) or not np.all(env.action_space.high == 1):
        raise ValueError("This pinball reconstruction expects native [-1, 1] actions")
    config = {"algorithm": args.algo, "observations": int(np.prod(env.observation_space.shape)),
              "actions": int(np.prod(env.action_space.shape)), "hidden": [256, 256],
              "activation": "ReLU", "gamma": 0.99, "gae_lambda": 0.95,
              "learning_rate": 0.0003 if args.algo == "PPO" else 0.001,
              "ppo_clip": 0.2, "ppo_minibatch": 32, "ppo_epochs": 10,
              "ppo_rollout": 200, "ppo_entropy_coeff": 0.0, "critic_coeff": 0.5,
              "replay_size": 100000, "offpolicy_batch": 256, "warmup": 1000,
              "target_tau": 0.005, "td3_policy_delay": 2, "target_noise": 0.2,
              "target_noise_clip": 0.5, "ou_theta": 0.15, "ou_sigma": 0.2, "ou_dt": 1.0,
              "torch_version": str(torch.__version__), "torchrl_version": "0.13.3",
              "parameter_source": "reconstructed; exact per-case author settings unavailable",
              "known_deviations": ["inferred 256x256 actor/critic and exact optimizer schedule",
                                   "fixed U_inf normalization and inferred sensors",
                                   "DDPG batch-normalization placement unspecified; omitted here",
                                   "OU theta/dt interpretation is inferred"]}
    saved = None
    if args.mode == "evaluate":
        if not args.checkpoint:
            raise ValueError("Evaluation requires a checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = saved["config"]
        if config["algorithm"] != args.algo or config["observations"] != int(np.prod(env.observation_space.shape)):
            raise ValueError("Checkpoint/environment mismatch")
    actor, critic = networks(config)
    if saved is not None:
        actor.load_state_dict(saved["actor"])
        actor.eval().requires_grad_(False)
        results = []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total = 0.0
            for _ in range(provenance["horizon"]):
                action = action_for(actor, obs, args.algo, deterministic=True)["action"].numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            results.append(total)
        (args.output / "evaluation.json").write_text(json.dumps({"status": "independent-pilot-evaluation",
             "checkpoint": str(args.checkpoint), "training_steps": saved["steps"], "returns": results,
             "mean_return": float(np.mean(results))}, indent=2) + "\n")
        return
    (args.output / "learner-config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.algo == "PPO":
        loss_module = ClipPPOLoss(actor, critic, clip_epsilon=0.2, entropy_bonus=False,
                                  critic_coeff=0.5, loss_critic_type="l2", normalize_advantage=True)
        advantage = GAE(gamma=0.99, lmbda=0.95, value_network=critic, time_dim=0)
        updater = None
    elif args.algo == "DDPG":
        loss_module = DDPGLoss(actor, critic, delay_actor=True, delay_value=True, loss_function="l2")
        loss_module.make_value_estimator(gamma=0.99)
        updater = SoftUpdate(loss_module, tau=0.005)
    else:
        loss_module = TD3Loss(actor, critic, bounds=(-1.0, 1.0), num_qvalue_nets=2,
                              policy_noise=0.2, noise_clip=0.5, loss_function="l2")
        loss_module.make_value_estimator(gamma=0.99)
        updater = SoftUpdate(loss_module, tau=0.005)
    parameters = list(loss_module.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    buffer = None if args.algo == "PPO" else TensorDictReplayBuffer(storage=LazyTensorStorage(100000), batch_size=256)
    log = (args.output / "updates.jsonl").open("a", buffering=1)
    started = time.monotonic()
    obs, _ = env.reset(seed=args.seed)
    ou = np.zeros(config["actions"], dtype=np.float32)
    frames, updates = 0, 0
    last_loss = last_norm = None

    def publish():
        pending = args.output / "candidate.pending.pt"
        torch.save({"actor": actor.state_dict(), "config": config, "steps": frames,
                    "seed": args.seed, "provenance": provenance}, pending)
        os.replace(pending, args.output / "candidate.pt")
        row = {"steps": frames, "updates": updates, "wall_sec": time.monotonic() - started,
               "loss": last_loss, "gradient_norm": last_norm, "algorithm": args.algo}
        log.write(json.dumps(row, allow_nan=False) + "\n")
        print("UPDATE", json.dumps(row), flush=True)

    try:
        while frames < args.steps:
            batch = []
            for _ in range(min(200, args.steps - frames)):
                td = action_for(actor, obs, args.algo)
                if args.algo != "PPO":
                    if frames < config["warmup"]:
                        value = np.random.uniform(-1, 1, config["actions"])
                    elif args.algo == "DDPG":
                        ou += -0.15 * ou + 0.2 * np.random.normal(size=ou.shape)
                        value = td["action"].numpy() + ou
                    else:
                        value = td["action"].numpy() + 0.1 * np.random.normal(size=config["actions"])
                    td["action"] = torch.as_tensor(np.clip(value, -1, 1), dtype=torch.float32)
                action = td["action"].numpy()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                td = transition(td, next_obs, reward, done, terminated)
                frames += 1
                if args.algo == "PPO":
                    batch.append(td)
                else:
                    buffer.add(td)
                    if frames >= max(config["warmup"], config["offpolicy_batch"]):
                        terms = loss_module(buffer.sample())
                        objective = terms["loss_value"] if args.algo == "DDPG" else terms["loss_qvalue"]
                        if args.algo == "DDPG" or updates % 2 == 0:
                            objective = objective + terms["loss_actor"]
                        last_norm = optimize(objective, optimizer, parameters)
                        last_loss = float(objective.detach())
                        if args.algo == "DDPG" or updates % 2 == 0:
                            updater.step()
                        updates += 1
                obs = next_obs
                if done:
                    obs, _ = env.reset()
                    ou.fill(0)
            if args.algo == "PPO":
                data = torch.stack(batch)
                with torch.no_grad():
                    advantage(data)
                for _ in range(config["ppo_epochs"]):
                    for indices in torch.randperm(len(data)).split(config["ppo_minibatch"]):
                        terms = loss_module(data[indices])
                        objective = terms["loss_objective"] + terms["loss_critic"]
                        last_norm = optimize(objective, optimizer, parameters)
                        last_loss = float(objective.detach())
                        updates += 1
            if updates:
                publish()
        (args.output / "completed.json").write_text(json.dumps({"status": "pilot-training-completed",
             "steps": frames, "updates": updates, "algorithm": args.algo}) + "\n")
    finally:
        log.close()


if __name__ == "__main__":
    # Dependency/optimizer smoke on synthetic dynamics, NOT a benchmark score.
    import argparse
    import gymnasium as gym
    import tempfile
    from types import SimpleNamespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-2, 2, shape=(3,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        def reset(self, seed=None):
            self.n = 0
            self.x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            return self.x.copy(), {}

        def step(self, action):
            self.n += 1
            self.x[:2] = np.clip(self.x[:2] + action * 0.02, -2, 2)
            return self.x.copy(), -float(np.square(self.x).mean()), self.n >= 200, False, {}

    with tempfile.TemporaryDirectory() as temp:
        for algorithm in ("PPO", "DDPG", "TD3"):
            output = Path(temp) / algorithm
            output.mkdir()
            args = SimpleNamespace(mode="train", seed=0, algo=algorithm, output=output, steps=1200, checkpoint=None, episodes=1)
            run(Toy(), args, {"horizon": 200})
            saved = torch.load(output / "candidate.pt", weights_only=True)
            assert saved["steps"] == 1200
            assert json.loads((output / "completed.json").read_text())["updates"] > 0
            args.mode, args.checkpoint = "evaluate", output / "candidate.pt"
            run(Toy(), args, {"horizon": 200})
            assert (output / "evaluation.json").is_file()
            print("SELF_TEST_PASSED", algorithm)
