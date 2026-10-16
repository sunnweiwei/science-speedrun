"""One shared PPO policy; padding/conditioning only, no specialist model bank."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy


def dimensions(suite):
    config = {"max_observation": max(c["observation_size"] for c in suite["cases"]),
            "max_action": max(c["action_size"] for c in suite["cases"]),
            "environments": sorted({c["environment"] for c in suite["cases"]}),
            "hidden": [256, 256]}
    if any(c.get("episode_end", "native") != "native" for c in suite["cases"]):
        # Keep legacy packages readable, but prohibit resuming them under changed
        # critic-bootstrap semantics just because tensor dimensions still match.
        config["episode_ends"] = {c.get("id", c["environment"]): c.get("episode_end", "native")
                                  for c in suite["cases"]}
    return config


def spaces(config):
    width = 2 * config["max_observation"] + len(config["environments"])
    return (gym.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32),
            gym.spaces.Box(-1, 1, shape=(config["max_action"],), dtype=np.float32))


def encode(observation, environment, config):
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    n, maximum = len(observation), config["max_observation"]
    if n > maximum or not np.isfinite(observation).all():
        raise ValueError("Observation exceeds starter capacity or is nonfinite")
    vector = np.zeros(spaces(config)[0].shape, dtype=np.float32)
    vector[:n] = np.clip(observation, -10, 10)
    vector[maximum:maximum + n] = 1.0
    if environment in config["environments"]:
        vector[2 * maximum + config["environments"].index(environment)] = 1.0
    return vector


def native_action(action, low, high):
    low, high = np.asarray(low), np.asarray(high)
    if len(action) < low.size:
        raise ValueError("Action exceeds starter capacity")
    normalized = np.clip(np.asarray(action).reshape(-1)[:low.size], -1, 1)
    return np.clip(low + (normalized + 1) * (high - low) / 2, low, high)


def build_policy(config):
    obs, action = spaces(config)
    return ActorCriticPolicy(obs, action, lr_schedule=lambda _: 0.0, net_arch=config["hidden"])


class SharedEnv(gym.Wrapper):
    def __init__(self, env, case, config):
        super().__init__(env)
        self.case, self.config = case, config
        self.observation_space, self.action_space = spaces(config)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return encode(observation, self.case["environment"], self.config), info

    def step(self, action):
        native = self.env.action_space
        action = native_action(action, native.low.reshape(-1), native.high.reshape(-1))
        obs, reward, terminated, truncated, info = self.env.step(action.astype(native.dtype).reshape(native.shape))
        if np.asarray(reward).ndim != 0 or not np.isfinite(reward):
            raise ValueError("Expected a finite scalar native reward")
        return encode(obs, self.case["environment"], self.config), float(reward), terminated, truncated, info
