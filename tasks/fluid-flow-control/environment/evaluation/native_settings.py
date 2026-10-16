"""Reviewed scalar MAIA settings; never alter geometry or replace native rewards."""

import math

from protocol import InfrastructureError


ATTRIBUTES = {
    "max_episode_steps": "max_episode_steps",
    "substeps_per_action": "num_substeps_per_iteration",
    "max_control": "MAX_CONTROL",
    "omega": "omega",
}


def validate_settings(settings, horizon, episode_end="native"):
    if episode_end not in ("native", "time_limit"):
        raise InfrastructureError("Unknown reviewed episode_end convention")
    if not isinstance(settings, dict) or set(settings) - ATTRIBUTES.keys():
        raise InfrastructureError("Unknown/invalid reviewed native settings")
    for key, value in settings.items():
        if key in ("max_episode_steps", "substeps_per_action"):
            minimum = 0 if key == "max_episode_steps" else 1
            if type(value) is not int or value < minimum:
                raise InfrastructureError(f"Invalid native setting: {key}")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise InfrastructureError(f"Invalid native setting: {key}")
        if key == "max_control" and value == 0:
            raise InfrastructureError("Native actuator limit must be positive")
    offset = 1 if episode_end == "native" else 0
    if "max_episode_steps" in settings and settings["max_episode_steps"] + offset != horizon:
        raise InfrastructureError("Native completion setting and reviewed horizon disagree")


def apply_settings(env, case):
    settings = case.get("native_settings", {})
    episode_end = case.get("episode_end", "native")
    validate_settings(settings, case["horizon"], episode_end)
    defaults = {}
    for key, attribute in ATTRIBUTES.items():
        if not hasattr(env, attribute):
            raise InfrastructureError(f"Pinned MAIA attribute missing: {attribute}")
        value = getattr(env, attribute)
        defaults[key] = int(value) if key in ("max_episode_steps", "substeps_per_action") else float(value)
    for key, value in settings.items():
        setattr(env, ATTRIBUTES[key], value)
    effective = {key: settings.get(key, value) for key, value in defaults.items()}
    offset = 1 if episode_end == "native" else 0
    if env.max_episode_steps + offset != case["horizon"]:
        raise InfrastructureError("Native horizon differs from the reviewed release")
    return {"native_defaults": defaults, "effective": effective, "episode_end": episode_end}


def with_episode_limit(env, case):
    """Apply the same reviewed boundary in training and independent evaluation.

    Legacy releases keep their native flags. Public StepCounter reconstructions
    leave native max_episode_steps=horizon and truncate before native completion.
    Never hide an early/native failure by relabelling it a timeout.
    """
    if case.get("episode_end", "native") == "native":
        return env
    validate_settings(case.get("native_settings", {}), case["horizon"], case["episode_end"])
    import gymnasium as gym

    class ReviewedTimeLimit(gym.Wrapper):
        def __init__(self):
            super().__init__(env)
            self.count = None
            self.reviewed_protocol = env.reviewed_protocol

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self.count = 0
            return result

        def step(self, action):
            if self.count is None or self.count >= case["horizon"]:
                raise InfrastructureError("Reset is required before another reviewed episode")
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.count += 1
            if terminated or truncated:
                raise InfrastructureError("Native completion preceded the reviewed external time limit")
            return observation, reward, False, self.count == case["horizon"], info

    return ReviewedTimeLimit()
