"""Supplementary Eq.36, with an explicit reconstructed calibration binding.

This adapter does not certify the unknown original probes/target vectors. It is
deliberately independent of Gym/MAIA so the formula and binding can be tested
without launching CFD. Observation scaling must not alter physical reward.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


class CavityReward:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float64, copy=True)
        self.std = np.array(std, dtype=np.float64, copy=True)
        if (self.mean.ndim != 1 or not self.mean.size or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("Target must have matching finite vectors and strictly positive std")
        self.mean.flags.writeable = self.std.flags.writeable = False

    def __call__(self, raw_observation):
        obs = np.asarray(raw_observation, dtype=np.float64)
        if obs.shape != self.mean.shape or not np.isfinite(obs).all():
            raise ValueError("Expected one finite, unnormalized probe vector in calibrated order")
        standardized = (obs - self.mean) / self.std
        reward = -float(np.dot(standardized, standardized))
        if not np.isfinite(reward):
            raise ValueError("Nonfinite cavity reward")
        return reward

    @classmethod
    def from_calibration(cls, path, *, reynolds, probe_locations, observation_types, substeps):
        path = Path(path)
        document = json.loads(path.read_text())
        if document.get("status") != "reconstructed-cavity-calibration-not-paper-certified":
            raise ValueError("Missing accepted reconstructed calibration")
        binding = document["binding"]
        expected = {"reynolds": reynolds, "probe_locations": list(probe_locations),
                    "observation_types": list(observation_types), "substeps": substeps}
        if binding != expected:
            raise ValueError("Cavity calibration does not match physics/sensor ordering")
        result = cls(document["mean"], document["std_population"])
        result.calibration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return result


def install(env, calibration):
    """Replace only reward, which native MAIA calls before normalizing self.obs."""
    reward = CavityReward.from_calibration(calibration, reynolds=float(env.Re),
        probe_locations=env.probe_locations, observation_types=env.observation_type,
        substeps=int(env.num_substeps_per_iteration))
    if len(reward.mean) != env.num_outputs:
        raise ValueError("Native observation dimension does not match calibrated targets")
    env.get_reward = lambda: (reward(env.obs), {})
    return reward


def evaluate_with_diagnostics(reward, raw_observation, *, record_probes=False):
    """Keep physical probe traces separate from normalized policy observations."""
    obs = np.asarray(raw_observation, dtype=np.float64)
    score = reward(obs)
    diagnostics = {"raw_deviation_l2": float(np.linalg.norm(obs - reward.mean))}
    if not np.isfinite(diagnostics["raw_deviation_l2"]):
        raise ValueError("Nonfinite physical deviation diagnostic")
    if record_probes:
        diagnostics["raw_probe_velocity"] = obs.tolist()
    return score, diagnostics
