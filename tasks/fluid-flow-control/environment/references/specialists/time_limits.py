"""Reconstruct the public TorchRL StepCounter without altering CFD rewards."""


def apply_time_limit(terminated, truncated, count, horizon, bootstrap):
    if not 0 < count <= horizon:
        raise ValueError("Control count outside the audited episode horizon")
    if not bootstrap:
        return bool(terminated), bool(truncated)
    # With native max_episode_steps=horizon, native completion (iter > limit)
    # must not fire within this window. Do not mask another termination cause.
    if terminated or truncated:
        raise ValueError("Native completion preceded the external time limit")
    return False, count == horizon
