"""Real corrected-CFD derivative and Eq.34 wiring check, without policy training."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_source import verify
from jax_corrections import apply_corrections
from jax_public import MixingFlow
from jax_gradient_reward import equation34_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("jax_gradient_reward_check.py", "jax_gradient_reward.py", "jax_public.py",
                 "jax_source.py", "jax_corrections.py"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    source_audit = verify()
    jax.config.update("jax_enable_x64", True)
    correction = apply_corrections("dealias-curl")
    plain = MixingFlow(env_config={}, flow_config={"Re": 250, "grid_size": (64, 64)})
    augmented = equation34_flow(MixingFlow)(env_config={}, flow_config={"Re": 250, "grid_size": (64, 64)})
    params = plain.default_params
    key = jax.random.PRNGKey(30)
    started = time.monotonic()
    _, state = jax.jit(plain.reset_env)(key, params)
    ordinary_step = jax.jit(plain.step_env)
    gradient_step = jax.jit(augmented.step_env)
    rows = []
    for action in (jnp.array([.13, .07, .04, .02]), jnp.array([.09, .02, .03, .04])):
        before = state
        obs, state, reward, done, info = ordinary_step(key, before, action, params)
        grad_obs, grad_state, shaped, grad_done, grad_info = gradient_step(key, before, action, params)
        shaped.block_until_ready()
        np.testing.assert_allclose(np.asarray(grad_obs), np.asarray(obs), rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(np.asarray(grad_state.omega_hat), np.asarray(state.omega_hat), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(float(grad_info["physical_reward"]), float(reward), rtol=1e-8, atol=1e-9)
        if bool(done) != bool(grad_done):
            raise RuntimeError("Eq.34 altered episode completion")
        gradient = np.asarray(grad_info["analytic_reward_gradient"])
        if not np.isfinite(gradient).all() or not np.isfinite(float(shaped)):
            raise RuntimeError("Nonfinite Eq.34 diagnostic")
        # All diagnostic actions are positive, so d(-75*L1)/da=-75 exactly.
        physical_gradient = gradient + 75.0
        if np.linalg.norm(physical_gradient) <= 1e-6:
            raise RuntimeError("No CFD sensitivity beyond the explicit action cost")
        np.testing.assert_allclose(float(shaped), float(reward) + gradient @ np.asarray(action), rtol=1e-10, atol=1e-10)
        direction = jnp.array([.3, -.2, .4, .1])
        analytical = float(physical_gradient @ np.asarray(direction))
        differences = []
        for eps in (1e-5, 1e-6):
            plus = ordinary_step(key, before, action + eps * direction, params)[4]["mean_tke"]
            minus = ordinary_step(key, before, action - eps * direction, params)[4]["mean_tke"]
            numerical = float((plus - minus) / (2 * eps))
            differences.append({"epsilon": eps, "analytical_tke": analytical, "numerical_tke": numerical})
            np.testing.assert_allclose(analytical, numerical, rtol=.02, atol=1e-6)
        row = {"action": action.tolist(), "physical_reward": float(reward), "augmented_reward": float(shaped),
               "reward_gradient": gradient.tolist(), "tke_gradient": physical_gradient.tolist(),
               "finite_differences": differences}
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    result = {"status": "Eq34-native-gradient-wiring-validated-not-paper-performance", "steps": rows,
              "wall_seconds": time.monotonic() - started, "imported_cfd_sources": source_audit,
              "solver_corrections": correction, "float64_diagnostic": True, "devices": str(jax.devices()),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
