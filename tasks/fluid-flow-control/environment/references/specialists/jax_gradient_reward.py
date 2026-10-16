"""Supplement Eq.34's explicit reward-augmentation alternative.

This is not the unpublished adaptive Eq.32 GPPO implementation. Differentiate
the full CFD interval with respect to its applied action, holding its initial
state fixed. For a maximized physical reward r, Eq.34 becomes r + grad(r).a.
The physical return is retained separately and must be used for reporting.
"""

import jax
import jax.numpy as jnp


def equation34_flow(base):
    class GradientRewardFlow(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Recompute RK intermediates in reverse mode; never change dt/grid.
            original_rk = self.integrator.RK4_CN
            self.integrator.RK4_CN = lambda control_field=None: jax.checkpoint(original_rk(control_field))

        def step_env(self, key, state, action, params):
            transition = super().step_env

            def objective(control):
                obs, next_state, reward, done, info = transition(key, state, control, params)
                return reward, (obs, next_state, done, info)

            (physical_reward, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(action)
            obs, next_state, done, info = auxiliary
            correction = jnp.vdot(gradient, action)
            info = dict(info, physical_reward=physical_reward,
                        analytic_reward_correction=correction,
                        analytic_reward_gradient=gradient)
            return obs, next_state, physical_reward + correction, done, info

    return GradientRewardFlow
