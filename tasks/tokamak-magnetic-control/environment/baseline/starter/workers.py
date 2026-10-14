"""Simulator workers: each process owns one FreeGSNKE environment per assigned case.

Workers run episodes with the latest exported actor (NumPy), add exploration noise,
and push complete episodes to the learner through a queue. Weight refresh happens at
episode boundaries by re-reading ``actor_path`` when its mtime changes.
"""

import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tokagym import TokamakMagneticEnv  # noqa: E402
from tokagym import cases as C  # noqa: E402

from common import NumpyActor  # noqa: E402

PRIV_KEYS = ("shape_rms", "xpoint_error", "ip_error_rel", "Z_axis", "R_axis", "coil_violation")
DESC_KEYS = ("Rin", "Rout", "Rx", "Zx", "Zcur")


def descriptors_of(info):
    return np.array([info.get(k, np.nan) for k in DESC_KEYS], dtype=np.float32)


def privileged(info, env):
    """Critic-only descriptors; a solver-failure step carries no axis position (use the reference)."""
    z = info.get("Z_axis", env.reference_axis[1]) - env.reference_axis[1]
    r = info.get("R_axis", env.reference_axis[0]) - env.reference_axis[0]
    return np.array([min(info["shape_rms"], 1.0), min(info["xpoint_error"], 0.5),
                     min(info["ip_error_rel"], 1.0), z, r, info["coil_violation"]], dtype=np.float32)


class DemoController:
    """Privileged PID/virtual-circuit reference used to seed the replay with stabilizing data."""

    def __init__(self, env):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pid"))
        import pid_reference as P
        self.P = P
        if env.case.diverted:
            lower = env.target_xpoints[np.argmin(env.target_xpoints[:, 1])]
            box = [(lower[0] - 0.3, lower[1] - 0.3), (lower[0] + 0.3, lower[1] + 0.3)]
            self.targets = ["Rin", "Rout", "Rx", "Zx"]
        else:
            box, self.targets = None, ["Rin", "Rout"]
        self.descriptors = P.make_descriptor_fn(box)
        self.env = env
        self.pcs = None
        self.n = len(P.CTRL_COILS)

    def start(self, env):
        # fresh controller state per episode (as in baseline/pid/pid_reference.py)
        self.pcs = self.P.build_pcs(env, self.targets, self.descriptors, tmax=env.horizon * C.DT)
        self.ip_hist = self.ip_err = 0.0
        self.T_err = np.zeros(len(self.targets)); self.T_hist = np.zeros(len(self.targets))
        self.I_prev = env._coil_currents()[: self.n]; self.V_prev = np.zeros(self.n)
        self.z_prev = env.stepping.eq1.Zcurrent(); self.zipv = 0.0; self.t = 0.0
        self.resist = np.asarray(env.tokamak.coil_resist[: env.n_active], dtype=float)

    def act(self, env):
        import contextlib, io
        eq = env.stepping.eq1
        ip = env._plasma_current(); z = eq.Zcurrent(); dt = env.stepping.dt_step
        with contextlib.redirect_stdout(io.StringIO()):
            V, self.ip_hist, self.ip_err, self.T_err, self.T_hist, I_appr, _ = self.pcs.calculate_ctrl_voltages(
                t=self.t, dt=dt, dt_simulator=dt, ip_meas=ip, ip_hist_prev=self.ip_hist, ip_err_prev=self.ip_err,
                T_meas=np.asarray(self.descriptors(eq), dtype=float), T_err_prev=self.T_err, T_hist_prev=self.T_hist,
                I_approved_prev=self.I_prev, I_meas=env._coil_currents()[: self.n], V_approved_prev=self.V_prev,
                zip_meas=z * ip, zipv_meas=self.zipv, active_coil_resists=self.resist.copy(), verbose=False)
        V = np.clip(np.asarray(V, dtype=float).ravel(), -env.vmax, env.vmax)
        self.zipv = ((z - self.z_prev) / dt) * ip
        self.z_prev = z
        self.I_prev = np.asarray(I_appr, dtype=float).ravel()[: self.n]
        self.V_prev = V[: self.n]
        self.t += dt
        return V


def load_base_controller(package_dir):
    """Deployable base package (``Controller`` with reset/act, e.g. baseline/shape_vc/package) for residual RL."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pid_core", str(Path(package_dir) / "policy.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Controller()


def worker_main(worker_id, case_names, queue, actor_path, cfg):
    """Loop forever: sample case, run one episode, push it."""
    np.random.seed(worker_id * 7919 + int(time.time()) % 1000)
    rng = np.random.default_rng(worker_id * 7919 + int(time.time()) % 1000)
    envs, demos = {}, {}
    actor, mtime = None, None
    episode_counter = 0
    demo_budget = int(cfg.get("demo_episodes", 0))
    base_ctl = load_base_controller(cfg["residual_package"]) if cfg.get("residual_package") else None
    residual_scale = float(cfg.get("residual_scale", 0.2))
    while True:
        try:
            if actor_path.exists() and actor_path.stat().st_mtime != mtime:
                try:  # the learner may be rewriting the file; keep the previous actor on failure
                    candidate = NumpyActor.load(actor_path.parent)
                    actor, mtime = candidate, actor_path.stat().st_mtime
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
            case = case_names[episode_counter % len(case_names)]
            if case not in envs:
                envs[case] = TokamakMagneticEnv(case, randomization=cfg["randomization"],
                                                solver=cfg["solver"], horizon=cfg.get("horizon"))
            env = envs[case]
            use_demo = demo_budget > 0
            if use_demo and case not in demos and base_ctl is None:
                demos[case] = DemoController(env)
            seed = int(rng.integers(0, 2**31 - 1))
            t_ep = time.monotonic()
            obs, info = env.reset(seed=seed)
            setup_time = info["setup_time_sec"]
            if use_demo and base_ctl is None:
                demos[case].start(env)
            if base_ctl is not None:
                base_ctl.reset((-env.vmax).tolist(), env.vmax.tolist())
            noise = cfg["explore_noise"]
            O, A, R, D, P, DS, BS = [], [], [], [], [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                base = base_ctl.act(obs) if base_ctl is not None else np.zeros(env.n_active)
                if base_ctl is not None:
                    # residual policy: demos run the base alone; afterwards actor + exploration noise
                    res = np.zeros(env.n_active) if (use_demo or actor is None) else actor(obs) * residual_scale
                    a = base + res + (0.0 if use_demo else rng.normal(0, 1, env.n_active) * env.vmax * noise)
                    a = np.clip(a, -env.vmax, env.vmax)
                elif use_demo:
                    a = demos[case].act(env)
                elif actor is None:
                    a = rng.uniform(-1, 1, env.n_active) * env.vmax * cfg["random_scale"]
                else:
                    a = actor(obs) + rng.normal(0, 1, env.n_active) * env.vmax * noise
                    a = np.clip(a, -env.vmax, env.vmax)
                next_obs, r, terminated, truncated, info = env.step(a)
                O.append(np.asarray(obs, dtype=np.float32)); A.append(np.asarray(a, dtype=np.float32))
                R.append(np.float32(r)); D.append(np.float32(terminated)); P.append(privileged(info, env))
                DS.append(descriptors_of(info)); BS.append(np.asarray(base, dtype=np.float32))
                obs = next_obs
                if env.k % 5 == 0:
                    queue.put({"worker": worker_id, "heartbeat": True})
            O.append(np.asarray(obs, dtype=np.float32))
            summary = env.episode_summary()
            arrays = {"obs": np.stack(O), "act": np.stack(A), "rew": np.array(R), "priv": np.stack(P),
                      "base": np.stack(BS)}
            bad = [k for k, v in arrays.items() if not np.all(np.isfinite(v))]
            if bad:
                queue.put({"worker": worker_id, "error": f"non-finite {bad} in episode of {case} seed {seed}; dropped"})
                episode_counter += 1
                continue
            queue.put({"worker": worker_id, "case": case, "seed": seed, "obs": np.stack(O),
                       "act": np.stack(A), "rew": np.array(R), "done": np.array(D), "priv": np.stack(P),
                       "desc": np.stack(DS), "base": np.stack(BS), "summary": summary, "actor_mtime": mtime,
                       "demo": use_demo,
                       "wall_time_sec": time.monotonic() - t_ep, "setup_time_sec": setup_time})
            if use_demo:
                demo_budget -= 1
            episode_counter += 1
        except Exception:  # noqa: BLE001
            queue.put({"worker": worker_id, "error": traceback.format_exc()[-1500:]})
            time.sleep(5)
