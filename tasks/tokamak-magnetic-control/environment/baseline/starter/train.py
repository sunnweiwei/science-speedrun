"""Starter: behaviour cloning from the privileged PID reference, then SAC fine-tuning.

Pipeline (all simulator work happens in parallel FreeGSNKE worker processes):
  1. Demonstrations: workers run the privileged PID/virtual-circuit reference on the
     randomized plant; episodes go to a demo replay.
  2. Normalizer: per-channel mean/std of the observation from the demo data (shipped
     inside the package as the actor's fixed preprocessing).
  3. Behaviour cloning: the actor (MLP on raw magnetics) regresses the PID voltages.
  4. SAC fine-tuning with an asymmetric twin critic (privileged descriptors), a BC
     regularizer on demo batches (TD3+BC style), conservative learning rates and
     target clipping; workers keep collecting with the latest exported actor.

Actor: MLP(obs) -> tanh -> voltages; deployable with NumPy only.
Usage: python baseline/starter/train.py --run /workspace/research/run-001 --workers 48 --hours 6
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from tokagym import CASES, TokamakMagneticEnv  # noqa: E402
from common import N_OBS, Scaler, save_actor, SL_PROBES, SL_COILS, SL_IP, N_PROBE, N_COIL  # noqa: E402
from workers import worker_main  # noqa: E402
from publish import publish  # noqa: E402

N_PRIV = 6


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, hidden=(256, 256, 256)):
        super().__init__()
        layers, d = [], n_obs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2 * n_act)
        self.n_act = n_act
        with torch.no_grad():  # start near zero volts, moderate exploration std
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()
            self.head.bias[n_act:] = -1.5

    def forward(self, x):
        out = self.head(self.body(x))
        return out[:, : self.n_act], out[:, self.n_act:].clamp(-5, 1)

    def sample(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    def export(self):
        params = []
        for m in self.body:
            if isinstance(m, nn.Linear):
                params.append((m.weight.detach().cpu().numpy().T, m.bias.detach().cpu().numpy()))
        W, b = self.head.weight.detach().cpu().numpy().T, self.head.bias.detach().cpu().numpy()
        params.append((W[:, : self.n_act], b[: self.n_act]))
        return params


class Critic(nn.Module):
    def __init__(self, n_in, hidden=(512, 512)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Replay:
    def __init__(self, capacity, n_obs, n_act, n_priv):
        self.cap = capacity; self.n = 0; self.i = 0
        self.obs = np.zeros((capacity, n_obs), np.float32)
        self.nobs = np.zeros((capacity, n_obs), np.float32)
        self.act = np.zeros((capacity, n_act), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.priv = np.zeros((capacity, n_priv), np.float32)
        self.npriv = np.zeros((capacity, n_priv), np.float32)
        self.base = np.zeros((capacity, n_act), np.float32)     # residual mode: base action at s
        self.nbase = np.zeros((capacity, n_act), np.float32)    # residual mode: base action at s'

    def add_episode(self, ep):
        T = len(ep["act"])
        priv_next = np.vstack([ep["priv"][1:], ep["priv"][-1:]])
        idx = (self.i + np.arange(T)) % self.cap
        self.obs[idx] = ep["obs"][:T]; self.nobs[idx] = ep["obs"][1:T + 1]; self.act[idx] = ep["act"]
        self.rew[idx] = ep["rew"]; self.done[idx] = ep["done"]
        self.priv[idx] = ep["priv"]; self.npriv[idx] = priv_next
        base = ep.get("base")
        if base is not None and len(base) == T:
            self.base[idx] = base; self.nbase[idx] = np.vstack([base[1:], base[-1:]])
        self.i = (self.i + T) % self.cap; self.n = min(self.n + T, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, batch)
        return (self.obs[idx], self.nobs[idx], self.act[idx], self.rew[idx], self.done[idx],
                self.priv[idx], self.npriv[idx], self.base[idx], self.nbase[idx])


def fit_scaler(obs):
    """Per-channel mean/std with physically sensible floors (Wb, T, A, m)."""
    mean = obs.mean(0).astype(np.float64)
    std = obs.std(0).astype(np.float64)
    floor = np.full(N_OBS, 1e-3)
    floor[SL_PROBES] = np.maximum(0.05 * np.abs(mean[SL_PROBES]), 1e-4)
    floor[SL_COILS] = 20.0
    floor[SL_IP] = 2e3
    floor[N_PROBE + N_COIL + 1:] = 2e-3      # geometry [m]
    floor[-2] = 2e3                          # target Ip [A]
    floor[-1] = 0.05                         # normalized time
    return Scaler(np.maximum(std, floor), mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", nargs="*", default=sorted(CASES))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--solver", default="nonlinear", choices=["nonlinear", "linear"])
    ap.add_argument("--randomization", default="visible")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--max-steps", type=int, default=3_000_000)
    ap.add_argument("--demo-episodes", type=int, default=4, help="PID demo episodes per worker")
    ap.add_argument("--demo-min-episodes", type=int, default=120, help="demos before normalizer + BC")
    ap.add_argument("--bc-steps", type=int, default=6000)
    ap.add_argument("--bc-batch", type=int, default=512)
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--bc-weight", type=float, default=1.0,
                    help="weight of the regularizer during SAC (BC-to-demos, or residual L2 in residual mode)")
    ap.add_argument("--q-weight", type=float, default=2.5,
                    help="TD3+BC normalisation: the actor maximises q_weight * Q / mean|Q| (so the regularizer keeps a fixed relative weight)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--actor-lr", type=float, default=3e-5)
    ap.add_argument("--critic-lr", type=float, default=1e-4)
    ap.add_argument("--alpha-lr", type=float, default=1e-4)
    ap.add_argument("--init-log-alpha", type=float, default=-3.0)
    ap.add_argument("--updates-per-step", type=float, default=0.5)
    ap.add_argument("--critic-warmup", type=int, default=3000, help="critic-only updates after BC")
    ap.add_argument("--explore-noise", type=float, default=0.02)
    ap.add_argument("--random-scale", type=float, default=0.02)
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--q-clip", type=float, default=150.0)
    ap.add_argument("--publish-every", type=int, default=48)
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--submission", default="/workspace/submission")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--residual-package", default=None,
                    help="deployable PID package dir; the actor then learns a residual on top of it")
    ap.add_argument("--residual-scale", type=float, default=0.2, help="residual amplitude as a fraction of vmax")
    ap.add_argument("--worker-timeout", type=float, default=900.0,
                    help="seconds without any worker message (heartbeats every 5 steps) before it is killed and respawned; "
                         "a killed episode is lost, so keep this well above 5x the slowest step on a loaded host")
    args = ap.parse_args()

    run = Path(args.run); run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(0); rng = np.random.default_rng(0)
    torch.set_num_threads(args.torch_threads)
    dev = torch.device(args.device)
    n_act = 12
    env0 = TokamakMagneticEnv(args.cases[0], randomization="none")
    vmax = env0.vmax.copy()
    vmax_t = torch.as_tensor(vmax, dtype=torch.float32, device=dev)
    scaler = Scaler(np.ones(N_OBS), np.zeros(N_OBS))   # replaced after the demo phase
    scale_t = torch.as_tensor(scaler.scale, dtype=torch.float32, device=dev)
    offset_t = torch.as_tensor(scaler.offset, dtype=torch.float32, device=dev)

    actor = Actor(N_OBS, n_act).to(dev)
    q1, q2 = Critic(N_OBS + N_PRIV + n_act).to(dev), Critic(N_OBS + N_PRIV + n_act).to(dev)
    q1t, q2t = Critic(N_OBS + N_PRIV + n_act).to(dev), Critic(N_OBS + N_PRIV + n_act).to(dev)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.full((), args.init_log_alpha, device=dev, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.critic_lr)
    opt_al = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = -float(n_act)
    log = open(run / "train.log", "a")

    residual = args.residual_package is not None
    res_scale_t = torch.as_tensor(args.residual_scale if residual else 0.0, device=dev)

    def export_actor(path_dir):
        extra = {"env_version": "tokagym-mastu/v0.2",
                 "algorithm": ("residual-sac on pid core" if residual else "bc-from-pid + sac-asymmetric-critic")}
        if residual:
            extra.update({"residual_package": str(Path(args.residual_package).resolve()),
                          "residual_scale": args.residual_scale})
        save_actor(path_dir, actor.export(), scaler, vmax, args.action_scale, extra)

    actor_dir = run / "actor"
    actor_path = actor_dir / "actor.npz"   # not exported until BC is done: workers use small random volts

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=4 * args.workers)
    wcfg = {"randomization": args.randomization, "solver": args.solver, "horizon": args.horizon,
            "explore_noise": args.explore_noise, "random_scale": args.random_scale,
            "demo_episodes": args.demo_episodes, "residual_package": args.residual_package,
            "residual_scale": args.residual_scale}
    procs, last_seen = {}, {}

    def spawn(w):
        cases = [args.cases[(w + k) % len(args.cases)] for k in range(len(args.cases))]
        p = ctx.Process(target=worker_main, args=(w, cases, queue, actor_path, wcfg), daemon=True)
        p.start(); procs[w] = p; last_seen[w] = time.time()

    for w in range(args.workers):
        spawn(w)

    replay = Replay(min(args.max_steps, 2_000_000), N_OBS, n_act, N_PRIV)
    demo = Replay(400_000, N_OBS, n_act, N_PRIV)
    steps = updates = episodes = demo_episodes = 0
    started = time.time()
    recent = {c: [] for c in args.cases}
    demo_recent = {c: [] for c in args.cases}
    best_score = None

    def window_score(windows):
        """(mean feasible fraction, -mean last-half shape RMS) over per-case windows; lexicographic."""
        feas = [np.mean([e[0] for e in v]) if v else 0.0 for v in windows.values()]
        shape = [np.mean([e[1] for e in v if e[1] is not None]) if any(e[1] is not None for e in v) else 1.0
                 for v in windows.values()]
        return (float(np.mean(feas)), -float(np.mean(shape)))
    update_debt = 0.0
    bc_done = False
    critic_updates = 0

    def scaled(x):
        return (torch.as_tensor(x, device=dev) - offset_t) / scale_t

    def bc_loss_on(o, a):
        mean, _ = actor(scaled(o))
        return F.mse_loss(torch.tanh(mean), torch.as_tensor(a, device=dev) / vmax_t)

    def run_bc():
        nonlocal scaler, scale_t, offset_t, best_score
        scaler = fit_scaler(demo.obs[: demo.n])
        scale_t = torch.as_tensor(scaler.scale, dtype=torch.float32, device=dev)
        offset_t = torch.as_tensor(scaler.offset, dtype=torch.float32, device=dev)
        opt_bc = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
        for k in range(0 if residual else args.bc_steps):
            o, _, a, *_ = demo.sample(args.bc_batch, rng)
            loss = bc_loss_on(o, a)
            opt_bc.zero_grad(); loss.backward(); opt_bc.step()
            if k % 500 == 0 or k == args.bc_steps - 1:
                log.write(f"bc step={k} loss={float(loss):.5f}\n"); log.flush()
        export_actor(actor_dir)
        publish(actor_dir, args.submission, {"stage": "bc", "demo_episodes": demo_episodes, "demo_steps": demo.n})
        # SAC snapshots are published only if they beat the demonstrator's own training-window score
        best_score = window_score(demo_recent)
        log.write(f"bc done: demos={demo_episodes} steps={demo.n}; actor exported and published; "
                  f"demo window score={best_score}\n"); log.flush()

    def compose(res, base):
        """Residual mode: total normalized action = clip((base + res * scale * vmax) / vmax)."""
        if not residual:
            return res
        return (base / vmax_t + res * res_scale_t).clamp(-1.0, 1.0)

    def sac_update(critic_only=False):
        o, no, a, r, d, pv, npv, b, nb = replay.sample(args.batch, rng)
        nan5 = (float("nan"), 0.0, float(log_alpha.exp()), float("nan"), float("nan"))
        if not all(np.isfinite(x).all() for x in (o, no, a, r, pv, npv, b, nb)):
            return nan5   # skip; workers drop such episodes
        o, no = scaled(o), scaled(no)
        a = torch.as_tensor(a, device=dev) / vmax_t
        b = torch.as_tensor(b, device=dev); nb = torch.as_tensor(nb, device=dev)
        r = torch.as_tensor(r, device=dev); d = torch.as_tensor(d, device=dev)
        pv = torch.as_tensor(pv, device=dev); npv = torch.as_tensor(npv, device=dev)
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            na, nlogp = actor.sample(no)
            qin = torch.cat([no, npv, compose(na, nb)], -1)
            qt = torch.min(q1t(qin), q2t(qin)) - alpha * nlogp
            y = (r + args.gamma * (1 - d) * qt).clamp(-args.q_clip, args.q_clip)
        qin = torch.cat([o, pv, a], -1)
        lq = F.mse_loss(q1(qin), y) + F.mse_loss(q2(qin), y)
        if not torch.isfinite(lq):
            return nan5
        opt_q.zero_grad(); lq.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
        opt_q.step()
        la = lal = torch.zeros(())
        res_mag = q_mean = float("nan")
        if not critic_only:
            pa, logp = actor.sample(o)
            qin = torch.cat([o, pv, compose(pa, b)], -1)
            q_pi = torch.min(q1(qin), q2(qin))
            # TD3+BC-style normalisation (Fujimoto & Gu 2021): the Q term has a fixed magnitude
            # q_weight relative to the regularizer, so critic extrapolation error cannot swamp it.
            lam = args.q_weight / (q_pi.abs().mean().detach() + 1e-3)
            la = (alpha * logp - lam * q_pi).mean()
            mean_res, _ = actor(o)
            if residual:   # keep the residual small: L2 penalty on the mean residual (normalized units)
                reg = (torch.tanh(mean_res) ** 2).mean()
            else:
                do, _, da, *_ = demo.sample(args.batch, rng)
                reg = bc_loss_on(do, da)
            la = la + args.bc_weight * reg
            res_mag = float(torch.tanh(mean_res).abs().mean()); q_mean = float(q_pi.mean())
            opt_a.zero_grad(); la.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            opt_a.step()
            lal = -(log_alpha * (logp.detach() + target_entropy)).mean()
            opt_al.zero_grad(); lal.backward(); opt_al.step()
            with torch.no_grad():
                log_alpha.clamp_(-8.0, 0.0)
        with torch.no_grad():
            for net, tgt in ((q1, q1t), (q2, q2t)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - args.tau).add_(args.tau * p)
        return float(lq), float(la), float(log_alpha.exp()), res_mag, q_mean

    import queue as queue_mod
    while steps < args.max_steps and (time.time() - started) < args.hours * 3600:
        try:
            msg = queue.get(timeout=30)
        except queue_mod.Empty:
            msg = None
        now = time.time()
        for w, p in list(procs.items()):   # watchdog: a worker stuck in a degenerate solve is replaced
            if now - last_seen[w] > args.worker_timeout or not p.is_alive():
                log.write(f"watchdog: worker {w} silent {now - last_seen[w]:.0f}s alive={p.is_alive()}; respawning\n")
                log.flush()
                try:
                    p.kill(); p.join(5)
                except Exception:  # noqa: BLE001
                    pass
                spawn(w)
        if msg is None:
            continue
        last_seen[msg["worker"]] = now
        if msg.get("heartbeat"):
            continue
        if "error" in msg:
            log.write(f"worker {msg['worker']} error: {msg['error']}\n"); log.flush(); continue
        T = len(msg["act"]); episodes += 1
        s = msg["summary"]
        if msg.get("demo"):
            demo.add_episode(msg); demo_episodes += 1
        replay.add_episode(msg); steps += T
        if msg.get("demo"):
            demo_recent[msg["case"]].append((s["feasible"], s["shape_rms_m_last_half"]))
            demo_recent[msg["case"]] = demo_recent[msg["case"]][-10:]
        elif bc_done:
            recent[msg["case"]].append((s["feasible"], s["shape_rms_m_last_half"]))
            recent[msg["case"]] = recent[msg["case"]][-10:]
        log.write(json.dumps({"episode": episodes, "steps": steps, "case": msg["case"], "seed": msg["seed"],
                              "demo": bool(msg.get("demo")), "wall_s": round(msg.get("wall_time_sec", 0), 1),
                              "setup_s": round(msg.get("setup_time_sec", 0), 1),
                              "feasible": s["feasible"], "shape_cm": None if s["shape_rms_m_last_half"] is None
                              else round(100 * s["shape_rms_m_last_half"], 3), "steps_ep": s["steps"],
                              "termination": s["termination"]}) + "\n"); log.flush()
        if not bc_done and demo_episodes >= args.demo_min_episodes:
            run_bc(); bc_done = True
            continue
        if bc_done:
            update_debt += T * args.updates_per_step
            last = None
            while update_debt >= 1:
                critic_only = critic_updates < args.critic_warmup
                last = sac_update(critic_only=critic_only); updates += 1; critic_updates += 1; update_debt -= 1
            if last is not None and updates % 500 < T * args.updates_per_step:
                log.write(f"t={time.time()-started:.0f}s steps={steps} updates={updates} lq={last[0]:.3f} "
                          f"la={last[1]:.3f} alpha={last[2]:.4f} |res|={last[3]:.3f} q={last[4]:.1f}\n"); log.flush()
            if episodes % 10 == 0:
                export_actor(actor_dir)
            if args.snapshot_every and episodes % args.snapshot_every == 0:
                snap = run / "snapshots" / f"ep{episodes:06d}-steps{steps:08d}"
                export_actor(snap)
                with open(run / "snapshots" / "index.jsonl", "a") as f:
                    f.write(json.dumps({"episodes": episodes, "steps": steps, "updates": updates,
                                        "time_sec": time.time() - started, "dir": str(snap)}) + "\n")
            if episodes % args.publish_every == 0:
                score = window_score(recent)
                torch.save({"actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                            "log_alpha": log_alpha.detach(), "steps": steps}, run / "checkpoint.pt")
                if best_score is None or score > best_score:
                    best_score = score
                    publish(actor_dir, args.submission, {"stage": "sac", "steps": steps, "episodes": episodes,
                                                         "score": score, "solver": args.solver,
                                                         "randomization": args.randomization})
                    log.write(f"published at episodes={episodes} steps={steps} score={score}\n"); log.flush()
    for p in procs.values():
        p.terminate()
    log.write("done\n"); log.close()
    (run / "DONE").write_text("done\n")


if __name__ == "__main__":
    main()
