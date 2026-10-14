"""Resident tokamak-policy/v1 server: estimator + PID/virtual circuits (no simulator access).

Set-points come from the observation (target boundary points, target lower X-point, target
Ip) and are ramped from the first estimate to the targets; descriptors come from a linear
estimator on the real-time magnetics; virtual circuits are scheduled on the stored
reference shape nearest to the observation's targets. The control law reuses FreeGSNKE's
PlasmaControlSystem, available in the declared runtime image.
"""

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = "tokamak-policy/v1"
DT = 5e-4
N_PROBE, N_COIL = 500, 12
N_RT = N_PROBE + N_COIL + 1
CTRL_COILS = ["Solenoid", "PX", "D1", "D2", "D3", "Dp", "D5", "D6", "D7", "P4", "P5"]
ACTIVE = CTRL_COILS + ["P6"]


def waveform(times, vals):
    return {"times": np.asarray(times, dtype=float), "vals": np.asarray(vals, dtype=float)}


class Controller:
    def __init__(self, filter_alpha=None):
        est = np.load(HERE / "estimator.npz")
        self.mu, self.sd, self.W = est["mu"], est["sd"], est["W"]
        self.m = json.loads((HERE / "machine.json").read_text())
        self.targets = self.m["targets"]
        self.refs = self.m.get("references") or [self.m]
        for r in self.refs:
            r["_pts"] = np.asarray(r["target_points"], dtype=float)
        self.ref = self.refs[0]
        self.ramp_steps = int(self.m.get("ramp_steps", 0))
        self.pcs = None
        self.pcs_ref = None
        # Exponential smoothing of the estimated descriptors (per-step measurement noise is
        # i.i.d.; the plasma moves slowly relative to 0.5 ms). Vertical position keeps a
        # faster filter because the vertical mode grows in ~15 ms.
        fa = self.m.get("filter_alpha", [0.15, 0.15, 0.15, 0.15, 0.35]) if filter_alpha is None else filter_alpha
        self.filter_alpha = np.asarray(fa, dtype=float)
        self.d_filt = None

    # ---- setpoints from the observation --------------------------------------------
    def select_reference(self, obs):
        """Nearest stored reference shape (RMS distance of the 32 target points, plus Ip)."""
        pts = np.asarray(obs[N_RT: N_RT + 64]).reshape(32, 2)
        ip_t = float(obs[N_RT + 70])
        def dist(r):
            return float(np.sqrt(((pts - r["_pts"]) ** 2).sum(1).mean())) + 0.5 * abs(ip_t - r["reference_ip"]) / r["reference_ip"]
        self.ref = min(self.refs, key=dist)
        return self.ref

    def setpoints(self, obs):
        base = N_RT
        pts = np.asarray(obs[base: base + 64]).reshape(32, 2)
        xp = np.asarray(obs[base + 64: base + 68]).reshape(2, 2)
        flags = np.asarray(obs[base + 68: base + 70])
        ip_t = float(obs[base + 70])
        z_mid = np.median(pts[:, 1])
        near = np.abs(pts[:, 1] - z_mid) < 0.25
        off = self.ref.get("setpoint_offsets", {"Rin": 0.0, "Rout": 0.0})
        rin = float(pts[near, 0].min()) + off["Rin"]
        rout = float(pts[near, 0].max()) + off["Rout"]
        xs = xp[flags > 0.5]
        if len(xs):
            lower = xs[np.argmin(xs[:, 1])]
            rx, zx = float(lower[0]), float(lower[1])
        else:
            rx, zx = np.nan, np.nan
        # vertical set-point: the reference current centroid, shifted with the targets
        zc = float(self.ref.get("reference_zcur", 0.0)) + float(z_mid - np.median(self.ref["_pts"][:, 1]))
        return {"Rin": rin, "Rout": rout, "Rx": rx, "Zx": zx, "Ip": ip_t, "Zc": zc}

    def estimate(self, obs):
        x = (np.asarray(obs[:N_RT], dtype=float) - self.mu) / self.sd
        return np.hstack([x, 1.0]) @ self.W        # Rin, Rout, Rx, Zx, Zcur

    # ---- PCS construction -----------------------------------------------------------
    def build(self, sp, tmax=1.0):
        from freegsnke.control_loop.pcs import PlasmaControlSystem
        m = self.m
        n = len(CTRL_COILS)
        plasma_data = {"ip_ref": waveform([0.0, tmax], [sp["Ip"], sp["Ip"]]),
                       "vloop_ff": waveform([0.0, tmax], [0.0, 0.0]), "ip_blend": waveform([0.0, tmax], [1.0, 1.0]),
                       "k_prop": waveform([0.0], [m["ip_gains"]["k_prop"]]), "k_int": waveform([0.0], [m["ip_gains"]["k_int"]]),
                       "k_deriv": waveform([0.0], [0.0]), "M_solenoid": waveform([0.0], [1.0])}
        shape_data = {}
        for name in self.targets:
            shape_data[name] = {"ff": waveform([0.0, tmax], [0.0, 0.0]),
                                "ref": waveform([0.0, tmax], [sp[name], sp[name]]),
                                "blend": waveform([0.0, tmax], [0.0 if name == "Zx" else 1.0] * 2),
                                "k_prop": waveform([0.0], [m["gains"][name]]), "k_int": waveform([0.0], [0.0]),
                                "k_deriv": waveform([0.0], [0.0])}
        circuits = {t: {"times": np.array([0.0]), "vals": [np.asarray(v, dtype=float)]}
                    for t, v in self.ref["virtual_circuits"].items()}
        circuits["Zx"] = circuits.get("Zx", {"times": np.array([0.0]), "vals": [np.zeros(n)]})
        circuits["plasma"] = {"times": np.array([0.0]), "vals": [np.array([1.0] + [0.0] * (n - 1))]}
        circuits["coil_order"] = CTRL_COILS
        zeros = waveform([0.0, tmax], [0.0, 0.0])
        for c in CTRL_COILS:
            circuits[c + "_ref"] = zeros
        systems = {"min_coil_curr_lims": {"times": [0.0], "vals": [np.asarray(m["min_coil_curr"])]},
                   "max_coil_curr_lims": {"times": [0.0], "vals": [np.asarray(m["max_coil_curr"])]},
                   "max_coil_curr_ramp_lims": {"times": [0.0], "vals": [1e12 * np.ones(n)]}}
        for c in CTRL_COILS:
            systems[c + "_pert"] = zeros
        pf = {"R_matrix": {"times": [0.0], "vals": [np.asarray(m["coil_resist"])]},
              "M_FF_matrix": {"times": [0.0], "vals": [np.asarray(m["coil_self_ind"])]},
              "M_FB_matrix": {"times": [0.0], "vals": [np.asarray(m["coil_self_ind"])]},
              "coil_gains": {"times": [0.0], "vals": [0.015 * np.ones(n)]},
              "coil_voltage_lims": {"times": np.array([0.0]), "vals": [list(m["coil_voltage_lims"])]},
              "coil_voltage_slew_lims": {"times": [0.0], "vals": [1e12 * np.ones(n)]}}
        vertical = {"z_ref": waveform([0.0, tmax], [sp["Zc"], sp["Zc"]]),
                    "k_prop": waveform([0.0], [m["vertical_gains"]["k_prop"]]),
                    "k_deriv": waveform([0.0], [m["vertical_gains"]["k_deriv"]])}
        activation = {c + "_activation": waveform([0.0], [1.0]) for c in ACTIVE}
        with contextlib.redirect_stdout(io.StringIO()):
            self.pcs = PlasmaControlSystem(
                plasma_data=plasma_data, shape_data=shape_data, shape_control_mode="PID",
                circuits_data=circuits, systems_data=systems, pf_data=pf, vertical_data=vertical,
                coil_activation_data=activation, active_coils=ACTIVE, ctrl_coils=CTRL_COILS,
                vertical_coils=["P6"], ctrl_targets=self.targets, plasma_target=["plasma"])
        self.pcs_ref = self.ref

    def refresh_setpoints(self, sp):
        """Track time-varying targets by rewriting the constant reference waveforms."""
        pc = self.pcs
        pc.PlasmaController.data["ip_ref"]["vals"][:] = sp["Ip"]
        pc.PlasmaController.update_interpolants()
        for name in self.targets:
            pc.ShapeController.data[name]["ref"]["vals"][:] = sp[name]
        pc.ShapeController.update_interpolants()
        pc.VerticalController.data["z_ref"]["vals"][:] = sp["Zc"]
        pc.VerticalController.update_interpolants()

    # ---- episode ---------------------------------------------------------------------
    def reset(self, low, high):
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.t = 0.0
        n = len(CTRL_COILS)
        self.ip_hist = self.ip_err = 0.0
        self.T_err = np.zeros(len(self.targets)); self.T_hist = np.zeros(len(self.targets))
        self.I_prev = None; self.V_prev = np.zeros(n); self.z_prev = None; self.zipv = 0.0
        self.resist = np.asarray(self.m["active_coil_resist"], dtype=float)
        self.last_sp = None
        self.d_filt = None
        self.k = 0
        self.sp0 = None
        self.pcs = None          # rebuilt at the first act: no controller state leaks between episodes

    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        if self.k == 0:
            self.select_reference(obs)
        sp = self.setpoints(obs)
        d_raw = self.estimate(obs)
        if self.d_filt is None:
            self.d_filt = d_raw.copy()
        else:
            self.d_filt = self.d_filt + self.filter_alpha * (d_raw - self.d_filt)
        d = self.d_filt
        if self.ramp_steps > 0:   # ramp the shape/vertical set-points from the first estimate to the targets
            if self.sp0 is None:
                self.sp0 = {"Rin": d[0], "Rout": d[1], "Rx": d[2], "Zx": d[3], "Zc": d[4]}
            w = min(1.0, self.k / self.ramp_steps)
            for key, v0 in self.sp0.items():
                if np.isfinite(v0) and np.isfinite(sp[key]):
                    sp[key] = float(v0 + w * (sp[key] - v0))
        if self.pcs is None or self.pcs_ref is not self.ref:
            self.build(sp)
        elif self.last_sp is None or any(abs(sp[k] - self.last_sp[k]) > 1e-9 for k in ("Rin", "Rout", "Rx", "Zx", "Ip", "Zc")):
            self.refresh_setpoints(sp)
        self.last_sp = sp
        self.k += 1
        T_meas = np.array([d[0], d[1], d[2], d[3]])[: len(self.targets)]
        ip = float(obs[N_PROBE + N_COIL])
        z = float(d[4])
        I_meas = np.asarray(obs[N_PROBE: N_PROBE + len(CTRL_COILS)], dtype=float)
        if self.I_prev is None:
            self.I_prev = I_meas.copy()
        if self.z_prev is not None:
            self.zipv = ((z - self.z_prev) / DT) * ip
        with contextlib.redirect_stdout(io.StringIO()):
            V, self.ip_hist, self.ip_err, self.T_err, self.T_hist, I_appr, _ = self.pcs.calculate_ctrl_voltages(
                t=self.t, dt=DT, dt_simulator=DT, ip_meas=ip, ip_hist_prev=self.ip_hist, ip_err_prev=self.ip_err,
                T_meas=T_meas, T_err_prev=self.T_err, T_hist_prev=self.T_hist, I_approved_prev=self.I_prev,
                I_meas=I_meas, V_approved_prev=self.V_prev, zip_meas=z * ip, zipv_meas=self.zipv,
                active_coil_resists=self.resist.copy(), verbose=False)
        V = np.asarray(V, dtype=float).ravel()
        self.z_prev = z
        self.I_prev = np.asarray(I_appr, dtype=float).ravel()[: len(CTRL_COILS)]
        self.V_prev = V[: len(CTRL_COILS)]
        self.t += DT
        V = np.where(np.isfinite(V), V, 0.0)
        return np.clip(V, self.low, self.high)


def main():
    ctl = Controller()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("protocol") != PROTOCOL:
            raise SystemExit("unsupported protocol")
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            ctl.reset(req["action_low"], req["action_high"])
            reply["ready"] = True
        elif req["op"] == "act":
            reply["action"] = [float(v) for v in ctl.act(req["observation"])]
        elif req["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
