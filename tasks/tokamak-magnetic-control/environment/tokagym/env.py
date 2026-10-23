"""Gymnasium environment: coil voltages in, magnetic diagnostics out, FreeGSNKE in between."""

import contextlib
import copy
import io
import math
import os
import signal
import sys
import time

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # the verifier only needs the step semantics, not gymnasium registration
    gym = None
    spaces = None

from . import cases as C
from . import metrics as M
from .machine import VERTICAL_COIL, MachineSpec, load_machine

# Visible (training) randomization; the verifier owns a sealed superset in its own image.
# Data-grounded entries use FAIR-MAST double-null flat-tops (data/fair_mast/derived_ranges.json, CC BY-SA
# 4.0, UKAEA) mapped through data/fair_mast/profile_calibration.json: the visible plant spans their
# interquartile range, the sealed plant their 5-95% range.
VISIBLE_RANDOMIZATION = {
    # ln(beta_p) IQR +/-0.385 / (d ln beta_p / d ln paxis = 1.16); li_3 IQR +/-0.074 via alpha_m, alpha_n
    "profile": {"pressure": 0.33, "alpha": 0.19, "ip": 0.03},   # log-uniform half-widths / additive
    "coil_current_rel": 0.02,          # relative perturbation of the initial coil currents
    "plasma_resistivity_log10": 0.11,  # +/- decades around BASE_PLASMA_RESISTIVITY (R_p IQR)
    "passive_resistance_rel": 0.15,    # +/- relative scaling of every passive filament
    "probe_noise_rel": 0.01,           # gaussian, relative to per-signal reference scale (EFIT residual median)
    "probe_offset_rel": 0.025,         # per-episode constant offsets (median |EFIT residual mean| / 0.674)
    "current_noise_rel": 0.002,
    "supply_delay_steps": (0, 1),      # first-order actuator delay, uniform integer
    "handover_steps": (0, 10),         # feedforward-hold steps before the controller takes over
}
NO_RANDOMIZATION = {"profile": {"pressure": 0.0, "alpha": 0.0, "ip": 0.0}, "coil_current_rel": 0.0,
                    "plasma_resistivity_log10": 0.0, "passive_resistance_rel": 0.0,
                    "probe_noise_rel": 0.0, "probe_offset_rel": 0.0, "current_noise_rel": 0.0,
                    "supply_delay_steps": (0, 0)}

# Plant mechanisms beyond parameter scatter (tokagym-mastu/v0.2). Off unless a randomization sets
# them; the sealed verifier family switches them on (instruction.md names the categories, not the
# values). Every draw comes from a separate per-seed stream, so they do not shift the draws above.
STRUCTURAL_DEFAULTS = {
    # time-varying plasma: per-episode linear drifts and Poisson pressure crashes
    "pressure_drift_log_per_s": 0.0,       # drift rate of ln(axis pressure or beta_p), uniform +/-
    "alpha_drift_per_s": 0.0,              # drift rate of alpha_n (current-profile peaking), uniform +/-
    "resistivity_drift_log10_per_s": 0.0,  # drift rate of log10(plasma resistivity), uniform +/-
    "crash_rate_hz": 0.0,                  # Poisson rate of fast pressure crashes
    "crash_depth": (0.15, 0.31),           # crash depth range (fraction of the pressure), uniform
    "crash_recovery_s": 0.03,              # exponential recovery time after a crash
    # power supplies
    "supply_offset_rel": 0.0,              # per-episode constant offset per coil, fraction of the voltage limit
    "supply_slew_rel_per_step": 0.0,       # max change of the applied voltage per step / limit (0: none)
    "supply_stuck_current_rel": 0.0,       # below this |current| / current limit ...
    "supply_stuck_voltage_rel": 0.0,       # ... a request below this |voltage| / limit gives 0 V
    # sensors
    "dead_channel_fraction": 0.0,          # magnetic channels that read 0 for the whole episode
    "flux_loop_drift_rel_per_s": 0.0,      # integrator drift of each flux loop, uniform +/- per s
    # modelling error
    "passive_filament_rel": 0.0,           # per-filament log-normal spread of passive resistance
    "betap_constraint_prob": 0.0,          # probability of a beta_p- instead of a pressure-constrained profile
    # handover
    "handover_steps": (0, 0),              # feedforward-hold steps before the controller takes over, uniform
}

# Reproduces the median flat-top plasma resistance of MAST double-null shots at 550-850 kA
# (loop voltage / Ip = 1.73 uOhm, data/fair_mast/derived_ranges.json) in the base case, where
# 1e-6 Ohm m gives 4.51 uOhm and a 2.8 V loop voltage.
BASE_PLASMA_RESISTIVITY = 3.8e-7
PROFILE_UPDATE_REL = 2e-3   # smallest relative profile change sent to the solver
MAX_TRIAL_FAILURES = 40   # see patch_freegsnke
MIN_GROWTH_TIME_S = 2e-3   # see reset: four 0.5 ms control periods
MAX_CONTROLLABILITY_REDRAWS = 5
# Work budgets count FreeGSNKE residual evaluations (dynamic currents/flux and static Grad-Shafranov) per
# solve. Unlike a wall-clock budget they do not depend on the host, so whether a solve fails is a property
# of the plant, the actions and the seed. A normal step takes ~550 evaluations (p99 ~2,600, ~3 ms each).
STEP_WORK_LIMIT = 20000
HANDOVER_WORK_LIMIT = 20000   # a feedforward-hold step over budget ends the handover, as a failed one does
STATIC_WORK_LIMIT = 20000     # a static solve over budget is a non-converged draw (redrawn from the seed)
# Wall-clock budgets only guard against hangs that the work budget cannot see (e.g. compiled loops); an
# expired one is a host-side failure that the evaluator reruns, never a control outcome.
STEP_TIME_BUDGET_SEC = 1800.0
RESET_TIME_BUDGET_SEC = 1800.0
# Numerical fallback ladder for one control step. Level 0 is the normal nonlinear step. When a step raises,
# returns a non-finite state or exceeds its work budget, the evaluator replays the episode exactly up to that
# step (the simulator is deterministic) and retries it: level 1 allows more Newton-Krylov iterations and
# work (a step that converges within the level-0 limits gives the level-0 result), level 2 advances this one
# step with FreeGSNKE's linearised model (relative observation change ~5e-4 in tests) and the next step
# returns to the nonlinear solver. Wall-clock budgets are multiplied by ``budget_scale``.
STEP_NUMERICS = (
    {"max_solving_iterations": 50, "linear_only": False, "work_limit": STEP_WORK_LIMIT},
    {"max_solving_iterations": 150, "linear_only": False, "work_limit": 4 * STEP_WORK_LIMIT},
    {"max_solving_iterations": 50, "linear_only": True, "work_limit": STEP_WORK_LIMIT},
)
LOSS_AXIS_M = 0.25          # axis displacement from the reference that counts as plasma loss
LOSS_SHAPE_RMS_M = 0.20     # boundary RMS error that counts as plasma loss
REWARD_SCALES = {"alive": 1.0, "shape_m": 0.02, "xpoint_m": 0.05, "ip_rel": 0.05, "voltage": 0.02,
                 "coil_violation": 0.1, "termination": 20.0, "clip": 2.0}


class SolverFailure(RuntimeError):
    pass


class EquilibriumError(RuntimeError):
    """A static Grad-Shafranov solve did not converge or produced no plasma."""


class StepTimeout(BaseException):
    """The solver exceeded its wall-clock budget (degenerate plasma state).

    Derives from BaseException on purpose: FreeGSNKE's iteration loops contain broad
    ``except Exception`` handlers that would otherwise swallow the timeout.
    """


class ResetTimeout(RuntimeError):
    """Setting up the plant exceeded its wall-clock budget at every budget scale.

    Not an EquilibriumError: a slow solve is a property of the host, so it must not redraw the
    plant (which would make the draw depend on machine load). The evaluator reruns the episode.
    """


def patch_freegsnke():
    """Work around a FreeGSNKE 3.1.0 bug that ends episodes with "... object has no attribute 'ny'".

    ``Jtor_universal.copy`` copies ``nx`` but not ``ny``, so a copied profile fails as soon as it
    reaches ``diverted_critical_complete``, which reshapes with ``(self.nx, self.ny)``. Every
    profile class copies through it (``super().copy()``), so the wrapper adds the missing field.
    """
    from freegsnke import jtor_update
    original = jtor_update.Jtor_universal.copy
    if getattr(original, "_tokagym_ny_patch", False):
        return

    def copy(self, obj=None):
        out = original(self, obj)
        if hasattr(self, "ny") and not hasattr(out, "ny"):
            out.ny = self.ny
        return out

    copy._tokagym_ny_patch = True
    jtor_update.Jtor_universal.copy = copy

    # A degenerate profile (e.g. a beta_p constraint whose pressure integral is zero) turns psi
    # non-finite; FreeGS4E's numba-compiled X-point search then never returns, and a numba loop
    # cannot be interrupted by the SIGALRM budgets. Refuse non-finite psi before entering it, so the
    # failure surfaces as an ordinary Python exception (a rejected reset draw or a solver failure).
    from freegs4e import critical
    find = critical.find_critical
    if not getattr(find, "_tokagym_finite_patch", False):
        def find_critical(R, Z, psi, *args, **kwargs):
            if not np.all(np.isfinite(psi)):
                raise FloatingPointError("non-finite psi passed to the X-point search")
            return find(R, Z, psi, *args, **kwargs)
        find_critical._tokagym_finite_patch = True
        critical.find_critical = find_critical

    # FreeGSNKE's Newton-Krylov solver retries a failed trial update in `while: try ... except: shrink`
    # loops with a bare except. When every trial fails (a lost plasma, a degenerate beta_p profile) the
    # loop never ends, and the bare except also swallows our wall-clock StepTimeout. After 40
    # consecutive failures the residual comes back non-finite, so the solve ends as "did not converge"
    # and the caller handles it (reset redraws or skips the pre-roll; a step ends as a solver failure).
    from freegsnke import GSstaticsolver
    f_function = GSstaticsolver.NKGSsolver.F_function
    if not getattr(f_function, "_tokagym_retry_patch", False):
        def F_function(self, plasma_psi, *args, **kwargs):
            if _over_work_budget():
                return np.full_like(np.asarray(plasma_psi, dtype=float), np.nan)
            # (plasma_psi is the first parameter of NKGSsolver.F_function, by position or keyword)
            try:
                out = f_function(self, plasma_psi, *args, **kwargs)
            except BaseException:
                self._tokagym_failures = getattr(self, "_tokagym_failures", 0) + 1
                if self._tokagym_failures >= MAX_TRIAL_FAILURES:
                    self._tokagym_failures = 0
                    return np.full_like(np.asarray(plasma_psi, dtype=float), np.nan)
                raise
            self._tokagym_failures = 0
            return out
        F_function._tokagym_retry_patch = True
        GSstaticsolver.NKGSsolver.F_function = F_function

    # Count the evolutive solver's residual evaluations against the work budget. Over budget they return
    # NaN instead of raising: FreeGSNKE's Arnoldi step retries a raising residual in a bare-except loop,
    # whereas NaN residuals end the solve, which the caller then reports as over budget.
    from freegsnke import nonlinear_solve
    for name in ("F_function_curr", "F_function_psi"):
        fn = getattr(nonlinear_solve.nl_solver, name)
        if not getattr(fn, "_tokagym_work_patch", False):
            setattr(nonlinear_solve.nl_solver, name, _counted(fn))


_WORK = {"calls": 0, "limit": None, "exceeded": False}


def _over_work_budget():
    _WORK["calls"] += 1
    if _WORK["limit"] is not None and _WORK["calls"] > _WORK["limit"]:
        _WORK["exceeded"] = True
        return True
    return False


def _counted(fn):
    def wrapper(self, *args, **kwargs):   # FreeGSNKE passes the trial point by position or by keyword
        if _over_work_budget():
            x = args[0] if args else next(iter(kwargs.values()))
            return np.full_like(np.asarray(x, dtype=float), np.nan)
        return fn(self, *args, **kwargs)
    wrapper._tokagym_work_patch = True
    return wrapper


class WorkBudgetExceeded(RuntimeError):
    """A solve used more residual evaluations than its work budget (deterministic)."""


@contextlib.contextmanager
def work_budget(limit):
    """Count residual evaluations in the block; raise WorkBudgetExceeded at its end if over ``limit``."""
    _WORK.update(calls=0, limit=limit, exceeded=False)
    try:
        yield _WORK
    finally:
        exceeded, calls = _WORK["exceeded"], _WORK["calls"]
        _WORK.update(limit=None, exceeded=False)
    if exceeded:
        raise WorkBudgetExceeded(f"{calls} residual evaluations exceed the work budget of {limit}")


@contextlib.contextmanager
def wall_clock_budget(seconds):
    """Raise StepTimeout if the block runs longer than ``seconds`` (main thread of a process)."""
    def handler(signum, frame):
        raise StepTimeout(f"exceeded {seconds:.0f}s wall-clock budget")
    previous = signal.signal(signal.SIGALRM, handler)
    # Repeat every second after the deadline: FreeGSNKE has bare ``except:`` clauses that
    # can swallow one raise; a repeating timer escapes as soon as control leaves them.
    signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TokamakMagneticEnv(gym.Env if gym else object):
    """One frozen case of the MAST-U-like magnetic-control suite.

    Observation (flat float vector, in this order):
      flux loops (102), pickup coils (398), active coil currents (12), plasma current (1),
      target control points (32 x 2), target X-points (2 x 2, zero-padded, plus 2 validity
      flags), target plasma current (1), normalized time (1).
    Action: 12 coil voltages [V] within the machine limits.
    """

    metadata = {"render_modes": []}

    def __init__(self, case, spec=None, randomization="visible", solver="nonlinear",
                 horizon=None, strict_actions=False, verbose=False, relinearise_threshold=0.05):
        self.case = C.get_case(case) if isinstance(case, str) else case
        self.numerics_level = 0     # index into STEP_NUMERICS for the next step (set by the evaluator)
        self.budget_scale = 1.0     # multiplies every wall-clock budget (the evaluator raises it on reruns)
        self.on_progress = None     # optional callable, called after each setup stage (evaluator heartbeat)
        self.failure_kind = None    # "exception", "timeout" or "nonfinite" after a failed step, else None
        self._episode_seed = None
        self.spec = spec or MachineSpec()
        chosen = (VISIBLE_RANDOMIZATION if randomization == "visible"
                  else NO_RANDOMIZATION if randomization in (None, "none")
                  else dict(randomization))
        self.randomization = {**STRUCTURAL_DEFAULTS, **chosen}
        if solver not in ("nonlinear", "linear"):
            raise ValueError(solver)
        self.solver = solver
        # Linear mode re-linearizes when the toroidal current changes by this relative amount
        # (FreeGSNKE's piecewise-linear "PwLD" mode); ignored by the nonlinear solver.
        self.relinearise_threshold = relinearise_threshold
        self.horizon = horizon or self.case.horizon
        self.strict_actions = strict_actions
        self.verbose = verbose
        patch_freegsnke()
        self.tokamak = load_machine(self.spec)
        self.n_active = self.spec.n_active
        self.vmax = self.spec.voltage_limits
        self.ilim = self.spec.current_limits
        self._build_reference()
        n_obs = (len(self.tokamak.probes.floop_order) + len(self.tokamak.probes.pickup_order)
                 + self.n_active + 1 + 2 * C.N_CONTROL_POINTS + 4 + 2 + 1 + 1)
        if spaces:
            self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float64)
            self.action_space = spaces.Box(-self.vmax, self.vmax, dtype=np.float64)
        self.n_obs = n_obs
        self._rng = np.random.default_rng(0)
        self.stepping = None

    # ------------------------------------------------------------------ setup
    def _new_equilibrium(self):
        from freegsnke import equilibrium_update
        g = self.spec.grid
        return equilibrium_update.Equilibrium(tokamak=self.tokamak, Rmin=g.Rmin, Rmax=g.Rmax,
                                              Zmin=g.Zmin, Zmax=g.Zmax, nx=g.nx, ny=g.ny)

    def _solve_static(self, eq, profiles, currents, max_iterations=60, tolerance=1e-8):
        """Forward solve in place; raise EquilibriumError on non-convergence or no plasma."""
        from freegsnke import GSstaticsolver
        for name, value in currents.items():
            eq.tokamak.set_coil_current(coil_label=name, current_value=value)
        solver = GSstaticsolver.NKGSsolver(eq)
        sink = io.StringIO()
        try:
            with wall_clock_budget(RESET_TIME_BUDGET_SEC * self.budget_scale), contextlib.redirect_stdout(sink), \
                    work_budget(STATIC_WORK_LIMIT):
                solver.solve(eq=eq, profiles=profiles, constrain=None, target_relative_tolerance=tolerance,
                             max_solving_iterations=max_iterations, verbose=False)
        except StepTimeout as error:
            raise ResetTimeout(f"static solve timed out: {error}") from error
        except WorkBudgetExceeded as error:
            raise EquilibriumError(f"static solve: {error}") from error
        text = sink.getvalue()
        if self.verbose:
            sys.stdout.write(text)
        if "DID NOT CONVERGE" in text or "SUCCESS" not in text:
            raise EquilibriumError("static solve did not converge: " + text.strip().splitlines()[-1][:160]
                                   if text.strip() else "static solve produced no report")
        if np.asarray(eq.opt).size == 0:
            raise EquilibriumError("static solve has no magnetic axis")
        return solver

    def _construct(self, eq, profiles, currents, steps=None, params=None):
        """Apply continuation steps (default: the case's construction) from a converged equilibrium."""
        params = dict(C.BASE_PROFILE if params is None else params)
        for step in (self.case.construction if steps is None else steps):
            kind = step[0]
            if kind == "currents_scale":
                for coil, f in step[1].items():
                    currents[coil] = currents[coil] * f
                eq = eq.create_auxiliary_equilibrium()
                self._solve_static(eq, profiles, currents)
            elif kind == "scale_all":
                # Similarity scaling: every coil current and Ip by s, axis pressure by s^2.
                s_ = float(step[1])
                currents = {c: v * s_ for c, v in currents.items()}
                params["Ip"] = params["Ip"] * s_
                params["paxis"] = params["paxis"] * s_ ** 2
                profiles = C.make_profiles(self.case, eq, params)
                eq = eq.create_auxiliary_equilibrium()
                self._solve_static(eq, profiles, currents)
            elif kind == "currents_file":
                target = C.load_currents_file(step[1], self.spec.data_dir)
                start = dict(currents)
                for k in range(1, step[2] + 1):
                    w = k / step[2]
                    currents = {c: (1 - w) * start[c] + w * target[c] for c in start}
                    eq = eq.create_auxiliary_equilibrium()
                    self._solve_static(eq, profiles, currents)
            elif kind == "profile":
                params.update(step[1])
                profiles = C.make_profiles(self.case, eq, params)
                eq = eq.create_auxiliary_equilibrium()
                self._solve_static(eq, profiles, currents)
            else:
                raise ValueError(step)
        target = dict(self.case.profile_params)
        same = all(np.allclose(params[k], target[k]) if isinstance(target[k], (int, float, list))
                   else params[k] == target[k] for k in target)
        if not same:
            profiles = C.make_profiles(self.case, eq, target)
            eq = eq.create_auxiliary_equilibrium()
            self._solve_static(eq, profiles, currents)
        return eq, profiles, currents

    def _build_reference(self):
        """Nominal equilibrium (by continuation from the base) -> frozen targets and probe scales."""
        eq = self._new_equilibrium()
        base_profiles = C.make_profiles(self.case, eq, C.BASE_PROFILE)
        currents = C.load_currents_file(C.BASE_CURRENTS_FILE, self.spec.data_dir)
        self._solve_static(eq, base_profiles, currents)
        eq, profiles, currents = self._construct(eq, base_profiles, currents)
        eq._updatePlasmaPsi(eq.plasma_psi)
        self.reference_currents = dict(currents)
        self.tokamak.probes.initialise_setup(eq)
        self.reference_ip = float(profiles.Ip)
        self.reference_axis = np.asarray(eq.magneticAxis()[:2], dtype=float)
        # Target corners: [shape][shift]. Shape 1 is the case's second shape (waveforms.shape_mix);
        # shift 1 is the same shape held by the vertical coil at the case's largest vertical offset
        # (waveforms.z_shift). Targets at time t blend the corners bilinearly, so every target is a
        # blend of equilibria this machine can hold (a rigid shift of a double null is not one: the
        # shaping coils are up-down symmetric series pairs and only P6 moves the plasma vertically).
        self.z_shift_max = float(max(self.case.waveforms.z_shift, key=abs))
        corners = {(0, 0): (eq, profiles, dict(currents))}
        if self.case.second_shape:
            eq2, prof2, cur2 = self._construct(eq.create_auxiliary_equilibrium(), profiles, dict(currents),
                                               steps=self.case.second_shape, params=self.case.profile_params)
            corners[(1, 0)] = (eq2, prof2, cur2)
        if self.z_shift_max:
            for shape in [k[0] for k in list(corners)]:
                e0, p0, c0 = corners[(shape, 0)]
                corners[(shape, 1)] = self._shifted(e0, p0, c0, self.z_shift_max)
        for (e_, _, _) in corners.values():
            e_._updatePlasmaPsi(e_.plasma_psi)
        self.corner_targets = {k: self._shape_targets(v[0]) for k, v in corners.items()}
        counts = {len(v["xpoints"]) for v in self.corner_targets.values()}
        if len(counts) != 1:
            raise RuntimeError(f"{self.case.name}: the target equilibria have different X-point counts")
        for name, value in currents.items():   # the corner solves changed the shared machine's currents
            self.tokamak.set_coil_current(coil_label=name, current_value=value)
        base = self.corner_targets[(0, 0)]
        self.target_points, self.target_xpoints = base["points"], base["xpoints"]
        self.reference_gaps, self.reference_zx = base["gaps"], base["zx"]
        self.second_targets = (1, 0) in corners
        floops = np.asarray(self.tokamak.probes.calculate_fluxloop_value(eq), dtype=float)
        pickups = np.asarray(self.tokamak.probes.calculate_pickup_value(eq), dtype=float)
        ref = np.concatenate([floops, pickups])
        self.probe_scale = np.maximum(np.abs(ref), 0.1 * np.median(np.abs(ref)) + 1e-9)
        self.reference_eq = eq
        self.reference_profiles = profiles
        self.reference_limited = bool(eq._profiles.flag_limiter)

    def _shifted(self, eq, profiles, currents, dz, tol=2e-4, max_iter=10):
        """The same plasma held ``dz`` higher by the vertical coil alone (secant on its current)."""
        z0 = float(eq.Zcurrent())
        cur = dict(currents)
        i0, z_prev = float(cur.get(VERTICAL_COIL, 0.0)), z0
        i1 = i0 - dz / 0.115 * 1000.0        # about -1 kA per +11.5 cm on the base plasma
        for _ in range(max_iter):
            cur[VERTICAL_COIL] = i1
            out = eq.create_auxiliary_equilibrium()
            prof = copy.deepcopy(profiles)
            self._solve_static(out, prof, cur)
            z1 = float(out.Zcurrent())
            if abs(z1 - (z0 + dz)) < tol:
                return out, prof, cur
            i0, i1 = i1, i1 + (z0 + dz - z1) * (i1 - i0) / (z1 - z_prev)
            z_prev = z1
        raise RuntimeError(f"{self.case.name}: could not hold the plasma {dz:+.3f} m higher with {VERTICAL_COIL}")

    def _shape_targets(self, eq):
        """Control points, X-points (lowest first), gap radii, lower X-point height, gap height and
        current-centroid height of an equilibrium.

        The gaps are how MAST-U controls the shape (Lvovskiy et al. 2025): inner/outer separatrix
        radii at the axis height and the height of the lower X-point. A diverted target keeps the
        X-point nearest the boundary flux below and above the axis (|psi_N - 1| <= the candidate
        tolerance), so a shifted double null keeps its lower X-point as a target.
        """
        points = C.control_points(eq)
        z_axis = float(np.asarray(eq.opt)[0, 1])
        xpoints = np.zeros((0, 2))
        if self.case.diverted:
            near = C.boundary_xpoints(eq)
            pick = [near[near[:, 1] < z_axis][:1], near[near[:, 1] > z_axis][:1]]
            xpoints = np.concatenate([p for p in pick if len(p)]) if any(len(p) for p in pick) else np.zeros((0, 2))
            if len(xpoints) == 0:
                raise RuntimeError(f"{self.case.name}: target equilibrium has no boundary X-point")
            xpoints = xpoints[np.argsort(xpoints[:, 1])]
        gaps = np.asarray(self._gaps(eq, z_axis), dtype=float)
        if not np.all(np.isfinite(gaps)):
            raise RuntimeError(f"{self.case.name}: target equilibrium has no inner/outer separatrix")
        zx = float(xpoints[0, 1]) if len(xpoints) else None
        return {"points": points, "xpoints": xpoints, "gaps": gaps, "zx": zx, "gap_z": z_axis,
                "zcur": float(eq.Zcurrent())}

    # ------------------------------------------------------------------ helpers
    def _blend(self, t):
        """Target quantities at time t: bilinear blend of the corner equilibria (shape mix x shift)."""
        wf = self.case.waveforms
        w = wf.shape_weight(t) if self.second_targets else 0.0
        s = wf.z_offset(t) / self.z_shift_max if self.z_shift_max else 0.0
        weights = {(0, 0): (1 - w) * (1 - s), (1, 0): w * (1 - s), (0, 1): (1 - w) * s, (1, 1): w * s}
        out = {}
        for key in ("points", "xpoints", "gaps", "gap_z", "zcur", "zx"):
            acc = None
            for corner, c in weights.items():
                if c == 0.0 or corner not in self.corner_targets:
                    continue
                v = self.corner_targets[corner][key]
                if v is None:
                    acc = None
                    break
                acc = c * np.asarray(v, dtype=float) if acc is None else acc + c * np.asarray(v, dtype=float)
            out[key] = acc
        out["zx"] = None if out["zx"] is None else float(out["zx"])
        return out

    def targets_at(self, t):
        b = self._blend(t)
        ip = self.reference_ip * self.case.waveforms.ip_factor(t)
        return b["points"], b["xpoints"], ip

    def gap_targets_at(self, t):
        """Inner/outer gap radii and lower X-point height targets at time t (zx None if limited)."""
        b = self._blend(t)
        return b["gaps"], b["zx"]

    def gap_height_at(self, t):
        """Height at which the inner/outer gaps are measured at time t (the target axis height)."""
        return float(self._blend(t)["gap_z"])

    def vertical_target_at(self, t):
        """Current-centroid height of the target equilibrium at time t (for reference controllers)."""
        return float(self._blend(t)["zcur"])

    def _gaps(self, eq, z):
        """Inner and outer separatrix radii at height ``z``, NaN if absent."""
        try:
            rin, rout = eq.innerOuterSeparatrix(Z=float(z))
            return float(rin), float(rout)
        except Exception:  # noqa: BLE001
            return math.nan, math.nan

    def steady_state_start(self):
        """First record index of the steady-state window (tokamak-shape-metrics/v2)."""
        t_end = self.horizon * C.DT
        start = min(self.case.last_change_time() + M.SETTLE_S, t_end - M.MIN_WINDOW_S)
        return max(0, int(round(start / C.DT)))

    def _coil_currents(self):
        return np.asarray(self.stepping.vessel_currents_vec[: self.n_active], dtype=float)

    def _plasma_current(self):
        return float(self.stepping.currents_vec[-1] * self.stepping.plasma_norm_factor)

    def _observe(self):
        eq = self.stepping.eq1
        floops = np.asarray(self.tokamak.probes.calculate_fluxloop_value(eq), dtype=float)
        pickups = np.asarray(self.tokamak.probes.calculate_pickup_value(eq), dtype=float)
        probes = np.concatenate([floops, pickups])
        probes = (probes + self._probe_offset + self._flux_drift * self.t
                  + self._rng.normal(0.0, 1.0, probes.shape) * self._probe_noise)
        probes[self._dead] = 0.0
        currents = self._coil_currents()
        currents = currents + self._rng.normal(0.0, 1.0, currents.shape) * (
            self.randomization["current_noise_rel"] * np.abs(currents).max())
        ip = self._plasma_current()
        pts, xpts, ip_t = self.targets_at(self.t)
        xp = np.zeros((2, 2)); flags = np.zeros(2)
        xp[: len(xpts)] = xpts; flags[: len(xpts)] = 1.0
        obs = np.concatenate([probes, currents, [ip], pts.ravel(), xp.ravel(), flags, [ip_t],
                              [self.t / (self.horizon * C.DT)]])
        return obs

    def _descriptors(self):
        eq = self.stepping.eq1
        sep = C.boundary_contour(eq)
        pts, xpts_t, ip_t = self.targets_at(self.t)
        ip = self._plasma_current()
        actual_x = C.candidate_xpoints(eq) if len(xpts_t) else np.zeros((0, 2))
        currents = self._coil_currents()
        violated = [c for c, i, (lo, hi) in zip(self.spec.active_coils, currents, self.ilim) if i < lo or i > hi]
        axis = eq.magneticAxis()
        rin, rout = self._gaps(eq, self.gap_height_at(self.t))
        lower = np.zeros(2) + np.nan
        if len(actual_x):
            # the candidate nearest to the target lower X-point: with no boundary X-point the
            # candidates include far-away saddle points, and the lowest of them is meaningless
            ref = xpts_t[0] if len(xpts_t) else actual_x[np.argmin(actual_x[:, 1])]
            lower = actual_x[np.argmin(np.linalg.norm(actual_x - ref, axis=1))]
        try:
            zcur = float(eq.Zcurrent())
        except Exception:  # noqa: BLE001
            zcur = math.nan

        def err(x):  # absolute error; a missing quantity counts as an infinite error
            return float(abs(x)) if math.isfinite(x) else math.inf

        gaps_t, zx_t = self.gap_targets_at(self.t)
        return {
            "Rin": float(rin), "Rout": float(rout), "Rx": float(lower[0]), "Zx": float(lower[1]),
            "Zcur": zcur,
            "shape_rms": M.shape_rms(sep, pts) if len(sep) > 2 else math.inf,
            "xpoint_error": M.xpoint_error(actual_x, xpts_t),
            "ip_error_rel": abs(ip - ip_t) / max(abs(ip_t), 1.0),
            "rin_error": err(rin - gaps_t[0]),
            "rout_error": err(rout - gaps_t[1]),
            "zx_error": min(err(lower[1] - zx_t), M.XPOINT_CAP_M) if zx_t is not None else 0.0,
            "coil_violation": float(bool(violated)),
            "violated_coils": violated,
            "ip": ip, "ip_target": ip_t,
            "R_axis": float(axis[0]), "Z_axis": float(axis[1]),
            "n_separatrix": int(len(sep)),
        }

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        """Draw a plant; redraw (deterministically from the seed) while it is not controllable.

        A draw whose fastest vertical-instability timescale is below MIN_GROWTH_TIME_S (four control
        periods; the supply adds up to two steps of delay) cannot be held by any controller - the
        privileged PID loses such plasmas within a few steps - and no programme would operate it.
        """
        self._episode_seed = seed
        for k in range(MAX_CONTROLLABILITY_REDRAWS + 1):
            sub = seed if (k == 0 or seed is None) else int(
                np.random.SeedSequence([int(seed), 104729, k]).generate_state(1)[0])
            obs, info = self._reset_once(sub, options)
            tau = info["growth_rate"]
            if tau is None or tau >= MIN_GROWTH_TIME_S:
                break
        info["controllability_redraws"] = k
        return obs, info

    def _reset_once(self, seed=None, options=None):
        if gym:
            super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        rz = self.randomization
        t0 = time.monotonic()
        attempts = 0
        fallback = False
        while True:
            attempts += 1
            if attempts <= 3:
                params = C.perturb_profile_params(self.case, rng, rz["profile"])
                currents = {k: v * (1.0 + rng.uniform(-rz["coil_current_rel"], rz["coil_current_rel"]))
                            for k, v in self.reference_currents.items()}
            else:  # last resort: nominal profile and currents, other draws unchanged
                params, currents, fallback = dict(self.case.profile_params), dict(self.reference_currents), True
            eq = self.reference_eq.create_auxiliary_equilibrium()
            profiles = C.make_profiles(self.case, eq, params)
            try:
                solver = self._solve_static(eq, profiles, currents)
                break
            except EquilibriumError as error:
                if attempts >= 4:
                    raise
                if self.verbose:
                    print(f"reset draw rejected ({error}); redrawing", file=sys.stderr)
        # structural mechanisms draw from their own stream so they do not shift the draws above
        srng = np.random.default_rng(None if seed is None else [int(seed), 7919])
        constraint = "pressure"
        if self.case.profile == "paxis" and srng.uniform() < rz["betap_constraint_prob"]:
            # same equilibrium, but beta_p rather than axis pressure is held during the evolution
            from freegsnke.jtor_update import ConstrainBetapIp
            betap = float(eq.poloidalBeta1())
            beq = self.reference_eq.create_auxiliary_equilibrium()
            bprofiles = ConstrainBetapIp(eq=beq, betap=betap, Ip=params["Ip"], fvac=params["fvac"],
                                         alpha_m=params["alpha_m"], alpha_n=params["alpha_n"])
            try:
                solver = self._solve_static(beq, bprofiles, currents)
                eq, profiles, constraint = beq, bprofiles, "betap"
                params = dict(params, betap=betap)
            except EquilibriumError:
                pass   # keep the pressure-constrained draw
        eq._updatePlasmaPsi(eq.plasma_psi)
        resistivity = BASE_PLASMA_RESISTIVITY * 10 ** rng.uniform(
            -rz["plasma_resistivity_log10"], rz["plasma_resistivity_log10"])
        passive_scale = 1.0 + rng.uniform(-rz["passive_resistance_rel"], rz["passive_resistance_rel"])
        coil_resist = np.array(self.tokamak.coil_resist, dtype=float).copy()
        n_passive = len(coil_resist) - self.n_active
        coil_resist[self.n_active:] *= passive_scale * np.exp(
            srng.normal(0.0, 1.0, n_passive) * rz["passive_filament_rel"])
        from freegsnke import nonlinear_solve
        sink = io.StringIO()

        def init_stepping():
            try:
                with wall_clock_budget(RESET_TIME_BUDGET_SEC * self.budget_scale), \
                        contextlib.redirect_stdout(sys.stdout if self.verbose else sink):
                    self.stepping = nonlinear_solve.nl_solver(
                        eq=eq, profiles=profiles, GSStaticSolver=solver, full_timestep=C.DT,
                        plasma_resistivity=resistivity, custom_coil_resist=coil_resist,
                        max_mode_frequency=10 ** 2.5, n_linearization_workers=1)
                    self.stepping.initialize_from_ICs(eq, profiles)
            except StepTimeout as error:
                raise ResetTimeout(f"stepper initialisation timed out: {error}") from error

        # The pre-roll steps mutate eq and profiles in place; keep the converged initial state so a
        # failed pre-roll restarts from it, not from the half-stepped objects of the failed solve.
        initial = copy.deepcopy((eq, profiles))
        if self.on_progress:
            self.on_progress()
        init_stepping()
        self.growth_rate = float(np.max(np.real(self.stepping.linearised_sol.all_timescales))) \
            if hasattr(self.stepping, "linearised_sol") else None
        # Handover: the plasma first evolves under a feedforward hold (V = R I) for a random number
        # of steps, so the controller takes over a moving plasma with eddy currents, not an equilibrium.
        lo, hi = rz["handover_steps"]
        handover = int(srng.integers(lo, hi + 1)) if hi > 0 else 0
        hold = self._coil_currents() * coil_resist[: self.n_active]
        for k in range(handover):
            if self.on_progress:
                self.on_progress()
            try:
                with wall_clock_budget(STEP_TIME_BUDGET_SEC * self.budget_scale), contextlib.redirect_stdout(sink), \
                        work_budget(HANDOVER_WORK_LIMIT):
                    self.stepping.nlstepper(active_voltage_vec=hold, verbose=False, max_solving_iterations=50)
                if not np.all(np.isfinite(self.stepping.trial_currents)):
                    raise SolverFailure("non-finite state after a handover step")
            except StepTimeout as error:   # a hang or an overloaded host: the evaluator reruns the episode
                raise ResetTimeout(f"handover step timed out: {error}") from error
            except Exception:   # numerical failure or work budget: never charge the controller for the pre-roll
                eq, profiles = copy.deepcopy(initial)
                init_stepping()
                handover = 0
                break
        self._profile_params = params
        self._constraint = constraint
        self._resistivity = resistivity
        self._probe_noise = rz["probe_noise_rel"] * self.probe_scale
        self._probe_offset = rng.normal(0.0, 1.0, self.probe_scale.shape) * rz["probe_offset_rel"] * self.probe_scale
        lo, hi = rz["supply_delay_steps"]
        self._delay = int(rng.integers(lo, hi + 1))
        self._pending = []
        self._disturbed = False
        # time-varying plasma
        t_end = self.horizon * C.DT
        self._drift = {k: float(srng.uniform(-1.0, 1.0) * rz[key]) for k, key in
                       (("log_p", "pressure_drift_log_per_s"), ("alpha_n", "alpha_drift_per_s"),
                        ("log10_eta", "resistivity_drift_log10_per_s"))}
        n_crash = int(srng.poisson(rz["crash_rate_hz"] * t_end)) if rz["crash_rate_hz"] > 0 else 0
        self._crashes = sorted(zip(srng.uniform(0.0, t_end, n_crash).tolist(),
                                   srng.uniform(*rz["crash_depth"], n_crash).tolist()))
        self._time_varying = bool(self._drift["log_p"] or self._drift["alpha_n"] or self._crashes)
        key = "betap" if constraint == "betap" else "paxis"
        self._applied_profile = ({key: params[key], "alpha_m": params["alpha_m"], "alpha_n": params["alpha_n"]}
                                 if self.case.profile == "paxis" else None)
        # power supplies
        self._supply_offset = srng.uniform(-1.0, 1.0, self.n_active) * rz["supply_offset_rel"] * self.vmax
        self._last_applied = np.zeros(self.n_active)
        # sensors
        n_floop = len(self.tokamak.probes.floop_order)
        self._dead = srng.random(self.probe_scale.shape) < rz["dead_channel_fraction"]
        self._flux_drift = np.zeros(self.probe_scale.shape)
        self._flux_drift[:n_floop] = (srng.uniform(-1.0, 1.0, n_floop) * rz["flux_loop_drift_rel_per_s"]
                                      * self.probe_scale[:n_floop])
        self.t = 0.0
        self.k = 0
        self.records = []
        self.terminated_reason = None
        self.failure_void = False
        self.failure_kind = None
        self._last_obs = self._observe()
        info = {"setup_time_sec": time.monotonic() - t0, "profile_params": params,
                "plasma_resistivity": resistivity, "passive_scale": passive_scale,
                "supply_delay_steps": self._delay, "growth_rate": self.growth_rate,
                "reset_attempts": attempts, "reset_fallback": fallback,
                "profile_constraint": constraint, "drift": dict(self._drift),
                "crashes": [[round(t, 4), round(d, 3)] for t, d in self._crashes],
                "dead_channels": int(self._dead.sum()), "handover_steps": handover}
        return self._last_obs, info

    def _profile_update(self):
        """Profile parameters for the coming step: drifts, pressure crashes and the case disturbance.

        Returns None when nothing changes, so a static plasma costs no profile updates.
        """
        d = self.case.disturbance
        disturbed = d.time is not None and self.t >= d.time - 1e-9
        newly = disturbed and not self._disturbed
        self._disturbed = self._disturbed or disturbed
        if not (self._time_varying or newly):
            return None
        p = self._profile_params
        if self.case.profile != "paxis":   # Lao85 (not in the v0.2 suite): the case disturbance only
            return ({"alpha": [a * d.alpha_scale * d.paxis_scale for a in p["alpha"]], "beta": p["beta"]}
                    if newly else None)
        scale = math.exp(self._drift["log_p"] * self.t) * (d.paxis_scale if disturbed else 1.0)
        for t_crash, depth in self._crashes:
            if self.t >= t_crash:
                scale *= 1.0 - depth * math.exp(-(self.t - t_crash) / self.randomization["crash_recovery_s"])
        key = "betap" if self._constraint == "betap" else "paxis"
        target = {key: p[key] * scale, "alpha_m": p["alpha_m"],
                  "alpha_n": max(0.2, p["alpha_n"] + self._drift["alpha_n"] * self.t)}
        # A profile change nearly doubles the cost of a step, so slow drifts are applied in
        # steps of PROFILE_UPDATE_REL (piecewise constant); crashes exceed it at once.
        last = self._applied_profile
        if (not newly and abs(target[key] / last[key] - 1.0) < PROFILE_UPDATE_REL
                and abs(target["alpha_n"] - last["alpha_n"]) < PROFILE_UPDATE_REL):
            return None
        self._applied_profile = target
        return target

    def _supply(self, requested):
        """Power-supply response to the (delayed) request: offset, slew limit, stuck-at-zero, limits.

        The vertical-control coil is exempt from sticking (Degrave et al. saw it on thyristor PF supplies).
        """
        rz = self.randomization
        out = requested + self._supply_offset
        if rz["supply_slew_rel_per_step"] > 0:
            step = rz["supply_slew_rel_per_step"] * self.vmax
            out = self._last_applied + np.clip(out - self._last_applied, -step, step)
        if rz["supply_stuck_current_rel"] > 0:
            imax = np.max(np.abs(self.ilim), axis=1)
            stuck = ((np.abs(self._coil_currents()) < rz["supply_stuck_current_rel"] * imax)
                     & (np.abs(out) < rz["supply_stuck_voltage_rel"] * self.vmax))
            stuck[list(self.spec.active_coils).index(VERTICAL_COIL)] = False
            out = np.where(stuck, 0.0, out)
        out = np.clip(out, -self.vmax, self.vmax)
        self._last_applied = out
        return out

    def step(self, action):
        if self.stepping is None:
            raise RuntimeError("reset() first")
        v = np.asarray(action, dtype=float).ravel()
        if v.shape != (self.n_active,) or not np.all(np.isfinite(v)):
            raise ValueError("Action must be a finite vector of coil voltages")
        if np.any(np.abs(v) > self.vmax * (1 + 1e-9)):
            if self.strict_actions:
                raise ValueError("Voltage outside machine limits")
            v = np.clip(v, -self.vmax, self.vmax)
        # First-order power-supply delay implemented as a FIFO of requested voltages.
        self._pending.append(v.copy())
        requested = self._pending.pop(0) if len(self._pending) > self._delay else np.zeros_like(v)
        applied = self._supply(requested)
        profile_update = self._profile_update()
        resistivity = (self._resistivity * 10 ** (self._drift["log10_eta"] * self.t)
                       if self._drift["log10_eta"] else None)
        numerics = STEP_NUMERICS[self.numerics_level]
        self.failure_kind = None
        t0 = time.monotonic()
        try:
            with wall_clock_budget(STEP_TIME_BUDGET_SEC * self.budget_scale), \
                    contextlib.redirect_stdout(sys.stdout if self.verbose else io.StringIO()), \
                    work_budget(numerics["work_limit"]):
                self._raise_test_fault()
                self.stepping.nlstepper(active_voltage_vec=applied, profiles_parameters=profile_update,
                                        plasma_resistivity=resistivity,
                                        linear_only=(self.solver == "linear" or numerics["linear_only"]),
                                        verbose=False,
                                        max_solving_iterations=numerics["max_solving_iterations"],
                                        relinearise_threshold=(self.relinearise_threshold
                                                               if self.solver == "linear" else None))
        except (Exception, StepTimeout) as error:  # numerical errors or the wall-clock budget
            self.failure_kind = ("timeout" if isinstance(error, StepTimeout)
                                 else "work" if isinstance(error, WorkBudgetExceeded) else "exception")
            self.terminated_reason = f"solver_failure: {type(error).__name__}: {error}"[:200]
            # Reported only: was the plasma within tolerance when the solver failed?
            self.failure_void = M.void_candidate(self.records)
            rec = {"shape_rms": math.inf, "xpoint_error": math.inf, "ip_error_rel": math.inf,
                   "rin_error": math.inf, "rout_error": math.inf, "zx_error": math.inf,
                   "coil_violation": 1.0, "step_time_sec": time.monotonic() - t0,
                   "numerics_level": self.numerics_level}
            self.records.append(rec)
            return self._last_obs, -REWARD_SCALES["termination"], True, False, self._info(rec, applied)
        self.t += self.stepping.dt_step
        self.k += 1
        desc = self._descriptors()
        desc["step_time_sec"] = time.monotonic() - t0
        desc["numerics_level"] = self.numerics_level
        self.records.append(desc)
        terminated = False
        # Plasma-lost criteria (physical, hardware-independent): no closed boundary, current
        # collapse, or the axis/boundary far outside the reference envelope.
        if (not math.isfinite(desc["shape_rms"]) or desc["ip"] < 0.2 * desc["ip_target"]
                or abs(desc["Z_axis"] - self.reference_axis[1]) > LOSS_AXIS_M
                or abs(desc["R_axis"] - self.reference_axis[0]) > LOSS_AXIS_M
                or desc["shape_rms"] > LOSS_SHAPE_RMS_M):
            terminated = True
            self.terminated_reason = "plasma_lost"
        elif desc["violated_coils"]:
            # Operational limits end the episode, as machine protection would (Degrave et al. 2022).
            terminated = True
            self.terminated_reason = "coil_limit: " + ",".join(desc["violated_coils"])
        obs = self._observe()
        if not np.all(np.isfinite(obs)):
            terminated = True
            self.terminated_reason = "solver_failure: nonfinite observation"
            self.failure_kind = "nonfinite"
            self.failure_void = M.void_candidate(self.records[:-1])
            obs = self._last_obs
        truncated = self.k >= self.horizon and not terminated
        reward = self._reward(desc, v, terminated)
        self._last_obs = obs
        return self._last_obs, reward, terminated, truncated, self._info(desc, applied)

    def _raise_test_fault(self):
        """Testing hook: TOKAGYM_TEST_FAULTS="case:seed:step:kind:max_level[,...]" makes the step that
        follows ``step`` completed steps of that episode fail ("exception" or "timeout") at numerics
        levels up to ``max_level``. Unset in every scored run."""
        spec = os.environ.get("TOKAGYM_TEST_FAULTS")
        if not spec:
            return
        for item in spec.split(","):
            case, seed, step, kind, max_level = item.split(":")
            if (case == self.case.name and self._episode_seed is not None and int(seed) == int(self._episode_seed)
                    and int(step) == self.k and self.numerics_level <= int(max_level)):
                if kind == "timeout":
                    raise StepTimeout("injected test timeout")
                raise SolverFailure("injected test failure")

    def _reward(self, d, v, terminated):
        """Training signal only (not a benchmark score): alive bonus minus clipped errors.

        Positive when the plasma is near its targets, so that surviving the horizon
        dominates early termination; the termination penalty is applied once.
        """
        s = REWARD_SCALES
        clip = s["clip"]

        def term(x, scale):  # non-finite errors count as the clipped maximum
            return clip if not math.isfinite(x) else min(x / scale, clip)

        r = s["alive"] \
            - 0.5 * term(d["shape_rms"], s["shape_m"]) \
            - 0.25 * term(d["xpoint_error"], s["xpoint_m"]) \
            - 0.25 * term(d["ip_error_rel"], s["ip_rel"]) \
            - s["coil_violation"] * d["coil_violation"] \
            - s["voltage"] * float(np.mean((v / self.vmax) ** 2))
        if terminated:
            r -= s["termination"]
        return float(r)

    def _info(self, desc, applied):
        info = {k: v for k, v in desc.items()}
        info["applied_voltage_max_abs"] = float(np.abs(applied).max())
        info["t"] = self.t
        info["termination"] = self.terminated_reason
        return info

    def episode_summary(self):
        s = M.summarize_episode(self.records, self.horizon, self.steady_state_start())
        s["termination"] = self.terminated_reason
        s["feasible"] = s["complete"] and self.terminated_reason is None
        reason = self.terminated_reason or ""
        if s["feasible"]:
            s["outcome"] = "success" if M.meets_success(s) else "kept"
        elif reason.startswith("solver_failure"):
            # Every episode counts (fixed denominator); the evaluator has already retried the step.
            s["outcome"] = "solver_failure"
            s["within_tolerance_at_failure"] = bool(self.failure_void)
        elif reason.startswith("coil_limit"):
            s["outcome"] = "coil_limit"
        else:
            s["outcome"] = "plasma_lost"
        s["success"] = s["outcome"] == "success"
        return s
