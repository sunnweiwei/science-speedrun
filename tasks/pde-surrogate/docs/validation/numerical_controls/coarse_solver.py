"""Run the numerical-control comparison from public contexts on the published SSI observation grid."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# The official APE2D simulation_setups_2d_low_res.py uses these internal steps per saved frame.
SUBSTEPS = {
    "Diffusion": 1,
    "Fisher-KPP": 1,
    "Burgers": 50,
    "Swift-Hohenberg": 5,
    "Kuramoto-Sivashinsky": 5,
    "Korteweg-de-Vries": 10,
    "Navier-Stokes: Decaying Turbulence": 500,
    "Navier-Stokes: Kolmogorov Flow": 1500,
}


def ape_operator(equations, name, extent, grid, step, params):
    args = (2, extent, grid, step)
    if name == "Diffusion":
        import jax.numpy as jnp
        nu = jnp.array([params["Viscosity X"], params["Viscosity Y"]])
        return equations.Diffusion(*args, diffusivity=nu)
    if name == "Fisher-KPP":
        return equations.reaction.FisherKPP(*args, diffusivity=params["Diffusivity"], reactivity=params["Reactivity"])
    if name == "Gray Scott":
        return equations.reaction.GrayScott(
            *args, diffusivity_1=2e-5, diffusivity_2=1e-5,
            feed_rate=params["Feed Rate"], kill_rate=params["Kill Rate"],
        )
    if name == "Swift-Hohenberg":
        return equations.reaction.SwiftHohenberg(
            *args, reactivity=params["Reactivity"], critical_number=params["Critical Number"]
        )
    if name == "Kuramoto-Sivashinsky":
        return equations.KuramotoSivashinsky(*args)
    operator = {
        "Burgers": equations.Burgers,
        "Korteweg-de-Vries": equations.KortewegDeVries,
        "Navier-Stokes: Decaying Turbulence": equations.NavierStokesVorticity,
        "Navier-Stokes: Kolmogorov Flow": equations.KolmogorovFlowVorticity,
    }[name]
    return operator(*args, diffusivity=params["Viscosity"])


def ape_forecaster():
    import equinox as eqx
    import exponax as ex
    import jax
    import jax.numpy as jnp

    @eqx.filter_jit
    def rollout(operator, initial, substeps, length):
        def macro_step(state, _):
            state = jax.lax.fori_loop(0, substeps, lambda i, v: operator.step_fourier(v), state)
            return state, jnp.fft.irfft2(state, s=initial.shape[-2:])

        return jax.lax.scan(macro_step, jnp.fft.rfft2(initial), None, length=length)[1]

    def predict(initial, meta, length):
        pde, dt = meta["pde"], float(meta["dt"])
        # All eight released Gray–Scott variants use an internal interval of one.
        substeps = int(dt) if pde == "Gray Scott" else SUBSTEPS[pde]
        n, m = initial.shape[-2:]
        lx, ly = meta["domain_extent"]
        if (n, m) != (256, 256) or lx != ly or tuple(meta["boundary_conditions"]) != ("periodic",) * 4:
            raise ValueError("Expected the released periodic, square APE2D grid")
        operator = ape_operator(ex.stepper, pde, float(lx), n, dt / substeps, meta["constants"])
        return np.asarray(rollout(operator, jnp.asarray(initial), substeps, length))

    return predict


def pdearena_forecast(initial, meta, length):
    from phi.flow import Box, CenteredGrid, StaggeredGrid, advect, diffuse, extrapolation, fluid, math
    from phi.math import reshaped_native

    n, m = initial.shape[-2:]
    # The archive publishes derived coordinates (32*128/127, 84/13), while the
    # original stepper uses a physical length of 32 and four 1.5-unit updates.
    if (
        (n, m) != (128, 128)
        or meta["pde"] != "Navier-Stokes: buoyant smoke (PDEArena)"
        or not np.allclose(meta["domain_extent"], [32 * 128 / 127] * 2)
        or not np.isclose(meta["dt"], 84 / 13)
        or tuple(meta["fields"]) != ("u", "vx", "vy")
    ):
        raise ValueError("Public metadata differs from the frozen PDEArena simulator")
    bounds = Box[0:32, 0:32]
    axes = math.spatial("x,y")
    smoke = CenteredGrid(math.tensor(initial[0], axes), extrapolation.BOUNDARY, bounds=bounds)
    # The archive includes the low wall and drops the high wall in each face-normal direction.
    faces = math.stack(
        [math.tensor(initial[1, 1:, :], axes), math.tensor(initial[2, :, 1:], axes)],
        math.channel(vector="x,y")
    )
    velocity = StaggeredGrid(faces, extrapolation.ZERO, bounds=bounds)
    buoyancy = float(meta["constants"]["buoyancy_y"])
    predictions = []
    for _ in range(length):
        for _ in range(4):
            smoke = advect.semi_lagrangian(smoke, velocity, 1.5)
            force = (smoke * (0, buoyancy)).at(velocity)
            velocity = advect.semi_lagrangian(velocity, velocity, 1.5) + 1.5 * force
            velocity = diffuse.explicit(velocity, 0.01, 1.5)
            velocity, _ = fluid.make_incompressible(velocity)
        c = np.asarray(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True)).squeeze()
        v = np.asarray(reshaped_native(velocity.staggered_tensor(), groups=("x", "y", "vector"), to_numpy=True))
        predictions.append(np.stack([c, v[:-1, :-1, 0], v[:-1, :-1, 1]]))
    return np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=["ape2d", "pdearena", "well"], required=True)
    args = parser.parse_args()
    with (args.input / "index.json").open() as source:
        manifest = json.load(source)
    cases = [c for c in manifest["cases"] if c["dataset"].split("/")[0] == args.group]
    if not cases:
        raise ValueError(f"The public index contains no {args.group} cases")
    if args.group == "ape2d":
        predict = ape_forecaster()
    elif args.group == "pdearena":
        predict = pdearena_forecast
    else:
        def predict(initial, meta, length):
            return np.repeat(initial[None], length, axis=0)
    (args.output / "cases").mkdir(exist_ok=True, parents=True)
    failures = []
    for case in cases:
        cid = case["case_id"]
        with h5py.File(args.input / "cases" / f"{cid}.h5", "r") as f:
            initial = f["input"][-1]
            meta = dict(f.attrs)
            meta["constants"] = json.loads(meta["constants"])
        target = args.output / "cases" / f"{cid}.h5"
        values = predict(initial, meta, case["n_steps_out"])
        if not np.isfinite(values).all():
            # The grader counts an absent prediction as maximum error for the entire case.
            target.unlink(missing_ok=True)
            failures.append(cid)
            continue
        temp = target.with_suffix(".partial.h5")
        with h5py.File(temp, "w") as f:
            f.create_dataset("prediction", data=values.astype(np.float32))
        temp.replace(target)
    (args.output / f"index.{args.group}.json").write_text(json.dumps({"contract": manifest["contract"], "cases": cases}) + "\n")
    (args.output / f"failed-cases.{args.group}.json").write_text(json.dumps(failures) + "\n")


if __name__ == "__main__":
    main()
