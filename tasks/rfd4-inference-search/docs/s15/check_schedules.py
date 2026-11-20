"""CPU preflight for the exact time grids and Langevin coefficients used in the monomer audit."""
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch
import rfproteina
from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher, get_gt
from rfproteina.flow_matching.rdn_flow_matcher import compute_tsr_ratio

from run_cohort import SAMPLERS, config, prior_schedule


def check():
    base = OmegaConf.load(Path(rfproteina.__file__).parent / "configs/generation/inference_base.yaml")
    matcher = SimpleNamespace(data_modes=("bb_ca", "local_latents"))
    native = ProductSpaceFlowMatcher.sample_schedule
    grids = {}
    for arm in SAMPLERS:
        gen = config(arm, 100, Path("/assets"), Path("/run"))["generation"]
        model = OmegaConf.merge(base.model, gen.get("model", {}))
        steps = gen["args"]["nsteps"]
        assert sum(n for n, _ in gen["n_recycle_schedule"]) == steps
        if arm == "prior-tsr":
            ts, g = prior_schedule(native, matcher, steps, model)
        else:
            ts, g = native(matcher, steps, model)
        grids[arm] = ts
        for key in matcher.data_modes:
            times = ts[key][:-1]
            assert len(times) == steps and torch.isfinite(ts[key]).all() and torch.all(ts[key][1:] >= times)
            mode = model[key].simulation_step_params
            gt = base.model[key].gt
            beta = get_gt(t=times, mode=gt.mode, param=gt.p, clamp_val=gt.clamp_val)
            if arm == "stochastic-tsr":
                expected = beta / compute_tsr_ratio(times, mode.tsr_k, mode.tsr_sigma).square()
            elif arm == "ode-tsr":
                expected = torch.zeros_like(beta)
            else:
                expected = beta
            torch.testing.assert_close(g[key], expected)
            scale = (0.3 if key == "bb_ca" else 0.1) if arm in ("prior-tsr", "full-sde", "short-sde") else 1.0
            assert mode.sc_scale_noise == scale
        print(arm, steps, sum(n*d for n,d in gen["n_recycle_schedule"]), "trunk passes per slot")
    for key in matcher.data_modes:
        for arm in ("prior-tsr", "ode-tsr"):
            torch.testing.assert_close(grids["stochastic-tsr"][key], grids[arm][key], rtol=0, atol=0)


if __name__ == "__main__":
    check()
