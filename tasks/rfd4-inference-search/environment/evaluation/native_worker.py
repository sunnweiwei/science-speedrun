"""Optional numeric pruning rule for native monomer search. The generator supplies all states."""
import json
from pathlib import Path

import torch
import rfproteina.generate as native
from rfproteina.search.beam_search import BeamSearchStrategy


def evaluate(expr, features):
    if isinstance(expr, str):
        return features[expr]
    if isinstance(expr, (int, float)):
        return torch.full_like(features["quality"], float(expr))
    op, left, right = expr
    fn = {"+": torch.add, "-": torch.sub, "*": torch.mul, "min": torch.minimum, "max": torch.maximum}[op]
    return fn(evaluate(left, features), evaluate(right, features)).clamp(-1e4, 1e4)


class MonomerSearch(BeamSearchStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        priorities = json.loads(Path("priorities.json").read_text())
        self.rules = {self.reward_models[int(k)]: (int(k), v) for k, v in priorities.items()}
        self.rng = torch.Generator(device="cpu").manual_seed(torch.initial_seed())

    def _compute_rewards(self, model, atom_array_results, reward_model, **kwargs):
        values, metrics = super()._compute_rewards(model, atom_array_results, reward_model, **kwargs)
        if reward_model not in self.rules:
            return values, metrics
        step, expr = self.rules[reward_model]
        branches = self.checkpoints[step].get("n_branch", self.n_branch)
        count = branches * self.beam_width
        # Native order is [branch, parent, independent replicate]. Rank within each replicate.
        q = values.reshape(count, -1)
        valid = torch.isfinite(q)
        if not valid.any():
            raise RuntimeError("No finite clash reward at the monomer search checkpoint")
        order = torch.argsort(q, dim=0, descending=True, stable=True)
        rank = 1 - torch.argsort(order, dim=0).float() / max(1, count - 1)
        parent = (torch.arange(count, device=q.device) % self.beam_width).reshape(-1, 1)
        features = {"quality": torch.where(valid, q, 0.0), "rank": rank,
                    "parent": (1 - parent.float() / max(1, self.beam_width - 1)).expand_as(q),
                    "draw": torch.rand(q.shape, generator=self.rng).to(q.device)}
        priority = evaluate(expr, features).masked_fill(~valid, -torch.inf)
        return priority.reshape_as(values), metrics


if __name__ == "__main__":
    native.BeamSearchStrategy = MonomerSearch
    native.main_hydra()
