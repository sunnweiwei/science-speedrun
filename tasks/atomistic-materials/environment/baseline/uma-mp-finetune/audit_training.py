"""Abort before an optimizer update with nonfinite gradients; retain pilot evidence."""

import json
import math
from pathlib import Path
import torch
from torchtnt.framework.callback import Callback


class FiniteGradientAudit(Callback):
    def __init__(self, output):
        self.path = Path(output)
        self.last_norm = None
        self.hook = None

    def on_train_start(self, state, unit):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def check(optimizer, args, kwargs):
            norms = [
                p.grad.detach().norm()
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            norm = torch.linalg.vector_norm(torch.stack(norms))
            self.last_norm = float(norm)
            if not math.isfinite(self.last_norm):
                raise FloatingPointError("Nonfinite gradient before optimizer step")

        self.hook = unit.optimizer.register_step_pre_hook(check)

    def on_train_step_end(self, state, unit):
        if not math.isfinite(unit.last_loss):
            raise FloatingPointError("Nonfinite training loss")
        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "step": unit.train_progress.num_steps_completed,
                        "loss": unit.last_loss,
                        "clipped_gradient_norm": self.last_norm,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
                + "\n"
            )

    def on_train_end(self, state, unit):
        self.hook.remove()
