"""Per-rank synchronized training timings with finite-gradient checks."""

import json
import time

import numpy as np
import torch
import torch.distributed as dist

from audit_training import FiniteGradientAudit


class DistributedTimingAudit(FiniteGradientAudit):
    def on_train_start(self, state, unit):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.root = self.path.parent / "timing"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"steps-rank-{self.rank}.jsonl"
        self.steps = []
        self.eval_seconds = 0.0
        self.eval_calls = 0
        super().on_train_start(state, unit)
        dist.barrier()
        torch.cuda.synchronize()
        self.started = time.time()
        self.monotonic_started = time.perf_counter()
        self.write_progress(unit, "training")

    def on_train_step_start(self, state, unit):
        torch.cuda.synchronize()
        self.step_started = time.perf_counter()

    def write_progress(self, unit, phase):
        row = dict(
            rank=self.rank,
            world_size=self.world,
            gpu=torch.cuda.get_device_name(),
            phase=phase,
            started_unix=self.started,
            updated_unix=time.time(),
            completed_steps=unit.train_progress.num_steps_completed,
            step_compute_seconds=sum(self.steps),
            eval_seconds=self.eval_seconds,
            eval_calls=self.eval_calls,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        path = self.root / f"progress-rank-{self.rank}.json"
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(path)

    def on_train_step_end(self, state, unit):
        torch.cuda.synchronize()
        duration = time.perf_counter() - self.step_started
        self.steps.append(duration)
        if not np.isfinite(unit.last_loss) or not np.isfinite(self.last_norm):
            raise FloatingPointError("Nonfinite training loss/gradient")
        with self.path.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        step=unit.train_progress.num_steps_completed,
                        loss=float(unit.last_loss),
                        learning_rate=float(unit.optimizer.param_groups[0]["lr"]),
                        clipped_gradient_norm=self.last_norm,
                        seconds=duration,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                )
                + "\n"
            )
        if unit.train_progress.num_steps_completed % 50 == 0:
            self.write_progress(unit, "training")
            if self.rank == 0:
                print(
                    f"TIMING_PROGRESS step={unit.train_progress.num_steps_completed} mean_step_seconds={np.mean(self.steps[-50:]):.4f}",
                    flush=True,
                )

    def on_eval_start(self, state, unit):
        torch.cuda.synchronize()
        self.eval_started = time.perf_counter()

    def on_eval_end(self, state, unit):
        torch.cuda.synchronize()
        self.eval_seconds += time.perf_counter() - self.eval_started
        self.eval_calls += 1

    def on_train_end(self, state, unit):
        torch.cuda.synchronize()
        self.write_progress(unit, "complete")
        result = json.loads((self.root / f"progress-rank-{self.rank}.json").read_text())
        steady = self.steps[20:] or self.steps
        result.update(
            train_callback_wall_seconds=time.perf_counter() - self.monotonic_started,
            mean_step_seconds=float(np.mean(self.steps)),
            steady_mean_step_seconds=float(np.mean(steady)),
            steady_median_step_seconds=float(np.median(steady)),
            steady_p95_step_seconds=float(np.quantile(steady, 0.95)),
            excluded_warmup_steps=min(20, len(self.steps)),
        )
        (self.root / f"summary-rank-{self.rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        super().on_train_end(state, unit)
