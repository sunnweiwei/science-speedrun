"""Capture exact validation returns without re-running distributed reductions."""

import json
from pathlib import Path
import time

from timing_training import DistributedTimingAudit


class LearningCurveAudit(DistributedTimingAudit):
    def checkpoint_ready(self, unit, name):
        if self.rank != 0:
            return
        folder = Path(unit.job_config.metadata.checkpoint_dir) / name
        if (folder / "inference_ckpt.pt").exists() and (folder / ".metadata").exists():
            (folder / "audit-ready.json").write_text(
                json.dumps(
                    {
                        "step": unit.train_progress.num_steps_completed,
                        "epoch": unit.train_progress.num_epochs_completed,
                        "weights": "model",
                        "unix": time.time(),
                    }
                )
            )

    def on_train_step_start(self, state, unit):
        # This callback follows FAIRChem's checkpoint callback in the config.
        self.checkpoint_ready(unit, f"step_{unit.train_progress.num_steps_completed}")
        super().on_train_step_start(state, unit)

    def on_train_start(self, state, unit):
        super().on_train_start(state, unit)
        self.original_eval_end = unit.eval_unit.on_eval_epoch_end

        def record_eval(eval_state):
            metrics = self.original_eval_end(eval_state)
            values = {key: float(value) for key, value in metrics.items()}
            record = dict(
                step=unit.train_progress.num_steps_completed,
                rank=self.rank,
                world_size=self.world,
                unix=time.time(),
                weights="ema" if unit.ema_model is not None else "model",
                metrics=values,
            )
            with (self.root / f"validation-rank-{self.rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if self.rank == 0:
                print("VALIDATION_PROGRESS " + json.dumps(record), flush=True)
            return metrics

        unit.eval_unit.on_eval_epoch_end = record_eval

    def on_train_end(self, state, unit):
        self.checkpoint_ready(unit, "final")
        super().on_train_end(state, unit)
        unit.eval_unit.on_eval_epoch_end = self.original_eval_end
