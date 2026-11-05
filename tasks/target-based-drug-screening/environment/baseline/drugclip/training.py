"""Stage-independent optimizer schedule and checkpoint selection helpers."""

from __future__ import annotations

import math


class PolynomialSchedule:
    """Uni-Core polynomial_decay: power 1, end LR 0 by default."""

    def __init__(self, optimizer, *, base_lr: float, total_updates: int,
                 warmup_ratio: float = 0.06, power: float = 1.0,
                 end_lr: float = 0.0) -> None:
        if total_updates < 1 or not 0 <= warmup_ratio < 1 or power <= 0:
            raise ValueError("invalid polynomial schedule")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_updates = total_updates
        self.warmup_updates = int(warmup_ratio * total_updates)
        self.power = power
        self.end_lr = end_lr
        self.updates = 0
        self._set_lr(self._lr_at(0))

    def _lr_at(self, step: int) -> float:
        warmup = self.warmup_updates
        if warmup and step <= warmup:
            return self.base_lr * max(1, step) / warmup
        if step >= self.total_updates:
            return self.end_lr
        remaining = 1 - (step - warmup) / (self.total_updates - warmup)
        return (self.base_lr - self.end_lr) * remaining ** self.power + self.end_lr

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> float:
        self.updates += 1
        value = self._lr_at(self.updates)
        self._set_lr(value)
        return value

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in
                ("base_lr", "total_updates", "warmup_updates", "power",
                 "end_lr", "updates")}

    def load_state_dict(self, state: dict) -> None:
        for key in ("base_lr", "total_updates", "warmup_updates", "power", "end_lr"):
            if state[key] != getattr(self, key):
                raise ValueError(f"resume changed polynomial schedule {key}")
        self.updates = int(state["updates"])
        self._set_lr(self._lr_at(self.updates))


def optimizer_updates(loader_length: int, accumulation: int) -> int:
    if loader_length < 1 or accumulation < 1:
        raise ValueError("empty loader or invalid accumulation")
    return math.ceil(loader_length / accumulation)
