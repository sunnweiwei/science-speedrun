"""KernelSynth-style synthetic series (Chronos, arXiv:2403.07815 §4.2), re-implemented from the paper's description.

A random composition (sum / product) of 1-6 Gaussian-process kernels drawn
from a bank of linear, RBF, rational-quadratic, white-noise and periodic
kernels (periods tied to common seasonalities) is sampled once per series;
the series is a draw from the resulting GP on a unit grid. Allowed by the
task rules as self-generated data; declare it as "synthetic:kernelsynth" in
model_manifest.json. Not the released chronos_datasets/kernel_synth file
(that file has no per-config license and is not part of the task).
"""
import numpy as np

PERIODS = (24, 48, 96, 168, 288, 336, 7, 12, 4, 52, 365)


def _rbf(t, ls):
    d = (t[:, None] - t[None, :]) / ls
    return np.exp(-0.5 * d * d)


def _periodic(t, period, ls):
    d = np.pi * np.abs(t[:, None] - t[None, :]) / period
    return np.exp(-2.0 * np.sin(d) ** 2 / (ls * ls))


def _rq(t, ls, alpha):
    d = (t[:, None] - t[None, :]) / ls
    return (1.0 + d * d / (2 * alpha)) ** (-alpha)


def _linear(t, c):
    return (t[:, None] - c) * (t[None, :] - c) / (len(t) ** 2)


def random_kernel(rng, t):
    kind = rng.choice(["rbf", "periodic", "rq", "linear", "white"], p=[0.3, 0.35, 0.15, 0.1, 0.1])
    if kind == "rbf":
        return _rbf(t, rng.uniform(1.0, len(t) / 2))
    if kind == "periodic":
        return _periodic(t, float(rng.choice(PERIODS)), rng.uniform(0.3, 2.0))
    if kind == "rq":
        return _rq(t, rng.uniform(1.0, len(t) / 2), rng.uniform(0.1, 10.0))
    if kind == "linear":
        return _linear(t, rng.uniform(0, len(t)))
    return np.eye(len(t)) * rng.uniform(0.01, 0.3)


def kernelsynth(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    k = random_kernel(rng, t)
    for _ in range(rng.integers(0, 6)):
        k = (k + random_kernel(rng, t)) if rng.random() < 0.5 else (k * random_kernel(rng, t))
    k = k + 1e-4 * np.eye(length)
    try:
        chol = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(k + 1e-2 * np.eye(length))
    x = chol @ rng.standard_normal(length)
    x = (x - x.mean()) / (x.std() + 1e-8)
    scale = np.exp(rng.uniform(-2, 4)); offset = rng.uniform(-3, 3) * scale
    return (x * scale + offset).astype(np.float32)


def _worker(args):
    seed, length = args
    return kernelsynth(np.random.default_rng(seed), length)


def make_bank(n: int, length: int, seed: int = 0, workers: int = 8) -> np.ndarray:
    """Pre-generate `n` synthetic series of `length` in parallel (about 1-2 s per series per core)."""
    from multiprocessing import Pool

    with Pool(workers) as pool:
        rows = pool.map(_worker, [(seed * 1_000_003 + i, length) for i in range(n)], chunksize=8)
    return np.stack(rows)


class BankSampler:
    """Callable (rng, length) -> a random window of a pre-generated bank, for WindowSampler(synthetic=...)."""

    def __init__(self, bank: np.ndarray):
        self.bank = bank

    def __call__(self, rng: np.random.Generator, length: int) -> np.ndarray:
        row = self.bank[rng.integers(len(self.bank))]
        if length >= len(row):
            return row[-length:] if length == len(row) else np.pad(row, (length - len(row), 0), mode="edge")
        off = rng.integers(0, len(row) - length + 1)
        return row[off:off + length]
