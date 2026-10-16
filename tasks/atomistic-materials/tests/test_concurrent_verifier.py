"""The verifier runs every suite at once: worker table, out-of-memory retries, item order."""
import pytest

import common
import main_verifier


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError, matched by name."""


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    import types
    import sys
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def test_every_suite_has_workers():
    assert set(main_verifier.WORKERS_PER_GPU) == set(main_verifier.SUITES)
    assert all(n >= 1 for n in main_verifier.WORKERS_PER_GPU.values())


def test_out_of_memory_is_retried():
    calls = []

    def run():
        calls.append(1)
        if len(calls) < 3:
            raise OutOfMemoryError("CUDA out of memory")
        return {"ok": True}
    assert common.with_oom_retry(run) == {"ok": True}
    assert len(calls) == 3


def test_persistent_out_of_memory_fails_the_case():
    def run():
        raise OutOfMemoryError("CUDA out of memory")
    with pytest.raises(OutOfMemoryError):
        common.with_oom_retry(run, attempts=2)


def test_other_errors_are_not_retried():
    calls = []

    def run():
        calls.append(1)
        raise ValueError("model error")
    with pytest.raises(ValueError):
        common.with_oom_retry(run)
    assert len(calls) == 1
