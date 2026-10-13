"""Resource gates must not reclaim running, missing, or already spent budget."""
import pytest

from baselines.accounting import allocation_ceiling


def test_pivot_reclaims_only_unused_budget() -> None:
    jobs = [dict(id="1", retired=True, gpu_count=1, array_tasks=2, max_hours_per_task=24),
            dict(id="2", gpu_count=2, array_tasks=1, max_hours_per_task=10)]
    rows = [dict(job_id="1_0", state="FAILED", gpu_hours=18),
            dict(job_id="1_1", state="CANCELLED by 10", gpu_hours=0)]
    assert allocation_ceiling(jobs, rows) == 38
    assert jobs[0]["max_hours_per_task"] == 24


@pytest.mark.parametrize("rows", [[], [dict(job_id="1_0", state="RUNNING", gpu_hours=18)],
                                  [dict(job_id="1_0", state="PENDING", gpu_hours=0)]])
def test_unsettled_usage_cannot_be_retired(rows: list[dict]) -> None:
    with pytest.raises(ValueError, match="terminal accounting"):
        allocation_ceiling([dict(id="1", retired=True)], rows)


def test_all_spent_compute_still_counts() -> None:
    assert allocation_ceiling([dict(id="1", retired=True)],
                              [dict(job_id="1", state="TIMEOUT", gpu_hours=400)]) == 400
