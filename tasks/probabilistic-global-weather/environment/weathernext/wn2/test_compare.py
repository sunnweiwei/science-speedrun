import pytest
from compare_legacy_reports import compare


def report(value):
    return dict(status="passed", protocol_sha256="p", weatherbenchx_revision="w", evaluator_sha256="e", estimator="empirical", truth_values_sha256="t",
                per_case=[dict(variable="t",level_hpa=None,lead_hours=6,pool_cells=1,case=str(i),crps=value) for i in range(8)])


def test_known_paired_skill_and_bootstrap():
    result = compare(report(1),report(2))[0]
    assert result["skill"] == .5
    assert result["paired_crps_difference"] == -1
    assert result["paired_date_bootstrap_95pct"] == [-1,-1]


@pytest.mark.parametrize("fault",["protocol","truth","coverage","duplicate","nan"])
def test_incomparable_reports_fail(fault):
    a,b=report(1),report(2)
    if fault=="protocol": b["protocol_sha256"]="wrong"
    if fault=="truth": b["truth_values_sha256"]="different observations"
    if fault=="coverage": b["per_case"].pop()
    if fault=="duplicate": b["per_case"].append(b["per_case"][0])
    if fault=="nan": b["per_case"][0]["crps"]=float("nan")
    with pytest.raises(ValueError): compare(a,b)
