"""What a public corpus release may and may not carry."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))
import build_release  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return build_release.permitted(json.loads((DATA / "sources.lock.json").read_text()))


def test_the_allow_list_is_the_95_directories(plan):
    _, allowed, _, files = plan
    assert len(allowed) == 95
    assert files and all(f["path"].split("/")[0] in allowed for f in files)


def test_no_leaking_directory_can_be_released(plan):
    """Six directories hold GIFT-Eval evaluation series; republishing them would propagate a leak."""
    _, allowed, leaking, files = plan
    assert leaking, "the leaking set should not be empty"
    assert not (allowed & leaking)
    shipped = {f["path"].split("/")[0] for f in files}
    for d in leaking:
        assert d not in shipped, f"{d} was measured to contain GIFT-Eval series and must not be republished"


def test_gifteval_itself_is_never_a_release_source():
    """The evaluation data is a separate repository and stays there."""
    assert build_release.SOURCE == "gifteval_pretrain"
    lock = json.loads((DATA / "sources.lock.json").read_text())
    _, _, _, files = build_release.permitted(lock)
    assert not any(f["path"].startswith("gifteval/") for f in files)


def test_a_leaking_directory_slipping_into_the_allow_list_is_caught():
    lock = json.loads((DATA / "sources.lock.json").read_text())
    src = next(s for s in lock["sources"] if s["id"] == "gifteval_pretrain")
    poisoned = json.loads(json.dumps(lock))
    tgt = next(s for s in poisoned["sources"] if s["id"] == "gifteval_pretrain")
    leaking = sorted(src["excluded_for_leakage"])[0]
    tgt["directories"][leaking] = {"bytes": 1, "files": 1}
    with pytest.raises(SystemExit):
        build_release.permitted(poisoned)


def test_no_released_file_describes_the_held_out_windows(plan):
    """pde-surrogate shipped generator seeds for its held-out set; nothing analogous exists here."""
    _, _, _, files = plan
    import re
    bad = [f["path"] for f in files if re.search(r"(?:^|[/_\-])(test|seed)(?:[/_\-.]|$)", f["path"], re.I)]
    assert not bad, bad[:5]
