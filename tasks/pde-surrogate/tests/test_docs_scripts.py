"""Every committed script a document tells a reader to run must at least be runnable.

This exists because of a real miss: the figure generators were committed as "regenerate with
python3 docs/make_*.py" while one hardcoded a scratch directory that no longer existed and the
others wrote their output into the working directory under a different name, so running them
from the repository did not reproduce the committed file. Prose consistency is greppable; a
claim nobody ever executed is not.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
def _referenced_scripts():
    """Every .py a committed document tells a reader to run, resolved to a path in this task."""
    import re
    docs = [ROOT / n for n in ("README.md", "docs/RUNNING.md", "docs/AGENT_EVALUATION.md")]
    docs += list(ROOT.rglob("environment/**/README.md")) + [ROOT / "tests" / "README.md"]
    found = set()
    for d in docs:
        if not d.exists():
            continue
        for m in re.finditer(r"python3\s+([A-Za-z0-9_./-]+\.py)", d.read_text()):
            raw = m.group(1).split("/")[-1]
            hits = [q for q in ROOT.rglob(raw) if "__pycache__" not in str(q)]
            found.update(hits)
    return found


SCRIPTS = sorted({*(ROOT / "docs").glob("make_*.py"),
                  *(p for p in (ROOT / "environment" / "data").glob("*.py") if p.name != "__init__.py"),
                  *_referenced_scripts()})


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_responds_to_help(script):
    """--help must work without any data present: it is how a reader discovers the inputs."""
    out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"{script.name} --help failed:\n{out.stderr[-800:]}"
    assert "usage:" in out.stdout.lower()


@pytest.mark.parametrize("script", sorted((ROOT / "docs").glob("make_*.py")), ids=lambda p: p.name)
def test_figure_script_defaults_to_the_committed_path(script):
    """A generator must default to writing the file the README shows, not to the cwd."""
    out = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120)
    expected = script.name.replace("make_", "").replace(".py", ".png")
    assert expected in out.stdout, f"{script.name} --help does not mention its default output {expected}"
    assert (ROOT / "docs" / expected).exists(), f"{expected} is referenced but not committed"
