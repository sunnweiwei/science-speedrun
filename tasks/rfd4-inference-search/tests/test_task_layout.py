"""Check the repository's task format and the two Harbor build entry points."""
from pathlib import Path
import re
import unittest


TASK = Path(__file__).resolve().parents[1]


class TaskLayoutTest(unittest.TestCase):
    def test_agent_and_developer_sections(self):
        prompt = (TASK / "instruction.md").read_text()
        self.assertEqual(re.findall(r"^## (.+)$", prompt, re.M),
                         ["Task and goal", "Workspace", "Rules", "Submission", "Continuous research"])
        readme = (TASK / "README.md").read_text()
        self.assertEqual(re.findall(r"^## (?:\d+\. )?(.+)$", readme, re.M),
                         ["Overview", "Data", "Workspace", "Evaluation", "Running", "Results and validation",
                          "Limitations", "Citation and licenses"])

    def test_each_baseline_has_instructions_and_run_evidence(self):
        baseline = TASK / "environment/baseline"
        for directory in baseline.iterdir():
            if not directory.is_dir() or directory.name == "__pycache__":
                continue
            with self.subTest(baseline=directory.name):
                for filename in ("README.md", "RESULTS.md"):
                    self.assertTrue((directory / filename).is_file())
                self.assertTrue(any((directory / "reproduction").glob("*.json")))

    def test_verifier_entry_matches_pinned_stages(self):
        # The original environment/Dockerfile --target verifier remains usable.
        canonical = (TASK / "environment/Dockerfile").read_text().split("FROM runtime AS development")[0]
        standalone = (TASK / "tests/Dockerfile").read_text()
        self.assertEqual(standalone[standalone.index("FROM "):].strip(), canonical.strip())
