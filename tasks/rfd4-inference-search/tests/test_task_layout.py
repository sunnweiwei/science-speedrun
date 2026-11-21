"""Check the repository's task format and the two Harbor build entry points."""
from pathlib import Path
import re
import unittest


TASK = Path(__file__).resolve().parents[1]


class TaskLayoutTest(unittest.TestCase):
    def test_distributed_text_and_filenames_have_no_chinese_script(self):
        han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U0003347f]")
        text_formats = {".md", ".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".csv", ".tsv", ".txt",
                        ".log", ".sha256", ".rst", ".html", ".js", ".svg"}
        for path in TASK.rglob("*"):
            relative = path.relative_to(TASK)
            if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in relative.parts):
                continue
            self.assertNotRegex(relative.as_posix(), han)
            if path.is_file() and (path.suffix.lower() in text_formats or path.name in {"Dockerfile", "LICENSE", ".dockerignore"}):
                self.assertIsNone(han.search(path.read_text(encoding="utf-8")), f"Chinese text in {relative}")

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
