"""Resume must retain completed files, preserve a bad tail, and exclude heldout data."""

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
import resume_lfmc_extraction as resume


class ResumeTests(unittest.TestCase):
    def test_closed_file_milestone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log"
            path.write_text(
                str({"files": 100000, "windows": resume.EXPECTED})
                + "\nerror: TIMEOUT\n"
            )
            self.assertEqual(resume.logged_prefix(path), 100000)

    def test_resume_suffix_and_preserve_bad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data.tar"
            out = root / "output"
            out.mkdir()
            (out / "config.json").write_bytes(b"{}")
            before = (out / "config.json").stat().st_mtime_ns
            expected = {"train": 1, "val": 0, "test": 1}
            entries = {
                "data/config.json": b"{}",
                "data/windows/spatial_split/train/metadata.json": json.dumps(
                    {"options": {"split": "train"}}
                ).encode(),
                "data/windows/spatial_split/train/payload.bin": b"complete train data",
                "data/windows/spatial_split/test/metadata.json": json.dumps(
                    {"options": {"split": "test"}}
                ).encode(),
                "data/windows/spatial_split/test/payload.bin": b"heldout payload",
            }
            target = out / "windows/spatial_split/train/payload.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"partial")
            with tarfile.open(archive, "w") as tar:
                for name, value in entries.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(value)
                    tar.addfile(member, io.BytesIO(value))
            log = root / "log"
            log.write_text("preserved log")
            argv = [
                "resume",
                "--previous-job",
                "123",
                "--previous-log",
                str(log),
                "--archive",
                str(archive),
                "--output",
                str(out),
                "--quarantine",
                str(root / "quarantine"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(resume, "EXPECTED", expected),
                patch.object(resume, "logged_prefix", return_value=1),
                patch.object(
                    resume.subprocess, "check_output", return_value="123|TIMEOUT\n"
                ),
            ):
                resume.main()
            self.assertEqual(target.read_bytes(), b"complete train data")
            self.assertEqual(
                (
                    root / "quarantine/windows/spatial_split/train/payload.bin"
                ).read_bytes(),
                b"partial",
            )
            self.assertEqual((out / "config.json").stat().st_mtime_ns, before)
            self.assertFalse((out / "windows/spatial_split/test").exists())
            report = json.loads((out / "extraction.json").read_text())
            self.assertEqual(report["files"], 3)
            self.assertEqual(report["quarantined_incomplete_files"], 1)


if __name__ == "__main__":
    unittest.main()
