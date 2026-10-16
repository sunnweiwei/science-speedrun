"""Novelty against the fit corpus: the largest TM-score of a design to any fit chain.

Protein-design benchmarks report novelty as the maximum TM-score of each designable design to the PDB or the
training set, computed with Foldseek in TM-align mode (lower is more novel); generative models typically sit
around 0.45-0.8 depending on the task. A design therefore passes only if, besides the published designability
criterion, its maximum TM-score to the fit corpus is below `common.NOVELTY_MAX_TM`.

The reference is the fit corpus the agent trains on (`development/fit/structures`), mounted read-only into
the grader; its Foldseek database is built once per grading run (a few minutes on 16k structures).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

FOLDSEEK = os.environ.get("FOLDSEEK", "/opt/foldseek/bin/foldseek")


class FitNovelty:
    def __init__(self, reference_dir, workdir=None, threads=None, foldseek=FOLDSEEK):
        self.reference_dir = Path(reference_dir)
        if not any(self.reference_dir.rglob("*.cif*")):
            raise FileNotFoundError(f"novelty reference has no structures: {self.reference_dir}")
        self.foldseek = foldseek
        self.threads = str(threads or os.cpu_count() or 1)
        self.work = Path(workdir or tempfile.mkdtemp(prefix="novelty-"))
        self.db = self.work / "fitdb"
        self._run("createdb", str(self.reference_dir), str(self.db), "--threads", self.threads)

    def _run(self, *arguments):
        subprocess.run([self.foldseek, *arguments], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)

    def max_tm(self, designs):
        """designs: {key: PDB path} -> {key: max TM-score (normalized by the design) to any fit chain}."""
        if not designs:
            return {}
        query = Path(tempfile.mkdtemp(dir=self.work, prefix="query-"))
        try:
            names = {}
            for index, (key, path) in enumerate(designs.items()):
                name = f"q{index}"
                shutil.copyfile(path, query / f"{name}.pdb")
                names[name] = key
            self._run("createdb", str(query), str(query / "db"), "--threads", self.threads)
            # TM-align alignment of every prefilter hit; -a keeps the alignment so qtmscore can be reported.
            self._run("search", str(query / "db"), str(self.db), str(query / "aln"), str(query / "tmp"),
                      "--alignment-type", "1", "-a", "-e", "inf", "--max-seqs", "1000", "-s", "9.5",
                      "--threads", self.threads)
            self._run("convertalis", str(query / "db"), str(self.db), str(query / "aln"),
                      str(query / "hits.tsv"), "--format-output", "query,target,qtmscore")
            best = {key: 0.0 for key in designs}
            for line in (query / "hits.tsv").read_text().splitlines():
                name, _, score = line.split("\t")
                key = names.get(name.split("_")[0])
                if key is not None:
                    best[key] = max(best[key], float(score))
            return best
        finally:
            shutil.rmtree(query, ignore_errors=True)
