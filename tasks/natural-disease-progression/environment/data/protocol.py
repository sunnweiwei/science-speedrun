"""Versioned, model-independent cohort and label rules."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

HORIZONS = tuple(range(30, 361, 30)) + (365,)
SPLIT_SEED = "natural-disease-progression-mimic-iv-3.1-v1"
GEM_SHA256 = "1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2"
GEM_URL = "https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip"
ICD10_CATEGORY = re.compile(r"^[A-Z][0-9][0-9A-Z]$")


def split_subjects(subject_ids: list[int]) -> dict[int, str]:
    """Exact 70/15/15 sizes, stable under CSV row reordering."""
    ordered = sorted(subject_ids, key=lambda x: hashlib.sha256(f"{SPLIT_SEED}:{x}".encode()).digest())
    n = len(ordered)
    a, b = int(n * .70), int(n * .85)
    return {subject: ("train" if i < a else "validation" if i < b else "test")
            for i, subject in enumerate(ordered)}


def patient_key(subject_id: int) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:patient:{subject_id}".encode()).hexdigest()[:24]


def icd10_category(code: str) -> str | None:
    category = code.strip().upper().replace(".", "")[:3]
    return category if ICD10_CATEGORY.fullmatch(category) else None


def is_scoring_disease(category: str) -> bool:
    """ICD-10-CM disease/congenital chapters plus special U codes.

    R symptoms, S/T injuries, V-Y external causes and Z encounter/status codes
    remain in history but are outside the v1 disease endpoint family.
    """
    return bool(ICD10_CATEGORY.fullmatch(category) and ("A" <= category[0] <= "Q" or category[0] == "U"))


def read_gem(path: Path) -> dict[str, tuple[str, str | None, tuple[str, ...]]]:
    """Return status, single target, all candidate categories for each ICD-9 code.

    GEM approximate/no-map/combination flags or multiple categories are never
    treated as a confirmed diagnosis. Candidate categories only mask scoring.
    """
    if hashlib.sha256(path.read_bytes()).hexdigest() != GEM_SHA256:
        raise ValueError("CMS 2018 GEM archive checksum mismatch")
    candidates: dict[str, set[str]] = {}
    flags: dict[str, set[str]] = {}
    with zipfile.ZipFile(path) as archive:
        with archive.open("2018_I9gem.txt") as stream:
            for raw in stream:
                fields = raw.decode("ascii").strip().split()
                if len(fields) != 3:
                    continue
                source, target, indicator = fields
                category = icd10_category(target)
                candidates.setdefault(source, set())
                flags.setdefault(source, set()).add(indicator)
                if category:
                    candidates[source].add(category)
    result = {}
    for source, targets in candidates.items():
        sorted_targets = tuple(sorted(targets))
        clear = len(targets) == 1 and flags[source] == {"00000"}
        status = "clear" if clear else "ambiguous" if targets else "unmapped"
        result[source] = status, sorted_targets[0] if clear else None, sorted_targets
    return result


def conservative_actual_end(anchor_year: int, group: str, shifted_cutoff_year: int) -> int:
    """Latest possible calendar year implied by the 3-year anchor group."""
    return shifted_cutoff_year - anchor_year + int(group[-4:])
