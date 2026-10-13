"""Small MIMIC-shaped source used only by the two-phase verifier smoke test."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def fixture_source(root: Path, *, future_value: str = "8",
                   reverse_diagnoses: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "CHANGELOG.txt").write_text("MIMIC-IV v3.1\n")
    hosp = root / "hosp"
    subjects = range(10000001, 10000013)
    _write(hosp / "patients.csv.gz",
           ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
           [[s, "F" if s % 2 else "M", 50, 2150, "2011 - 2013", ""] for s in subjects])
    admissions, diagnoses, labs, prescriptions, pharmacy, emar = [], [], [], [], [], []
    for subject in subjects:
        first = subject + 10000000
        admissions.append([subject, first, "2150-01-01 00:00:00",
                           "2150-01-03 12:00:00", "", 0])
        diagnoses.extend([[subject, first, 2, "A001", 10],
                          [subject, first, 1, "A002", 10]])
        labs.append([subject, first, subject * 10, 50000,
                     "2150-01-02 08:00:00", "2150-01-02 10:00:00", "2", "2", "mg/dL"])
        prescriptions.append([subject, first, first, "2150-01-02 12:00:00",
                              "2150-01-05 00:00:00", "00000000001", "1", "mg"])
        pharmacy.append([subject, first, first, "2150-01-02 11:00:00"])
        emar.append([subject, first, first, first, "2150-01-02 13:00:00",
                     "2150-01-02 14:00:00", "Administered"])
        if subject != 10000001:
            second = first + 100
            admissions.append([subject, second, "2150-03-01 00:00:00",
                               "2150-03-03 10:00:00", "", 0])
            diagnoses.extend([[subject, second, 1, "B001", 10],
                              [subject, second, 2, "B002", 10]])
            labs.append([subject, second, subject * 10 + 1, 50000,
                         "2150-03-02 08:00:00", "2150-03-02 10:00:00",
                         future_value, future_value, "mg/dL"])
    _write(hosp / "admissions.csv.gz",
           ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"], admissions)
    _write(hosp / "diagnoses_icd.csv.gz",
           ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
           list(reversed(diagnoses)) if reverse_diagnoses else diagnoses)
    _write(hosp / "d_icd_diagnoses.csv.gz", ["icd_code", "icd_version", "long_title"],
           [[code, 10, "synthetic category"] for code in ("A001", "A002", "B001", "B002")])
    _write(hosp / "labevents.csv.gz",
           ["subject_id", "hadm_id", "labevent_id", "itemid", "charttime", "storetime",
            "value", "valuenum", "valueuom"], labs)
    _write(hosp / "prescriptions.csv.gz",
           ["subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime", "ndc",
            "dose_val_rx", "dose_unit_rx"], prescriptions)
    _write(hosp / "pharmacy.csv.gz",
           ["subject_id", "hadm_id", "pharmacy_id", "entertime"], pharmacy)
    _write(hosp / "emar.csv.gz",
           ["subject_id", "hadm_id", "emar_id", "pharmacy_id", "charttime", "storetime",
            "event_txt"], emar)
