"""Export reviewable aggregate audit facts without rare candidate-code cells."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    audit.pop("candidate_disease_positive_counts_365d", None)
    stays = audit["admissions_per_patient"]
    audit["admissions_per_patient"] = {str(i): stays.get(str(i), 0) for i in range(1, 10)}
    audit["admissions_per_patient"]["10_plus"] = sum(n for k, n in stays.items() if int(k) >= 10)
    audit["per_target_positive_counts"] = "See scored_diseases.json for the frozen target list."
    args.output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
