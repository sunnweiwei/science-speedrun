"""Preserve a completed audit and retry failed STAC requests with encoded timezones."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from external_lfmc_availability import query


def encode_datetime_plus(url):
    """PC next links may put literal UTC '+' in a query, decoded as a space."""
    parts = urlsplit(url)
    fields = parts.query.split("&")
    fields = [f.replace("+", "%2B") if f.startswith("datetime=") else f for f in fields]
    return urlunsplit(parts._replace(query="&".join(fields)))


class PaginationSession(requests.Session):
    def request(self, method, url, **kwargs):
        return super().request(method, encode_datetime_plus(url), **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((args.previous / "summary.json").read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != previous["manifest_sha256"]:
        raise ValueError("Candidate manifest changed")
    sites = {s["site_id"]: s for s in json.loads(args.manifest.read_text())["sites"]}
    args.output.mkdir(parents=True, exist_ok=False)
    session = PaginationSession()
    reports = []
    for entry in previous["sites"]:
        result = json.loads((args.previous / entry["report"]).read_text())
        for modality in entry["errors"]:
            try:
                result["modalities"][modality] = query(
                    session, sites[entry["site_id"]], modality
                )
            except Exception as exc:
                result["modalities"][modality] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
        (args.output / entry["report"]).write_text(json.dumps(result, indent=2) + "\n")
        reports.append(
            {
                **entry,
                "errors": [m for m, r in result["modalities"].items() if "error" in r],
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    summary = {
        **previous,
        "sites": reports,
        "sites_with_errors": sum(bool(r["errors"]) for r in reports),
        "previous_audit": str(args.previous),
        "previous_summary_sha256": hashlib.sha256(
            (args.previous / "summary.json").read_bytes()
        ).hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
