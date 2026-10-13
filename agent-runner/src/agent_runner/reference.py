from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


AUDIT_RELATIVE_PATH = Path("environment/data/source_allowlist_manifest.json")
RUNTIME_RELATIVE_PATH = Path("public/reference/source-allowlist.json")
RUNTIME_SCHEMA = "biomedicine-source-allowlist/v1"
AUDIT_SCHEMA = "biomedicine-source-audit/v1"
HOST = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\Z")
EXPECTED_BIOMEDICAL_HOSTS = {
    "api.fda.gov", "api.geneontology.org", "api.ncbi.nlm.nih.gov",
    "api.platform.opentargets.org", "biit.cs.ut.ee", "clinicaltrials.gov",
    "eutils.ncbi.nlm.nih.gov", "maayanlab.cloud", "mychem.info",
    "mydisease.info", "mygene.info", "myvariant.info", "ontology.jax.org",
    "open-genes.com", "pubchem.ncbi.nlm.nih.gov", "reactome.org",
    "rest.ensembl.org", "rest.kegg.jp", "rest.uniprot.org", "www.ebi.ac.uk",
    "www.ncbi.nlm.nih.gov", "www.proteinatlas.org",
}
CODEX_CONTROL_PLANE_HOSTS = {"api.openai.com", "auth.openai.com", "chatgpt.com"}
HARNESS_GATEWAY_HOST = "provider-egress-control"


def with_harness_gateway_host(hosts: list[str] | set[str]) -> list[str]:
    """Allow the verifier to reach only the task's internal gateway endpoint."""
    return sorted(set(hosts) | {HARNESS_GATEWAY_HOST})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{description} is missing: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unreadable or invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _exact_hostname(value: object, field: str) -> str:
    if not isinstance(value, str) or value != value.lower() or not HOST.fullmatch(value):
        raise ValueError(f"{field} must be one exact lowercase hostname")
    if "*" in value:
        raise ValueError(f"{field} cannot contain a wildcard")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    raise ValueError(f"{field} cannot be an IP literal")


def expected_allowlist(audit: dict) -> dict:
    inventory = audit.get("inventory")
    network = audit.get("network")
    sources = audit.get("sources")
    if (audit.get("schema") != AUDIT_SCHEMA or not isinstance(inventory, dict) or
            set(inventory) != {"repository", "revision", "tools_path"} or
            inventory.get("repository") != "https://github.com/RyanWangZf/BioDSA" or
            not re.fullmatch(r"[0-9a-f]{40}", str(inventory.get("revision", ""))) or
            inventory.get("tools_path") != "biodsa/tools" or
            network != {"transport": "tcp", "ports": "all"} or
            not isinstance(sources, list) or not sources):
        raise ValueError("invalid biomedical source audit manifest")
    names: set[str] = set()
    allowed: set[str] = set()
    normalized_sources: list[dict] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {"name", "base_urls", "hosts"}:
            raise ValueError(f"invalid biomedical source entry at index {index}")
        name = source["name"]
        urls = source["base_urls"]
        raw_hosts = source["hosts"]
        if (not isinstance(name, str) or not name or name in names or
                not isinstance(urls, list) or not urls or
                not isinstance(raw_hosts, list) or not raw_hosts):
            raise ValueError(f"invalid or duplicate biomedical source at index {index}")
        names.add(name)
        hosts = [_exact_hostname(host, f"sources[{index}].hosts") for host in raw_hosts]
        if len(hosts) != len(set(hosts)):
            raise ValueError(f"duplicate host in biomedical source {name!r}")
        for url in urls:
            if not isinstance(url, str):
                raise ValueError(f"invalid base URL in biomedical source {name!r}")
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.username or parsed.password or
                    parsed.hostname not in hosts):
                raise ValueError(f"invalid base URL in biomedical source {name!r}: {url!r}")
        allowed.update(hosts)
        normalized_sources.append({"name": name, "base_urls": urls, "hosts": hosts})
    if allowed != EXPECTED_BIOMEDICAL_HOSTS:
        raise ValueError("biomedical source audit does not contain the approved 22-host set")
    return {
        "schema": RUNTIME_SCHEMA,
        "inventory": {"repository": inventory["repository"],
                      "revision": inventory["revision"]},
        "network": {"transport": "tcp", "ports": "all",
                    "allowed_hosts": sorted(allowed)},
        "sources": normalized_sources,
    }


def load_reference_policy(task_dir: Path, data_dir: Path | None) -> dict | None:
    """Fail closed for tasks that declare a task-owned biomedical source audit."""
    audit_path = task_dir / AUDIT_RELATIVE_PATH
    if not audit_path.is_file():
        return None
    if data_dir is None:
        raise FileNotFoundError(
            "This task version requires newly materialized public/reference data; "
            "provide --data-dir and run materialize explicitly"
        )
    runtime_path = data_dir / RUNTIME_RELATIVE_PATH
    audit = _json_object(audit_path, "biomedical source audit manifest")
    expected = expected_allowlist(audit)
    if not runtime_path.is_file():
        raise FileNotFoundError(
            "materialized public/reference data is missing for this task version; "
            "run materialize explicitly to refresh the data directory"
        )
    runtime = _json_object(runtime_path, "materialized biomedical source allowlist")
    if runtime != expected:
        raise ValueError(
            "materialized biomedical source allowlist does not match the task audit manifest; "
            "run materialize explicitly to refresh this task version"
        )
    risk_spec = audit.get("riskcalcs")
    if (not isinstance(risk_spec, dict) or
            risk_spec.get("repository") !=
            "https://github.com/ncbi-nlp/Clinical-Tool-Learning" or
            risk_spec.get("revision") !=
            "1c64a637b8eecce887cb4b1ee97c8429cdfbfade" or
            risk_spec.get("path") != "riskqa_evaluation/tools/riskcalcs.json" or
            risk_spec.get("size") != 5664374 or
            risk_spec.get("sha256") !=
            "00a7a0089afffb66f2f32903bad94a5a2ea842841defb2d78e0686b0a5eb9ab9"):
        raise ValueError("invalid or unpinned RiskCalcs audit entry")
    riskcalcs_path = runtime_path.parent / "riskcalcs.json"
    if (not riskcalcs_path.is_file() or riskcalcs_path.is_symlink() or
            riskcalcs_path.stat().st_size != risk_spec["size"] or
            _sha256(riskcalcs_path) != risk_spec["sha256"]):
        raise ValueError("materialized RiskCalcs is missing or has failed its checksum")
    digest = _sha256(runtime_path)
    for view in ("development", "verifier"):
        manifest_path = data_dir / "public" / view / "manifest.json"
        manifest = _json_object(manifest_path, f"materialized {view} manifest")
        if manifest.get("source_allowlist_sha256") != digest:
            raise ValueError(
                f"materialized {view} manifest has a missing or stale source allowlist hash"
            )
    return {
        "revision": runtime["inventory"]["revision"],
        "sha256": digest,
        "hosts": runtime["network"]["allowed_hosts"],
        "reference_dir": str(runtime_path.parent.resolve()),
    }


def provider_endpoint_hostname(provider: str | None, runtime_env: dict[str, str]) -> str | None:
    """Resolve and validate the one provider endpoint selected by the profile."""
    if not provider or runtime_env.get("BIOMED_GATEWAY_ONLINE", "0") != "1":
        return None
    if provider == "openai":
        url = (runtime_env.get("OPENAI_API_BASE") or
               runtime_env.get("OPENAI_BASE_URL") or "https://api.openai.com")
    elif provider == "azure":
        url = runtime_env.get("AZURE_API_BASE")
        if not url:
            raise ValueError("online Azure profile requires AZURE_API_BASE")
    elif provider == "gemini":
        url = "https://generativelanguage.googleapis.com"
    elif provider in {"claude", "anthropic"}:
        url = "https://api.anthropic.com"
    else:
        raise ValueError(f"unsupported harness LLM provider: {provider}")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("provider base URL must be HTTPS and contain no credentials")
    return _exact_hostname(parsed.hostname.lower(), "provider endpoint")
