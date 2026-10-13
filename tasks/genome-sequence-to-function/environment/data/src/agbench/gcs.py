"""Shared GCS object identity helpers for the benchmark bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.parse import urlencode
from urllib.parse import quote
import urllib.request

from .constants import SOURCE_BUCKET
from .errors import ContractError

_SHARD_RE = re.compile(r"data_(?P<chromosome>.+)_(?P<shard>\d+)-(?P<count>\d+)\.gz\.tfrecord$")


@dataclass(frozen=True)
class ObjectInfo:
    bundle: str
    split: str
    chromosome: str
    shard: int
    shard_count: int
    name: str
    size: int
    generation: str
    md5_base64: str
    crc32c_base64: str

    @property
    def gs_uri(self) -> str:
        return f"gs://{SOURCE_BUCKET}/{self.name}"

    @property
    def pair_key(self) -> tuple[str, int, int]:
        return (self.chromosome, self.shard, self.shard_count)


def _list_prefix(prefix: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    page_token: str | None = None
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = f"https://storage.googleapis.com/storage/v1/b/{SOURCE_BUCKET}/o?{urlencode(query)}"
        with urllib.request.urlopen(url) as response:
            payload = json.load(response)
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return items


def verify_object_current(item: ObjectInfo) -> None:
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{SOURCE_BUCKET}/o/"
        f"{quote(item.name, safe='')}"
    )
    with urllib.request.urlopen(url) as response:
        observed = json.load(response)
    identity = (
        observed.get("generation"),
        int(observed.get("size", -1)),
        observed.get("md5Hash", ""),
        observed.get("crc32c", ""),
    )
    expected = (item.generation, item.size, item.md5_base64, item.crc32c_base64)
    if identity != expected:
        raise ContractError(
            f"GCS object changed after inventory: {item.name}; "
            f"observed={identity}, expected={expected}"
        )
