"""Anonymous S3 listing without requiring the AWS CLI or credentials."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str


def list_public_bucket(bucket: str, prefix: str, *, max_keys: int = 1000) -> list[S3Object]:
    params = urlencode({"list-type": "2", "prefix": prefix, "max-keys": max_keys})
    response = requests.get(f"https://{bucket}.s3.amazonaws.com/?{params}", timeout=60)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects = []
    for item in root.findall("s3:Contents", ns):
        objects.append(
            S3Object(
                key=item.findtext("s3:Key", default="", namespaces=ns),
                size=int(item.findtext("s3:Size", default="0", namespaces=ns)),
                etag=item.findtext("s3:ETag", default="", namespaces=ns).strip('"'),
            )
        )
    return objects


def object_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{key}"

