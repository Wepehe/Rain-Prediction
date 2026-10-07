"""Immutable-download helpers with reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_immutable(
    url: str,
    destination: Path,
    manifest_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Stream a new file atomically and append its provenance; never overwrite raw data."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"raw destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                # Keep the temporary basename short.  Using the full destination name as
                # the prefix can exceed the legacy Windows MAX_PATH limit in deep,
                # event-specific raw-data directories.
                fd, temp_name = tempfile.mkstemp(prefix=".part-", dir=destination.parent)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    os.replace(temp_name, destination)
                except BaseException:
                    Path(temp_name).unlink(missing_ok=True)
                    raise
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(attempt)
    else:  # pragma: no cover - loop either succeeds or raises
        raise RuntimeError(f"download failed without an exception: {url}") from last_error
    record = {
        "source_url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "local_path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **(metadata or {}),
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record
