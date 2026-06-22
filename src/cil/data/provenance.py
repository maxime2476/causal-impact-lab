"""Provenance records and raw-payload caching.

Every external pull is recorded with its source, URL, retrieval timestamp, and
(where applicable) the data vintage date, plus a content hash. Raw payloads are
cached to disk so analysis-ready tables are reproducible and auditable without
re-hitting providers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Audit record for a single retrieved payload.

    Parameters
    ----------
    source
        Logical source name (e.g. ``"alfred"``, ``"qcew"``).
    url
        The exact URL requested.
    retrieved_at
        UTC timestamp of retrieval.
    vintage_date
        Data vintage / real-time date where the source is point-in-time;
        ``None`` for as-published sources.
    params
        Query parameters or selection keys, serialised for the record.
    sha256
        Hex SHA-256 of the raw payload bytes.
    n_bytes
        Size of the raw payload in bytes.
    cache_path
        Path to the cached raw payload, relative to the data root.
    """

    source: str
    url: str
    retrieved_at: dt.datetime
    vintage_date: dt.date | None = None
    params: dict[str, str] = Field(default_factory=dict)
    sha256: str
    n_bytes: int
    cache_path: str


def sha256_hex(content: bytes) -> str:
    """Return the hex SHA-256 digest of *content*."""
    return hashlib.sha256(content).hexdigest()


def cache_raw(
    data_dir: Path,
    source: str,
    filename: str,
    content: bytes,
    url: str,
    *,
    vintage_date: dt.date | None = None,
    params: dict[str, str] | None = None,
) -> Provenance:
    """Write *content* to the raw cache and return its :class:`Provenance`.

    Parameters
    ----------
    data_dir
        Data root; raw payloads are stored under ``<data_dir>/raw/<source>/``.
    source
        Logical source name.
    filename
        File name for the cached payload within the source directory.
    content
        Raw payload bytes.
    url
        The URL the payload was retrieved from.
    vintage_date
        Vintage / real-time date for point-in-time sources.
    params
        Selection keys to record.

    Returns
    -------
    Provenance
        The provenance record for the cached payload.
    """
    raw_dir = data_dir / "raw" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / filename
    target.write_bytes(content)
    prov = Provenance(
        source=source,
        url=url,
        retrieved_at=dt.datetime.now(dt.UTC),
        vintage_date=vintage_date,
        params=params or {},
        sha256=sha256_hex(content),
        n_bytes=len(content),
        cache_path=str(target.relative_to(data_dir)),
    )
    _sidecar(target).write_text(prov.model_dump_json(), encoding="utf-8")
    return prov


def _sidecar(target: Path) -> Path:
    """Return the provenance sidecar path for a cached payload."""
    return target.with_suffix(target.suffix + ".prov.json")


def load_provenance(data_dir: Path, source: str, filename: str) -> Provenance | None:
    """Load the provenance sidecar for a cached payload, if it exists.

    Parameters
    ----------
    data_dir
        Data root.
    source
        Logical source name.
    filename
        Cached payload file name within the source directory.

    Returns
    -------
    Provenance or None
        The recorded provenance, or ``None`` if no sidecar is present.
    """
    sidecar = _sidecar(data_dir / "raw" / source / filename)
    if not sidecar.exists():
        return None
    return Provenance.model_validate_json(sidecar.read_text(encoding="utf-8"))
