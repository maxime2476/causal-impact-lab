"""Refresh path: force_refresh re-fetches, otherwise the cache is reused."""

from __future__ import annotations

from pathlib import Path

from cil.data.pipeline import _cache_or_fetch
from cil.data.store import Store


def test_force_refresh_controls_refetch(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fetch() -> tuple[bytes, str, dict[str, str]]:
        calls["n"] += 1
        return (f"payload-{calls['n']}".encode(), "http://example/data", {})

    with Store(":memory:") as store:
        # No cache yet -> fetch and cache.
        first = _cache_or_fetch(store, tmp_path, "src", "f.bin", fetch)
        assert calls["n"] == 1
        assert first == b"payload-1"

        # Cache present, no refresh -> serve from cache (no fetch).
        cached = _cache_or_fetch(store, tmp_path, "src", "f.bin", fetch)
        assert calls["n"] == 1
        assert cached == first

        # force_refresh -> re-fetch and overwrite the cache with new data.
        refreshed = _cache_or_fetch(
            store, tmp_path, "src", "f.bin", fetch, force_refresh=True
        )
        assert calls["n"] == 2
        assert refreshed == b"payload-2"
