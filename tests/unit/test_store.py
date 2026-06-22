"""DuckDB store: round-trip, append, provenance, and identifier safety."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from cil.data.provenance import Provenance, cache_raw
from cil.data.store import Store


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def test_write_read_round_trip() -> None:
    with Store(":memory:") as store:
        assert store.write_table("t", _frame()) == 2
        assert store.read_table("t").to_dicts() == _frame().to_dicts()
        assert store.table_exists("t")
        assert "t" in store.list_tables()


def test_append_mode() -> None:
    with Store(":memory:") as store:
        store.write_table("t", _frame())
        store.write_table("t", pl.DataFrame({"a": [3], "b": ["z"]}), mode="append")
        assert store.read_table("t").height == 3


def test_replace_mode_overwrites() -> None:
    with Store(":memory:") as store:
        store.write_table("t", _frame())
        store.write_table("t", pl.DataFrame({"a": [9], "b": ["q"]}), mode="replace")
        assert store.read_table("t").height == 1


def test_unsafe_identifier_rejected() -> None:
    with Store(":memory:") as store, pytest.raises(ValueError, match="identifier"):
        store.read_table("t; DROP TABLE x")


def test_provenance_round_trip() -> None:
    prov = Provenance(
        source="alfred",
        url="https://example.test/x",
        retrieved_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        vintage_date=dt.date(2026, 1, 1),
        params={"series_id": "PAYEMS"},
        sha256="deadbeef",
        n_bytes=4,
        cache_path="raw/alfred/x.json",
    )
    with Store(":memory:") as store:
        store.record_provenance(prov)
        out = store.read_provenance()
        assert out.height == 1
        assert out["source"].to_list() == ["alfred"]


def test_cache_raw_writes_payload(tmp_path: object) -> None:
    from pathlib import Path

    data_dir = Path(str(tmp_path))
    prov = cache_raw(data_dir, "brw", "x.csv", b"hello", "https://example.test/x.csv")
    assert (data_dir / prov.cache_path).read_bytes() == b"hello"
    assert prov.n_bytes == 5
    assert prov.source == "brw"
