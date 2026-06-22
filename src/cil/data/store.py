"""DuckDB-backed store for analysis-ready tables and provenance.

A thin wrapper over a DuckDB connection. Polars frames are written and read
back; provenance records accumulate in a dedicated table so every analysis
table can be traced to its source pulls.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Literal

import duckdb
import polars as pl

from cil.data.provenance import Provenance

WriteMode = Literal["replace", "append"]

_PROVENANCE_DDL = """
CREATE TABLE IF NOT EXISTS provenance (
    source        VARCHAR,
    url           VARCHAR,
    retrieved_at  TIMESTAMP,
    vintage_date  DATE,
    params        VARCHAR,
    sha256        VARCHAR,
    n_bytes       BIGINT,
    cache_path    VARCHAR
)
"""


class Store:
    """A DuckDB connection holding analysis-ready tables and provenance.

    Parameters
    ----------
    path
        Filesystem path to the DuckDB database. Parent directories are created.
        Use ``":memory:"`` for an ephemeral in-memory store (tests).
    """

    def __init__(self, path: Path | str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._con.execute(_PROVENANCE_DDL)

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        self._con.close()

    def write_table(
        self, name: str, df: pl.DataFrame, *, mode: WriteMode = "replace"
    ) -> int:
        """Write a polars frame as a table and return the row count.

        Parameters
        ----------
        name
            Destination table name (a SQL identifier).
        df
            The frame to write.
        mode
            ``"replace"`` overwrites any existing table; ``"append"`` inserts
            into an existing table (created if absent).

        Returns
        -------
        int
            Number of rows in *df*.
        """
        ident = _safe_identifier(name)
        self._con.register("_cil_write", df)
        try:
            if mode == "replace":
                self._con.execute(
                    f"CREATE OR REPLACE TABLE {ident} AS SELECT * FROM _cil_write"
                )
            else:
                self._con.execute(
                    f"CREATE TABLE IF NOT EXISTS {ident} AS "
                    "SELECT * FROM _cil_write WHERE FALSE"
                )
                self._con.execute(f"INSERT INTO {ident} SELECT * FROM _cil_write")
        finally:
            self._con.unregister("_cil_write")
        return df.height

    def read_table(self, name: str) -> pl.DataFrame:
        """Read a table back as a polars frame.

        Parameters
        ----------
        name
            Table name to read.

        Returns
        -------
        polars.DataFrame
            The table contents.
        """
        ident = _safe_identifier(name)
        return self._con.execute(f"SELECT * FROM {ident}").pl()

    def table_exists(self, name: str) -> bool:
        """Return whether a table of the given name exists."""
        rows = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchall()
        return len(rows) > 0

    def list_tables(self) -> list[str]:
        """Return the names of all tables in the store, sorted."""
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def record_provenance(self, prov: Provenance) -> None:
        """Append a provenance record.

        Parameters
        ----------
        prov
            The provenance record to persist.
        """
        params_json = "&".join(f"{k}={v}" for k, v in sorted(prov.params.items()))
        self._con.execute(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                prov.source,
                prov.url,
                prov.retrieved_at,
                prov.vintage_date,
                params_json,
                prov.sha256,
                prov.n_bytes,
                prov.cache_path,
            ],
        )

    def read_provenance(self) -> pl.DataFrame:
        """Return all provenance records as a polars frame."""
        return self._con.execute("SELECT * FROM provenance").pl()


def _safe_identifier(name: str) -> str:
    """Validate *name* as a simple SQL identifier and return it quoted.

    Parameters
    ----------
    name
        Candidate table name.

    Returns
    -------
    str
        The double-quoted identifier.

    Raises
    ------
    ValueError
        If *name* is not a plain ``[A-Za-z_][A-Za-z0-9_]*`` identifier.
    """
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        msg = f"Unsafe table identifier: {name!r}"
        raise ValueError(msg)
    return f'"{name}"'
