"""Schema contracts for dataset boundaries.

Every frame crossing a boundary (raw -> typed -> analysis-ready) is validated
against one of these :class:`pandera.polars.DataFrameSchema` contracts. A
contract violation is a hard error, surfaced at the boundary rather than
silently propagated downstream.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.polars import Column, DataFrameSchema

_NONEMPTY = pa.Check.str_length(min_value=1)

#: Point-in-time macro series: one row per (series, reference period, vintage).
MACRO_PIT_SCHEMA = DataFrameSchema(
    {
        "series_id": Column(pl.Utf8, checks=_NONEMPTY),
        "reference_date": Column(pl.Date),
        "vintage_date": Column(pl.Date),
        "value": Column(pl.Float64, nullable=True),
    },
    unique=["series_id", "reference_date", "vintage_date"],
    strict=True,
    coerce=True,
)

#: Latest (as-published) macro series: one row per (series, reference period).
MACRO_CURRENT_SCHEMA = DataFrameSchema(
    {
        "series_id": Column(pl.Utf8, checks=_NONEMPTY),
        "reference_date": Column(pl.Date),
        "value": Column(pl.Float64, nullable=True),
    },
    unique=["series_id", "reference_date"],
    strict=True,
    coerce=True,
)

#: Spliced monthly policy rate: EFFR with the Wu-Xia shadow rate at the ZLB.
POLICY_RATE_SCHEMA = DataFrameSchema(
    {
        "date": Column(pl.Date),
        "effr": Column(pl.Float64, nullable=True),
        "shadow_rate": Column(pl.Float64, nullable=True),
        "policy_rate": Column(pl.Float64, nullable=True),
        "is_zlb_splice": Column(pl.Boolean),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

#: Bu-Rogers-Wu monthly shock series.
BRW_SHOCK_SCHEMA = DataFrameSchema(
    {
        "date": Column(pl.Date),
        "brw_monthly": Column(pl.Float64, nullable=True),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

#: Bauer-Swanson monthly monetary policy surprises (raw + orthogonalized).
MPS_MONTHLY_SCHEMA = DataFrameSchema(
    {
        "date": Column(pl.Date),
        "mps": Column(pl.Float64, nullable=True),
        "mps_orth": Column(pl.Float64, nullable=True),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

#: Bauer-Swanson per-FOMC surprises with the same-window S&P 500 change.
MPS_FOMC_SCHEMA = DataFrameSchema(
    {
        "date": Column(pl.Date),
        "mps": Column(pl.Float64, nullable=True),
        "mps_orth": Column(pl.Float64, nullable=True),
        "sp500": Column(pl.Float64, nullable=True),
    },
    strict=True,
    coerce=True,
)

#: Updated Romer-Romer (2004) quarterly narrative shocks (three vintages).
RR_SHOCKS_SCHEMA = DataFrameSchema(
    {
        "date": Column(pl.Date),
        "rr_org": Column(pl.Float64, nullable=True),
        "rr08": Column(pl.Float64, nullable=True),
        "rr12": Column(pl.Float64, nullable=True),
    },
    unique=["date"],
    strict=True,
    coerce=True,
)

#: QCEW state-by-supersector monthly employment, with a suppression flag.
QCEW_CELL_SCHEMA = DataFrameSchema(
    {
        "state_fips": Column(pl.Utf8, checks=pa.Check.str_length(2, 2)),
        "supersector_code": Column(pl.Utf8, checks=_NONEMPTY),
        "date": Column(pl.Date),
        "employment": Column(pl.Float64, nullable=True, checks=pa.Check.ge(0)),
        "suppressed": Column(pl.Boolean),
    },
    unique=["state_fips", "supersector_code", "date"],
    strict=True,
    coerce=True,
)

#: BLS CES-SAE state-by-supersector monthly employment (cross-check source).
CES_SAE_SCHEMA = DataFrameSchema(
    {
        "state_fips": Column(pl.Utf8, checks=pa.Check.str_length(2, 2)),
        "supersector_code": Column(pl.Utf8, checks=_NONEMPTY),
        "date": Column(pl.Date),
        "employment": Column(pl.Float64, nullable=True, checks=pa.Check.ge(0)),
        "seasonal": Column(pl.Utf8, checks=pa.Check.isin(["S", "U"])),
    },
    unique=["state_fips", "supersector_code", "date", "seasonal"],
    strict=True,
    coerce=True,
)

#: Analysis-ready cell panel: log employment per (cell, month).
PANEL_CELL_SCHEMA = DataFrameSchema(
    {
        "unit_id": Column(pl.Utf8, checks=_NONEMPTY),
        "state_fips": Column(pl.Utf8, checks=pa.Check.str_length(2, 2)),
        "supersector_code": Column(pl.Utf8, checks=_NONEMPTY),
        "date": Column(pl.Date),
        "employment": Column(pl.Float64, checks=pa.Check.gt(0)),
        "log_employment": Column(pl.Float64),
    },
    unique=["unit_id", "date"],
    strict=True,
    coerce=True,
)


def validate(schema: DataFrameSchema, df: pl.DataFrame) -> pl.DataFrame:
    """Validate *df* against *schema*, returning the (coerced) frame.

    Parameters
    ----------
    schema
        The contract to enforce.
    df
        The frame to validate.

    Returns
    -------
    polars.DataFrame
        The validated, dtype-coerced frame.

    Raises
    ------
    pandera.errors.SchemaError
        If the frame violates the contract.
    """
    return schema.validate(df, lazy=False)
