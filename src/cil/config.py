"""Centralised, typed configuration.

All paths, thresholds, horizons, dates, and secrets live here (or in the
companion TOML / environment), never hard-coded in estimators or pipelines
(per the project contract). Settings are loaded via ``pydantic-settings`` from
environment variables (prefix ``CIL_``), an optional ``.env`` file, and an
optional ``config.toml`` ``[cil]`` table.

Examples
--------
>>> from cil.config import get_settings
>>> settings = get_settings()
>>> settings.horizons.max_horizon >= 0
True
"""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HorizonConfig(BaseModel):
    """Local-projection horizon grid.

    Parameters
    ----------
    min_horizon
        Smallest (most negative) lead horizon for the event-study pre-trend
        test. Non-positive.
    max_horizon
        Largest response horizon ``H``. Non-negative.
    """

    min_horizon: int = Field(default=-6, le=0)
    max_horizon: int = Field(default=24, ge=0)

    @property
    def response_horizons(self) -> tuple[int, ...]:
        """Non-negative response horizons ``0..max_horizon`` inclusive."""
        return tuple(range(0, self.max_horizon + 1))

    @property
    def lead_horizons(self) -> tuple[int, ...]:
        """Negative lead horizons ``min_horizon..-1`` for pre-trend testing."""
        return tuple(range(self.min_horizon, 0))


class InferenceConfig(BaseModel):
    """Inference and multiple-testing parameters.

    Parameters
    ----------
    confidence_level
        Two-sided confidence level for reported intervals, in ``(0, 1)``.
    fdr_alpha
        Benjamini-Hochberg false-discovery-rate level, in ``(0, 1)``.
    bootstrap_draws
        Number of wild-cluster-bootstrap resamples.
    seed
        Master random seed; logged with every reproducible artifact.
    """

    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    fdr_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    bootstrap_draws: int = Field(default=9999, ge=1)
    seed: int = Field(default=20260101, ge=0)


class PathsConfig(BaseModel):
    """Filesystem locations for caches, the store, and outputs.

    Parameters
    ----------
    data_dir
        Root for cached raw pulls and intermediate artifacts.
    store_path
        DuckDB database holding analysis-ready panels.
    figures_dir
        Output directory for generated figures and tables.
    """

    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    store_path: Path = Field(default=_PROJECT_ROOT / "data" / "cil.duckdb")
    figures_dir: Path = Field(default=_PROJECT_ROOT / "docs" / "figures")


class SourceUrls(BaseModel):
    """Base URLs and direct file locations for external data sources.

    Parameters
    ----------
    fred_base
        FRED/ALFRED REST API base.
    qcew_area_template
        QCEW open-data area CSV template; formatted with ``year``, ``qtr``,
        and ``area`` (e.g. ``"01000"`` for Alabama, ``"US000"`` national).
    qcew_bulk_template
        QCEW annual ``by_area`` bulk-zip template (formatted with ``year``);
        used to extend history before the open-data API's 2014 floor.
    bls_flat_base
        BLS time-series flat-file root (used for CES-SAE / ``sm`` files).
    wuxia_xlsx
        Direct URL of the Atlanta Fed Wu-Xia shadow-rate workbook.
    brw_csv
        Direct URL of the Bu-Rogers-Wu shock-series CSV.
    """

    fred_base: str = "https://api.stlouisfed.org/fred"
    qcew_area_template: str = (
        "https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area}.csv"
    )
    qcew_bulk_template: str = (
        "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_by_area.zip"
    )
    bls_flat_base: str = "https://download.bls.gov/pub/time.series"
    wuxia_xlsx: str = (
        "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/"
        "datafiles/cqer/research/wu-xia-shadow-federal-funds-rate/WuXiaShadowRate.xlsx"
    )
    brw_csv: str = (
        "https://www.federalreserve.gov/econres/feds/files/brw-shock-series.csv"
    )


class MacroSeries(BaseModel):
    """FRED/ALFRED series identifiers for national and macro-confounder data.

    Parameters
    ----------
    national_employment
        Total nonfarm payroll employment (the national outcome).
    cpi, pce_price, industrial_production, unemployment, oil_price
        Macro confounders. CPI and the PCE price index proxy inflation;
        industrial production and the unemployment rate proxy real activity /
        slack; the oil price proxies a common supply shock.
    fed_funds
        Effective federal funds rate (monthly average), the conventional policy
        rate spliced with the Wu-Xia shadow rate at the ZLB.
    strict_pit
        Series for which the as-of (real-time vintage) panel is enforced and the
        point-in-time invariant is tested.
    """

    national_employment: str = "PAYEMS"
    cpi: str = "CPIAUCSL"
    pce_price: str = "PCEPI"
    industrial_production: str = "INDPRO"
    unemployment: str = "UNRATE"
    oil_price: str = "MCOILWTICO"
    fed_funds: str = "FEDFUNDS"
    strict_pit: tuple[str, ...] = (
        "PAYEMS",
        "CPIAUCSL",
        "PCEPI",
        "INDPRO",
        "UNRATE",
    )

    @property
    def all_ids(self) -> tuple[str, ...]:
        """All distinct FRED series identifiers referenced by this config."""
        ids = (
            self.national_employment,
            self.cpi,
            self.pce_price,
            self.industrial_production,
            self.unemployment,
            self.oil_price,
            self.fed_funds,
        )
        return tuple(dict.fromkeys(ids))


class QcewConfig(BaseModel):
    """QCEW state-by-supersector extraction parameters.

    Parameters
    ----------
    aggregation_level
        QCEW ``agglvl_code``. ``53`` selects "State, by NAICS Supersector"
        (~11 supersectors), minimising disclosure suppression relative to finer
        NAICS levels.
    ownership_code
        QCEW ``own_code``. ``0`` selects "Total Covered" employment.
    coverage_min_fraction
        Minimum fraction of in-sample months a (state, supersector) cell must be
        non-suppressed and positive to be retained; cells below this are flagged
        and dropped. In ``[0, 1]``.
    api_min_year
        Earliest year served by the QCEW open-data API (2014). Years from this
        year onward are pulled via the API; earlier years come from the bulk
        flat-file path.
    bulk_min_year
        Earliest year pulled from the QCEW ``by_area`` bulk flat files (NAICS
        reconstructed back to 1990). Extends history before ``api_min_year``.
    """

    aggregation_level: int = 55
    ownership_code: int = 0
    coverage_min_fraction: float = Field(default=0.90, ge=0.0, le=1.0)
    api_min_year: int = 2014
    bulk_min_year: int = 1990


class ZlbWindow(BaseModel):
    """A zero-lower-bound window over which the shadow rate is spliced in.

    Parameters
    ----------
    start, end
        Inclusive month bounds (first day of month) of the ZLB window.
    """

    start: dt.date
    end: dt.date


class SampleConfig(BaseModel):
    """Sample window and structural-break dates.

    Parameters
    ----------
    start, end
        Inclusive bounds of the analysis sample (first day of month). Ingestion
        pulls the full available history; windowing is applied downstream.
    zlb_windows
        Windows over which the Wu-Xia shadow rate replaces the EFFR.
    """

    start: dt.date = dt.date(1994, 1, 1)
    end: dt.date = dt.date(2020, 12, 1)
    zlb_windows: tuple[ZlbWindow, ...] = (
        ZlbWindow(start=dt.date(2008, 12, 1), end=dt.date(2015, 12, 1)),
        ZlbWindow(start=dt.date(2020, 3, 1), end=dt.date(2022, 2, 1)),
    )


class DataConfig(BaseModel):
    """Data-layer configuration (sources, series, sample, QCEW parameters).

    Parameters
    ----------
    urls
        External source locations.
    series
        FRED/ALFRED series identifiers.
    qcew
        QCEW extraction parameters.
    sample
        Sample window and ZLB splice windows.
    contact_email
        Contact address sent in the ``User-Agent`` for BLS requests, per BLS
        access policy.
    request_timeout_seconds
        Per-request timeout for external pulls.
    """

    urls: SourceUrls = Field(default_factory=SourceUrls)
    series: MacroSeries = Field(default_factory=MacroSeries)
    qcew: QcewConfig = Field(default_factory=QcewConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    contact_email: str = "research@example.com"
    request_timeout_seconds: float = Field(default=60.0, gt=0.0)


class ShocksConfig(BaseModel):
    """Monetary-shock identification parameters.

    Parameters
    ----------
    equity_series_id
        FRED series for the broad-equity gauge used in the information-effect
        test (NASDAQ Composite as a long-history broad proxy).
    rr_lags
        Number of monthly lags of the real-time information set included in the
        Romer-Romer orthogonalization regression.
    predictability_lags
        Number of lags of predictors used in the predictability (Bauer-Swanson)
        test.
    svar_lags
        Lag order of the reduced-form VAR underlying the proxy-SVAR.
    weak_iv_f_threshold
        First-stage F below which the external instrument is flagged weak
        (Montiel Olea-Pflueger effective-F guidance; ~10 rule of thumb).
    instrument
        Which series instruments the policy equation in the proxy-SVAR.
    """

    equity_series_id: str = "NASDAQCOM"
    rr_lags: int = Field(default=2, ge=0)
    predictability_lags: int = Field(default=3, ge=1)
    svar_lags: int = Field(default=12, ge=1)
    weak_iv_f_threshold: float = Field(default=10.0, gt=0.0)
    instrument: str = "brw_monthly"


class Settings(BaseSettings):
    """Top-level project settings.

    Parameters
    ----------
    fred_api_key
        FRED/ALFRED API key. Read from the environment; never committed.
    use_shadow_rate
        Whether to splice the Wu-Xia shadow rate during ZLB windows.
        Gated behind this flag by contract.
    horizons
        Local-projection horizon grid.
    inference
        Inference and multiple-testing parameters.
    paths
        Filesystem locations.
    data
        Data-layer configuration (sources, series, sample, QCEW parameters).
    shocks
        Monetary-shock identification parameters.
    """

    model_config = SettingsConfigDict(
        env_prefix="CIL_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    fred_api_key: str | None = None
    use_shadow_rate: bool = True
    horizons: HorizonConfig = Field(default_factory=HorizonConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    shocks: ShocksConfig = Field(default_factory=ShocksConfig)

    @field_validator("horizons")
    @classmethod
    def _check_horizon_order(cls, value: HorizonConfig) -> HorizonConfig:
        if value.max_horizon < 0 or value.min_horizon > 0:
            msg = "Require min_horizon <= 0 <= max_horizon."
            raise ValueError(msg)
        return value


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    Returns
    -------
    Settings
        The resolved settings, constructed once and memoised.
    """
    return Settings()
