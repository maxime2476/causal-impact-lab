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
