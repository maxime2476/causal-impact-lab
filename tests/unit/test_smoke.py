"""Smoke tests: the package imports and configuration is internally consistent."""

from __future__ import annotations

import cil
from cil.config import Settings, get_settings


def test_version_is_exposed() -> None:
    assert isinstance(cil.__version__, str)
    assert cil.__version__


def test_settings_construct_with_defaults() -> None:
    settings = Settings()
    assert settings.horizons.min_horizon <= 0 <= settings.horizons.max_horizon
    assert 0.0 < settings.inference.confidence_level < 1.0
    assert settings.use_shadow_rate is True


def test_horizon_grids_are_consistent() -> None:
    settings = Settings()
    assert settings.horizons.response_horizons[0] == 0
    assert settings.horizons.response_horizons[-1] == settings.horizons.max_horizon
    assert all(h < 0 for h in settings.horizons.lead_horizons)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
