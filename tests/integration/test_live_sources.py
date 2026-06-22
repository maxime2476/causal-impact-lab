"""Live integration checks against external data providers.

Marked ``integration`` and excluded from the default run (no network in unit
tests). Run explicitly with::

    uv run pytest -m integration

ALFRED/CES checks require ``CIL_FRED_API_KEY``; they skip if it is absent.
"""

from __future__ import annotations

import httpx
import pytest

from cil.config import get_settings
from cil.data import alfred, brw, http, qcew, wuxia

pytestmark = pytest.mark.integration


def _client() -> httpx.Client:
    settings = get_settings()
    return http.build_client(
        settings.data.contact_email, settings.data.request_timeout_seconds
    )


def test_alfred_pit_live() -> None:
    settings = get_settings()
    if settings.fred_api_key is None:
        pytest.skip("CIL_FRED_API_KEY not set")
    client = _client()
    try:
        content, _, _ = alfred.fetch_raw_series(
            client, settings.data.urls.fred_base, settings.fred_api_key, "PAYEMS"
        )
    finally:
        client.close()
    pit = alfred.parse_pit(content, "PAYEMS")
    assert pit.height > 1000
    assert pit["vintage_date"].n_unique() > 1


def test_qcew_industry_live() -> None:
    settings = get_settings()
    client = _client()
    try:
        content, _, _ = qcew.fetch_raw_industry(
            client, settings.data.urls.qcew_area_template, "1013", 2019, 1
        )
    finally:
        client.close()
    cells = qcew.parse_industry(content, 2019, 1)
    assert cells["state_fips"].n_unique() >= 50


def test_wuxia_and_brw_live() -> None:
    settings = get_settings()
    client = _client()
    try:
        wx_content, _, _ = wuxia.fetch_raw(client, settings.data.urls.wuxia_xlsx)
        brw_content, _, _ = brw.fetch_raw(client, settings.data.urls.brw_csv)
    finally:
        client.close()
    assert wuxia.parse(wx_content).height > 100
    assert brw.parse(brw_content).height > 100
