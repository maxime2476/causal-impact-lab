"""HTTP helpers: User-Agent and secret redaction."""

from __future__ import annotations

from cil.data import http


def test_user_agent_includes_contact() -> None:
    ua = http.build_user_agent("person@example.test")
    assert "causal-impact-lab" in ua
    assert "person@example.test" in ua


def test_redact_secrets_masks_api_key() -> None:
    url = "https://api.stlouisfed.org/fred/series?series_id=PAYEMS&api_key=abc123secret&x=1"
    redacted = http.redact_secrets(url)
    assert "abc123secret" not in redacted
    assert "api_key=REDACTED" in redacted
    assert "series_id=PAYEMS" in redacted
