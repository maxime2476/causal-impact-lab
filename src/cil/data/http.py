"""Shared HTTP access for data ingestion.

Provides a configured :class:`httpx.Client` with a descriptive ``User-Agent``
(BLS access policy asks for a contact address) and a small retry loop for
transient failures. No look-ahead or caching logic lives here; that is the
caller's concern.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def build_user_agent(contact_email: str) -> str:
    """Return a ``User-Agent`` string identifying the project and a contact.

    Parameters
    ----------
    contact_email
        Address included so data providers (notably BLS) can reach the operator.

    Returns
    -------
    str
        The ``User-Agent`` header value.
    """
    return f"causal-impact-lab/0.0 (research; {contact_email})"


def build_client(contact_email: str, timeout_seconds: float) -> httpx.Client:
    """Return an :class:`httpx.Client` configured for data pulls.

    Parameters
    ----------
    contact_email
        Contact address for the ``User-Agent`` header.
    timeout_seconds
        Per-request timeout.

    Returns
    -------
    httpx.Client
        A client following redirects with the project ``User-Agent`` set.
    """
    return httpx.Client(
        headers={"User-Agent": build_user_agent(contact_email)},
        timeout=timeout_seconds,
        follow_redirects=True,
    )


def fetch(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, str | int] | None = None,
    max_attempts: int = 4,
    backoff_seconds: float = 1.5,
) -> httpx.Response:
    """GET *url* with bounded retries on transient errors.

    Parameters
    ----------
    client
        The HTTP client to use.
    url
        Absolute URL to request.
    params
        Optional query parameters.
    max_attempts
        Maximum number of attempts before giving up.
    backoff_seconds
        Base of the exponential backoff between attempts.

    Returns
    -------
    httpx.Response
        The successful response (status < 400).

    Raises
    ------
    httpx.HTTPStatusError
        If a non-retryable error status is returned, or attempts are exhausted.
    httpx.TransportError
        If transport-level failures persist across all attempts.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code in _RETRYABLE_STATUS:
                response.raise_for_status()
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, httpx.TransportError) or (
                status in _RETRYABLE_STATUS
            )
            if not retryable or attempt == max_attempts:
                raise
            time.sleep(backoff_seconds ** (attempt - 1))
    # Unreachable: the loop either returns or raises.
    raise RuntimeError("fetch exhausted retries") from last_exc
