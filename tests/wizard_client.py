"""Requests that carry the wizard's token, as every wizard route needs."""

from __future__ import annotations

from http.client import HTTPResponse
from typing import cast
from urllib.request import Request, urlopen

from super_processor.review import TOKEN_HEADER


def auth(token: str | None, **headers: str) -> dict[str, str]:
    """``headers`` plus the token header (urllib keeps it across redirects)."""
    return {TOKEN_HEADER: token or "", **headers}


def get(url: str, token: str | None, **headers: str) -> HTTPResponse:
    """``urlopen`` of ``url`` with the token; use it as a context manager."""
    return cast(HTTPResponse, urlopen(Request(url, headers=auth(token, **headers))))
