"""HTTP transport abstraction for the BRAIN client.

Design decisions:
- HttpTransport is a Protocol. BrainClient depends on the Protocol, not
  on requests. Tests inject a fake transport; no monkeypatching of
  requests internals. See D-071.
- Session cookies are owned by the transport. RequestsTransport uses a
  requests.Session, which persists cookies across calls. See D-072.
- The response body is returned as bytes, not pre-decoded. JSON parsing
  is the caller's concern and can fail more meaningfully.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import requests


class TransportError(RuntimeError):
    """Raised on a low-level transport failure (connection, timeout)."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Immutable snapshot of one HTTP response."""
    status_code: int
    headers: tuple[tuple[str, str], ...]   # hashable; use header_map() for dict
    body: bytes

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup; first match wins."""
        lower = name.lower()
        for k, v in self.headers:
            if k.lower() == lower:
                return v
        return None

    def header_map(self) -> dict[str, str]:
        return {k: v for k, v in self.headers}

    def json(self) -> Any:
        """Parse the body as JSON. Raises ValueError on malformed input."""
        return json.loads(self.body.decode("utf-8"))


class HttpTransport(Protocol):
    """Minimal HTTP interface used by the BRAIN client."""

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse: ...

    def close(self) -> None: ...


class RequestsTransport:
    """Requests-backed transport with persistent cookies.

    One instance per BrainClient. The underlying requests.Session
    persists cookies across calls, so login -> submit -> poll all share
    the same authenticated session.
    """

    __slots__ = ("_session",)

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session if session is not None else requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        try:
            resp = self._session.request(
                method=method,
                url=url,
                json=json_body,
                headers=headers,
                auth=auth,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise TransportError(str(e)) from e

        return HttpResponse(
            status_code=resp.status_code,
            headers=tuple(resp.headers.items()),
            body=resp.content,
        )

    def close(self) -> None:
        self._session.close()


def make_response(
    status_code: int,
    *,
    json_body: Any | None = None,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> HttpResponse:
    """Convenience constructor. Used by tests and by callers that need to
    build a response by hand."""
    if body is None:
        if json_body is None:
            body = b""
        else:
            body = json.dumps(json_body).encode("utf-8")
    header_tuples = tuple((headers or {}).items())
    return HttpResponse(
        status_code=status_code,
        headers=header_tuples,
        body=body,
    )