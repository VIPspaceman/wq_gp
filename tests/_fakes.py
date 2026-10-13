"""Shared test doubles.

ScriptedTransport was extracted from tests/test_brain_client.py so
tests/test_pool_builder.py and tests/test_notification.py can use the
same fake without duplicating the class.

URL matching uses path.endswith(prefix), NOT path.startswith(prefix).
Rationale: BRAIN URLs put the identifying segment at the end of the
path (/authentication, /simulations/123), and Telegram URLs put a
variable bot-token segment in the middle (/bot<TOKEN>/sendMessage).
Suffix matching handles both. See D-099.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from wqgp.brain.transport import HttpResponse


@dataclass
class ScriptedTransport:
    """Test double for HttpTransport.

    Register responses in FIFO order per (method, url-path-suffix).
    Every request pops the next queued response. Records every call.

    Suffixes are URL PATHS ("/sendMessage"), matched against the end of
    the request's path. See D-099.
    """
    calls: list[tuple[str, str, dict[str, Any]]] = field(
        default_factory=list
    )
    _queues: dict[tuple[str, str], list[Any]] = field(default_factory=dict)

    def expect(self, method: str, url_suffix: str, *responses: Any) -> None:
        key = (method.upper(), url_suffix)
        self._queues.setdefault(key, []).extend(responses)

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
        self.calls.append(
            (method.upper(), url, {
                "json_body": json_body,
                "headers": headers,
                "auth": auth,
            })
        )
        path = urlparse(url).path
        for (m, suffix), q in self._queues.items():
            if m == method.upper() and path.endswith(suffix) and q:
                item = q.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(
            f"no scripted response for {method} {url}; "
            f"queues: {list(self._queues)}"
        )

    def close(self) -> None:
        pass