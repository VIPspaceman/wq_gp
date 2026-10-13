"""Shared test doubles.

ScriptedTransport was extracted from tests/test_brain_client.py so
tests/test_pool_builder.py can use the same fake without duplicating
the class. Behavior is unchanged. See D-079 for the path-matching rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from wqgp.brain.transport import HttpResponse


@dataclass
class ScriptedTransport:
    """Test double for HttpTransport.

    Register responses in FIFO order per (method, url-path-prefix).
    Every request pops the next queued response. Records every call.

    Prefixes are URL PATHS ("/authentication"), not full URLs.
    See D-079.
    """
    calls: list[tuple[str, str, dict[str, Any]]] = field(
        default_factory=list
    )
    _queues: dict[tuple[str, str], list[Any]] = field(default_factory=dict)

    def expect(self, method: str, url_prefix: str, *responses: Any) -> None:
        key = (method.upper(), url_prefix)
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
        for (m, prefix), q in self._queues.items():
            if m == method.upper() and path.startswith(prefix) and q:
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