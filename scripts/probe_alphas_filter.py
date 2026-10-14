#!/usr/bin/env python3
"""Probe: which filter parameters does /users/self/alphas accept?

Each test issues a single request with a different filter and prints
the HTTP status plus the returned count. Goal: find filters that
partition the ~2700 alphas into chunks under 1000 each.

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/probe_alphas_filter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.client import BrainCredentials
from wqgp.brain.transport import RequestsTransport, TransportError

BASE_URL = "https://api.worldquantbrain.com"
ALPHAS_URL = f"{BASE_URL}/users/self/alphas"


def _login(transport, creds):
    resp = transport.request(
        "POST",
        f"{BASE_URL}/authentication",
        auth=(creds.username, creds.password),
        timeout=30.0,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"login failed: HTTP {resp.status_code}")
    print(f"login OK (user={creds.username})")
    print()


def _try(transport, label: str, query: str) -> None:
    url = f"{ALPHAS_URL}?{query}"
    try:
        resp = transport.request("GET", url, timeout=60.0)
    except TransportError as e:
        print(f"  [{label}]  transport error: {e}")
        return

    code = resp.status_code
    if code == 200:
        try:
            data = resp.json()
            count = data.get("count") if isinstance(data, dict) else "?"
            results = (
                data.get("results", [])
                if isinstance(data, dict) else data
            )
            print(
                f"  [{label}]  200  count={count}  "
                f"returned={len(results)}"
            )
        except Exception as e:
            print(f"  [{label}]  200  but bad JSON: {e}")
    else:
        body = resp.body.decode("utf-8", errors="replace")[:200]
        print(f"  [{label}]  {code}  {body}")


def main() -> int:
    try:
        creds = BrainCredentials.from_env()
    except Exception as e:
        print(f"missing credentials: {e}", file=sys.stderr)
        return 1

    transport = RequestsTransport()
    try:
        _login(transport, creds)

        print("=== probing filter parameters (limit=50) ===")
        print()

        _try(transport, "no filter", "limit=50&offset=0")
        _try(
            transport, "status!=UNSUBMITTED",
            "limit=50&offset=0&status!=UNSUBMITTED",
        )
        _try(
            transport, "status!=SUBMITTED",
            "limit=50&offset=0&status!=SUBMITTED",
        )

        print()
        print("-- date filters --")
        _try(
            transport, "dateCreated>2024-01-01",
            "limit=50&offset=0&dateCreated>2024-01-01",
        )
        _try(
            transport, "dateCreated>=2024-01-01",
            "limit=50&offset=0&dateCreated>=2024-01-01",
        )
        _try(
            transport, "dateCreated__gte=2024-01-01",
            "limit=50&offset=0&dateCreated__gte=2024-01-01",
        )
        _try(
            transport, "created>2024-01-01",
            "limit=50&offset=0&created>2024-01-01",
        )

        print()
        print("-- metric filters --")
        _try(
            transport, "is.sharpe>=1",
            "limit=50&offset=0&is.sharpe>=1",
        )
        _try(
            transport, "is.sharpe>1",
            "limit=50&offset=0&is.sharpe>1",
        )
        _try(
            transport, "sharpe>=1",
            "limit=50&offset=0&sharpe>=1",
        )
        _try(
            transport, "fitness>=1",
            "limit=50&offset=0&fitness>=1",
        )

        print()
        print("-- settings filters --")
        _try(
            transport, "region=USA",
            "limit=50&offset=0&region=USA",
        )
        _try(
            transport, "settings.region=USA",
            "limit=50&offset=0&settings.region=USA",
        )
        _try(
            transport, "universe=TOP3000",
            "limit=50&offset=0&universe=TOP3000",
        )
        _try(
            transport, "delay=1",
            "limit=50&offset=0&delay=1",
        )

        print()
        print("-- ordering --")
        _try(
            transport, "ordering=-dateCreated",
            "limit=50&offset=0&ordering=-dateCreated",
        )
        _try(
            transport, "order=-dateCreated",
            "limit=50&offset=0&order=-dateCreated",
        )

        print()
        print("-- combinations --")
        _try(
            transport,
            "status!=UNSUBMITTED & dateCreated>2024-01-01",
            "limit=50&offset=0"
            "&status!=UNSUBMITTED&dateCreated>2024-01-01",
        )
        _try(
            transport,
            "status!=UNSUBMITTED & is.sharpe>=0.5",
            "limit=50&offset=0"
            "&status!=UNSUBMITTED&is.sharpe>=0.5",
        )

    finally:
        transport.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())