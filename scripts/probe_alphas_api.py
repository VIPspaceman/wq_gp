#!/usr/bin/env python3
"""Probe: list all alphas from BRAIN's /users/self/alphas endpoint.

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/probe_alphas_api.py [--status all|submitted|unsubmitted]

This is a diagnostic tool. It does NOT write to any database. Its only
job is to confirm the endpoint works and show what the response looks
like, so we can decide how to integrate it into the cache later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.client import BrainCredentials  # noqa: E402
from wqgp.brain.transport import (  # noqa: E402
    RequestsTransport,
    TransportError,
)

BASE_URL = "https://api.worldquantbrain.com"
ALPHAS_URL = f"{BASE_URL}/users/self/alphas"


def _login(transport: RequestsTransport, creds: BrainCredentials) -> None:
    """Authenticate. Raises on failure."""
    resp = transport.request(
        "POST",
        f"{BASE_URL}/authentication",
        auth=(creds.username, creds.password),
        timeout=30.0,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"login failed: HTTP {resp.status_code}"
        )
    print(f"login OK (user={creds.username})")


def _fetch_page(
    transport: RequestsTransport,
    offset: int,
    limit: int,
    status_filter: str,
) -> dict:
    """Fetch one page. Returns the parsed JSON body."""
    params = [f"limit={limit}", f"offset={offset}"]
    if status_filter == "submitted":
        params.append("status!=UNSUBMITTED")
    elif status_filter == "unsubmitted":
        params.append("status!=SUBMITTED")

    url = f"{ALPHAS_URL}?{'&'.join(params)}"
    resp = transport.request("GET", url, timeout=60.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code}: {resp.body[:300]!r}"
        )
    return resp.json()


def _extract_code(alpha: dict) -> str | None:
    """BRAIN stores the expression under regular.code or regular (string)."""
    regular = alpha.get("regular")
    if isinstance(regular, dict):
        code = regular.get("code")
        return code if isinstance(code, str) else None
    if isinstance(regular, str):
        return regular
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status", choices=("all", "submitted", "unsubmitted"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--max-pages", type=int, default=200,
        help="safety cap; 200 pages * 50 = 10000 alphas",
    )
    parser.add_argument(
        "--out", default="./alphas_probe.json",
        help="where to dump the raw response (default: ./alphas_probe.json)",
    )
    args = parser.parse_args()

    try:
        creds = BrainCredentials.from_env()
    except Exception as e:
        print(f"missing credentials: {e}", file=sys.stderr)
        return 1

    transport = RequestsTransport()
    try:
        _login(transport, creds)

        all_items: list[dict] = []
        offset = 0
        for page in range(args.max_pages):
            data = _fetch_page(
                transport, offset, args.limit, args.status,
            )
            # BRAIN list endpoints return {"count": N, "results": [...]}.
            items = (
                data.get("results", data)
                if isinstance(data, dict) else data
            )
            if not isinstance(items, list):
                print(
                    f"unexpected response shape: {type(data)}",
                    file=sys.stderr,
                )
                print(
                    json.dumps(data, indent=2)[:500],
                    file=sys.stderr,
                )
                return 1

            all_items.extend(items)
            print(
                f"  page {page + 1}: +{len(items)} "
                f"(total {len(all_items)})"
            )

            if len(items) < args.limit:
                break
            offset += len(items)

        print()
        print(f"=== fetched {len(all_items)} alphas ===")

        from collections import Counter
        statuses = Counter(a.get("status", "?") for a in all_items)
        print(f"statuses: {dict(statuses)}")

        print()
        print("=== first 3 samples ===")
        for a in all_items[:3]:
            settings = a.get("settings") or {}
            is_block = a.get("is") or {}
            print(json.dumps({
                "id": a.get("id"),
                "status": a.get("status"),
                "regular": _extract_code(a),
                "settings_region": settings.get("region"),
                "settings_universe": settings.get("universe"),
                "settings_delay": settings.get("delay"),
                "settings_decay": settings.get("decay"),
                "settings_neutralization": settings.get("neutralization"),
                "sharpe": is_block.get("sharpe"),
                "fitness": is_block.get("fitness"),
                "turnover": is_block.get("turnover"),
            }, indent=2))

        out = Path(args.out)
        out.write_text(
            json.dumps(all_items, indent=2), encoding="utf-8",
        )
        print(f"\nfull response written to {out}")

    except TransportError as e:
        print(f"transport error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        transport.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())