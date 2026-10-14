#!/usr/bin/env python3
"""Probe: are ts_backfill and ts_av_diff usable on this account?

Submits one minimal expression per operator and polls to completion.
Takes about 5 minutes total (two sequential simulations).

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/probe_new_operators.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.client import BrainCredentials
from wqgp.brain.transport import RequestsTransport


SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 4,
    "neutralization": "INDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "nanHandling": "OFF",
    "unitHandling": "VERIFY",
    "language": "FASTEXPR",
    "visualization": False,
    "testPeriod": "P0Y0M",
}


def _test(transport, expr: str) -> None:
    print(f"\n=== {expr} ===")
    r = transport.request(
        "POST",
        "https://api.worldquantbrain.com/simulations",
        json_body={
            "type": "REGULAR",
            "settings": SETTINGS,
            "regular": expr,
        },
    )
    print(f"  submit: HTTP {r.status_code}")
    if r.status_code not in (200, 201):
        print(f"  body: {r.body[:300].decode('utf-8', errors='replace')}")
        return

    loc = r.header("Location")
    if not loc:
        print("  no Location header")
        return

    for i in range(40):
        r = transport.request("GET", loc)
        if r.status_code != 200:
            print(f"  poll {i}: HTTP {r.status_code}")
            time.sleep(3)
            continue
        try:
            data = r.json()
        except Exception:
            print(f"  poll {i}: non-JSON")
            time.sleep(3)
            continue
        if isinstance(data, dict) and "status" in data:
            status = data.get("status")
            msg = data.get("message", "")
            print(f"  FINAL: status={status}")
            if msg:
                print(f"  message: {msg[:250]}")
            alpha = data.get("alpha")
            if alpha:
                print(f"  alpha: {alpha}")
            return
        progress = data.get("progress", "?") if isinstance(data, dict) else "?"
        print(f"  poll {i}: progress={progress}")
        time.sleep(6)

    print("  TIMEOUT after 40 polls")


def main() -> int:
    creds = BrainCredentials.from_env()
    transport = RequestsTransport()
    try:
        r = transport.request(
            "POST",
            "https://api.worldquantbrain.com/authentication",
            auth=(creds.username, creds.password),
            timeout=30.0,
        )
        if r.status_code not in (200, 201):
            print(f"login failed: HTTP {r.status_code}", file=sys.stderr)
            return 1
        print(f"login OK (user={creds.username})")

        _test(transport, "ts_backfill(close, 120)")
        _test(transport, "ts_av_diff(close, 20)")

    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())