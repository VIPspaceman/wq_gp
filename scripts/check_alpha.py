#!/usr/bin/env python3
"""Fetch an alpha's full checks from BRAIN.

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/check_alpha.py <alpha_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.client import BrainCredentials
from wqgp.brain.transport import RequestsTransport

BASE_URL = "https://api.worldquantbrain.com"


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/check_alpha.py <alpha_id>", file=sys.stderr)
        return 1
    alpha_id = sys.argv[1]

    creds = BrainCredentials.from_env()
    t = RequestsTransport()
    try:
        r = t.request(
            "POST", f"{BASE_URL}/authentication",
            auth=(creds.username, creds.password), timeout=30.0,
        )
        if r.status_code not in (200, 201):
            print(f"login failed: HTTP {r.status_code}", file=sys.stderr)
            return 1

        r = t.request("GET", f"{BASE_URL}/alphas/{alpha_id}", timeout=30.0)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.body[:300]!r}", file=sys.stderr)
            return 1

        data = r.json()
        is_block = data.get("is") or {}
        checks = is_block.get("checks") or []
        regular = data.get("regular")
        code = regular.get("code") if isinstance(regular, dict) else regular

        print(f"alpha_id: {alpha_id}")
        print(f"status:   {data.get('status')}")
        print(f"expr:     {code}")
        print()
        print("=== metrics ===")
        for k in ("sharpe", "fitness", "turnover", "drawdown", "returns", "margin"):
            if k in is_block:
                print(f"  {k:15} {is_block[k]}")
        print()
        print("=== checks ===")
        for c in checks:
            result = c.get("result", "?")
            name = c.get("name", "?")
            limit = c.get("limit")
            value = c.get("value")
            mark = {"PASS": "✓", "FAIL": "✗", "PENDING": "…"}.get(result, "?")
            line = f"  {mark} {result:7}  {name}"
            if limit is not None and value is not None:
                line += f"  limit={limit}  value={value}"
            elif value is not None:
                line += f"  value={value}"
            print(line)
            if c.get("message"):
                print(f"           {c['message']}")
        print()
        print(f"URL: https://platform.worldquantbrain.com/alpha/{alpha_id}")
    finally:
        t.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())