#!/usr/bin/env python3
"""Fetch historical alphas from BRAIN into a reference JSON file.

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/fetch_alphas_reference.py \\
        [--min-sharpe 0.5] \\
        [--out data/alphas_reference.json]

This script ONLY reads from BRAIN. It does not touch the evaluation
cache, the run DBs, or any part of the wqgp package. Its output is a
plain JSON file used by analyze_alphas_reference.py.

BRAIN limits /users/self/alphas to the first 1000 rows. A single
--min-sharpe=0.5 query returns ~971 rows on this account (measured),
which fits. If the filtered count exceeds the limit, the fetch stops
and the file is marked partial.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.client import BrainCredentials
from wqgp.brain.transport import RequestsTransport, TransportError

BASE_URL = "https://api.worldquantbrain.com"
ALPHAS_URL = f"{BASE_URL}/users/self/alphas"
PAGE_LIMIT = 100


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


def _extract_code(alpha):
    regular = alpha.get("regular")
    if isinstance(regular, dict):
        code = regular.get("code")
        return code if isinstance(code, str) else None
    if isinstance(regular, str):
        return regular
    return None


def _slim(alpha):
    """Extract the fields we care about from one raw alpha row."""
    settings = alpha.get("settings") or {}
    is_block = alpha.get("is") or {}
    return {
        "alpha_id": alpha.get("id"),
        "status": alpha.get("status"),
        "expression": _extract_code(alpha),
        "settings": {
            "instrument_type": settings.get("instrumentType"),
            "region": settings.get("region"),
            "universe": settings.get("universe"),
            "delay": settings.get("delay"),
            "decay": settings.get("decay"),
            "neutralization": settings.get("neutralization"),
            "truncation": settings.get("truncation"),
        },
        "metrics": {
            "sharpe": is_block.get("sharpe"),
            "fitness": is_block.get("fitness"),
            "turnover": is_block.get("turnover"),
            "drawdown": is_block.get("drawdown"),
            "returns": is_block.get("returns"),
        },
        "date_created": alpha.get("dateCreated"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-sharpe", type=float, default=0.5,
        help="only fetch alphas with sharpe >= this (default 0.5)",
    )
    parser.add_argument(
        "--out", default="data/alphas_reference.json",
        help="output path",
    )
    parser.add_argument(
        "--status", choices=("all", "submitted"),
        default="all",
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

        params = [
            f"limit={PAGE_LIMIT}",
            "offset=0",
            f"is.sharpe>={args.min_sharpe}",
        ]
        if args.status == "submitted":
            params.append("status!=UNSUBMITTED")

        all_items = []
        offset = 0
        partial = False
        count_total = None

        while True:
            params[1] = f"offset={offset}"
            url = f"{ALPHAS_URL}?{'&'.join(params)}"
            try:
                resp = transport.request("GET", url, timeout=60.0)
            except TransportError as e:
                print(
                    f"transport error at offset {offset}: {e}",
                    file=sys.stderr,
                )
                break

            if resp.status_code == 400:
                body = resp.body.decode("utf-8", errors="replace")
                print(f"HTTP 400 at offset {offset}: {body[:200]}")
                print("(likely the 1000-row cap; marking as partial)")
                partial = True
                break

            if resp.status_code != 200:
                print(
                    f"HTTP {resp.status_code} at offset {offset}",
                    file=sys.stderr,
                )
                break

            data = resp.json()
            if count_total is None and isinstance(data, dict):
                count_total = data.get("count")

            items = (
                data.get("results", data)
                if isinstance(data, dict) else data
            )
            if not isinstance(items, list):
                print(
                    f"unexpected payload at offset {offset}",
                    file=sys.stderr,
                )
                break

            if not items:
                break

            all_items.extend(items)
            print(
                f"  page {offset // PAGE_LIMIT + 1}: "
                f"+{len(items)} (total {len(all_items)} / {count_total})"
            )

            if len(items) < PAGE_LIMIT:
                break

            offset += len(items)
            time.sleep(0.2)  # polite pacing

        slim = [_slim(a) for a in all_items]
        slim = [s for s in slim if s["expression"]]

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "min_sharpe": args.min_sharpe,
            "status_filter": args.status,
            "count_reported_by_brain": count_total,
            "count_fetched": len(slim),
            "partial": partial,
            "alphas": slim,
        }
        out_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )

        print()
        print(f"=== fetched {len(slim)} alphas ===")
        print(f"reported by BRAIN: {count_total}")
        print(f"partial:           {partial}")
        print(f"written to:        {out_path}")
        print(
            f"file size:         "
            f"{out_path.stat().st_size / 1024:.1f} KB"
        )

    finally:
        transport.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())