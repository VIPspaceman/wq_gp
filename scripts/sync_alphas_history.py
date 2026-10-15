#!/usr/bin/env python3
"""Sync BRAIN alphas into the local alphas.db registry.

Usage:
    export WQ_BRAIN_USERNAME=...
    export WQ_BRAIN_PASSWORD=...
    python scripts/sync_alphas_history.py \\
        [--db ./checkpoints/alphas.db] \\
        [--mode full|single] \\
        [--min-sharpe X] [--max-sharpe Y] \\
        [--status all|unsubmitted|submitted] \\
        [--dry-run]

--mode full (default): segments the sharpe range into buckets that
each stay under BRAIN's 1000-row cap, and syncs every bucket.

--mode single: one query with the given --min/--max-sharpe bounds.

Idempotent: existing alpha_ids are updated, new ones inserted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.alpha_registry import AlphaRegistry
from wqgp.brain.client import BrainCredentials
from wqgp.brain.transport import RequestsTransport

BASE_URL = "https://api.worldquantbrain.com"
PAGE_LIMIT = 100


# Sharpe buckets for --mode full. Each bucket must stay under BRAIN's
# 1000-row cap. Bounds are [lo, hi): lo is inclusive, hi exclusive.
DEFAULT_BUCKETS: tuple[tuple[float | None, float | None], ...] = (
    (2.0, None),
    (1.5, 2.0),
    (1.25, 1.5),
    (1.0, 1.25),
    (0.5, 1.0),
    (0.0, 0.5),
    (-0.5, 0.0),
    (-1.0, -0.5),
    (None, -1.0),
)


def _login(t, creds):
    r = t.request(
        "POST", f"{BASE_URL}/authentication",
        auth=(creds.username, creds.password), timeout=30.0,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"login failed: HTTP {r.status_code}")
    print(f"login OK (user={creds.username})")


def _fetch_one_segment(
    t,
    min_sharpe: float | None,
    max_sharpe: float | None,
    status: str,
) -> tuple[list[dict], int | None, bool]:
    """Fetch every row in [min_sharpe, max_sharpe) for the given status.

    Returns (rows, count_reported, partial).
    """
    params = [f"limit={PAGE_LIMIT}", "offset=0"]
    if min_sharpe is not None:
        params.append(f"is.sharpe>={min_sharpe}")
    if max_sharpe is not None:
        params.append(f"is.sharpe<{max_sharpe}")
    if status == "submitted":
        params.append("status!=UNSUBMITTED")
    elif status == "unsubmitted":
        params.append("status!=SUBMITTED")

    all_raw: list[dict] = []
    offset = 0
    count_reported = None
    partial = False

    while True:
        params[1] = f"offset={offset}"
        url = f"{BASE_URL}/users/self/alphas?{'&'.join(params)}"
        r = t.request("GET", url, timeout=60.0)

        if r.status_code == 400:
            partial = True
            break

        if r.status_code != 200:
            print(f"    HTTP {r.status_code} at offset {offset}")
            break

        data = r.json()
        if count_reported is None and isinstance(data, dict):
            count_reported = data.get("count")

        items = (
            data.get("results", data)
            if isinstance(data, dict) else data
        )
        if not isinstance(items, list) or not items:
            break

        all_raw.extend(items)
        print(
            f"    offset={offset:>4}  +{len(items):>3}  "
            f"total={len(all_raw):>4} / {count_reported or '?'}"
        )
        if len(items) < PAGE_LIMIT:
            break
        offset += len(items)
        time.sleep(0.2)

    return all_raw, count_reported, partial


def _fmt_bounds(lo: float | None, hi: float | None) -> str:
    lo_s = f"{lo}" if lo is not None else "-inf"
    hi_s = f"{hi}" if hi is not None else "+inf"
    return f"[{lo_s}, {hi_s})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./checkpoints/alphas.db")
    ap.add_argument(
        "--mode", choices=("full", "single"), default="full",
    )
    ap.add_argument("--min-sharpe", type=float, default=None)
    ap.add_argument("--max-sharpe", type=float, default=None)
    ap.add_argument(
        "--status", choices=("all", "unsubmitted", "submitted"),
        default="all",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "single":
        buckets: list[tuple[float | None, float | None]] = [
            (args.min_sharpe, args.max_sharpe),
        ]
    else:
        buckets = list(DEFAULT_BUCKETS)

    creds = BrainCredentials.from_env()
    t = RequestsTransport()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    reg = AlphaRegistry(db_path)

    total_upserted = 0
    total_fetched = 0

    try:
        _login(t, creds)

        for i, (lo, hi) in enumerate(buckets, start=1):
            print()
            print(f"=== bucket {i}/{len(buckets)}: "
                  f"{_fmt_bounds(lo, hi)} ===")
            rows, count, partial = _fetch_one_segment(
                t, lo, hi, args.status,
            )
            total_fetched += len(rows)
            suffix = "  (partial)" if partial else ""
            print(f"  fetched {len(rows)} rows{suffix}")
            if not args.dry_run and rows:
                reg.upsert_raw_many(rows)
                total_upserted += len(rows)

        print()
        print("=" * 60)
        print(f"total fetched:  {total_fetched}")
        print(f"total upserted: {total_upserted}")
        print(f"registry total: {reg.count()}")
        print()
        print("status breakdown:")
        for s, n in sorted(reg.count_by_status().items()):
            print(f"  {s:<20} {n}")
        print()
        print("failed checks breakdown:")
        for name, n in sorted(
            reg.count_by_failed_check().items(),
            key=lambda kv: -kv[1],
        ):
            print(f"  {name:<30} {n}")

        # Submittable count.
        subs = reg.list_submittable()
        print()
        print(f"submittable (0 FAIL): {len(subs)}")

    finally:
        reg.close()
        t.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())