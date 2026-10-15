#!/usr/bin/env python3
"""Query the local alphas.db registry.

Usage:
    python scripts/query_alphas.py [options]

Options:
    --db PATH                alphas.db (default ./checkpoints/alphas.db)
    --submittable            only alphas with 0 FAIL checks
    --min-sharpe X           filter by sharpe
    --status STATUS          UNSUBMITTED / ACTIVE
    --fail-check NAME        alphas that FAIL this specific check
    --contains SUBSTR        expression substring match
    --limit N                max rows (default 30)
    --sort {sharpe,date}     default sharpe
    --urls                   print only the BRAIN URL per row
    --summary                print only aggregate stats, no rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.alpha_registry import AlphaRecord, AlphaRegistry


def _fmt_row(r: AlphaRecord, *, urls_only: bool) -> str:
    if urls_only:
        return f"https://platform.worldquantbrain.com/alpha/{r.alpha_id}"
    sharpe = f"{r.sharpe:.4f}" if r.sharpe is not None else "  n/a"
    fit = f"{r.fitness:.4f}" if r.fitness is not None else "  n/a"
    turn = f"{r.turnover:.4f}" if r.turnover is not None else "  n/a"
    fails = ",".join(r.failed_check_names()) or "-"
    return (
        f"  {r.alpha_id}  "
        f"sh={sharpe}  fit={fit}  to={turn}  "
        f"pass={r.n_pass} fail={r.n_fail} pend={r.n_pending}  "
        f"[{fails}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="./checkpoints/alphas.db")
    ap.add_argument("--submittable", action="store_true")
    ap.add_argument("--min-sharpe", type=float, default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--fail-check", default=None)
    ap.add_argument("--contains", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--sort", choices=("sharpe", "date"), default="sharpe")
    ap.add_argument("--urls", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no registry at {db_path}", file=sys.stderr)
        print("run scripts/sync_alphas_history.py first", file=sys.stderr)
        return 1

    with AlphaRegistry(db_path) as reg:
        total = reg.count()
        by_status = reg.count_by_status()
        by_fail = reg.count_by_failed_check()

        if args.summary:
            print(f"registry: {db_path}")
            print(f"total:    {total}")
            print()
            print("status:")
            for s, n in sorted(by_status.items()):
                print(f"  {s:<20} {n}")
            print()
            print("failed checks:")
            for name, n in sorted(by_fail.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<30} {n}")
            return 0

        # Build the query set.
        if args.submittable:
            rows = reg.list_submittable(min_sharpe=args.min_sharpe)
        elif args.status is not None:
            rows = reg.list_by_status(args.status)
            if args.min_sharpe is not None:
                rows = [
                    r for r in rows
                    if r.sharpe is not None and r.sharpe >= args.min_sharpe
                ]
        elif args.fail_check is not None:
            rows = reg.list_failing_check(args.fail_check)
            if args.min_sharpe is not None:
                rows = [
                    r for r in rows
                    if r.sharpe is not None and r.sharpe >= args.min_sharpe
                ]
        elif args.contains is not None:
            rows = reg.find_expression_substring(args.contains)
            if args.min_sharpe is not None:
                rows = [
                    r for r in rows
                    if r.sharpe is not None and r.sharpe >= args.min_sharpe
                ]
        else:
            rows = reg.top_by_sharpe(
                max(args.limit, 500),
                min_sharpe=args.min_sharpe,
            )

        if args.sort == "date" and rows:
            rows = sorted(
                rows,
                key=lambda r: (r.date_created or "", r.alpha_id),
                reverse=True,
            )

        n = len(rows)
        shown = rows[: args.limit]
        print(f"matched: {n}  showing: {len(shown)}")
        print()
        for r in shown:
            print(_fmt_row(r, urls_only=args.urls))

        if args.urls:
            return 0

        # Show full details for the first row only, as a sample.
        if shown:
            r = shown[0]
            print()
            print("=" * 70)
            print("sample (first row, full details)")
            print("=" * 70)
            print(f"alpha_id:    {r.alpha_id}")
            print(f"status:      {r.status}")
            print(f"region/univ: {r.region}/{r.universe} delay={r.delay} "
                  f"decay={r.decay} neut={r.neutralization}")
            print(f"date:        {r.date_created}")
            print(f"expression:  {r.expression[:120]}")
            print()
            print("checks:")
            for c in r.checks:
                mark = {
                    "PASS": "✓", "FAIL": "✗",
                    "PENDING": "…", "WARNING": "!",
                }.get(c.result, "?")
                line = f"  {mark} {c.result:<8} {c.name}"
                if c.limit is not None and c.value is not None:
                    line += f"  limit={c.limit} value={c.value}"
                print(line)
            print()
            print(f"URL: https://platform.worldquantbrain.com/alpha/{r.alpha_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())