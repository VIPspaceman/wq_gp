#!/usr/bin/env python3
"""One-off: import historical run DBs into the cross-run evaluation cache.

Usage:
    python scripts/import_runs_to_cache.py \\
        --checkpoint-root ./checkpoints \\
        --cache-path ./checkpoints/cache.db \\
        [--region USA] [--universe TOP3000] [--delay 1] [--decay 4] \\
        [--neutralization INDUSTRY] \\
        [--dry-run]

Why this exists
---------------
Each run has its own SQLite DB (overnight_2.db, channel_test.db, ...).
Those DBs are complete experiment records but they do not participate in
the cross-run cache. This script copies the reusable results (SUCCESS
and PERMANENT failures) into cache.db so the next run can skip them.

The catch: run DBs do not store the settings used for each evaluation.
This script assumes a single settings profile for every imported row.
By default that profile is `BrainSettings()` (USA / TOP3000 / delay 1 /
decay 4 / INDUSTRY), which matches every run this project has produced
since D-113. Override the flags if that assumption is wrong for your
history.

What is imported
----------------
- SUCCESS rows: always imported. fitness and metrics are copied.
- FAILED rows: imported only when the error string matches a known
  PERMANENT pattern (see _PERMANENT_PATTERNS). Anything else — network
  errors, rate limits, timeouts, unknown messages — is skipped, because
  caching a transient failure permanently blocks a good expression.

What is NOT imported
--------------------
- PENDING / SKIPPED rows: they represent "never completed", not a result.
- FAILED rows whose error is not recognized as PERMANENT.
- Rows for runs whose fingerprint already exists in the cache under the
  same key: the first writer wins, keeping the oldest created_at so TTL
  ages from the true evaluation time.

Idempotent
----------
Running twice is safe. The second run skips every row whose cache_key
already exists.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Allow `python scripts/import_runs_to_cache.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wqgp.brain.settings import BrainSettings  # noqa: E402
from wqgp.cache import (  # noqa: E402
    CacheEntry,
    CacheStatus,
    ErrorType,
    EvaluationCache,
)
from wqgp.evaluation_config import EvaluationConfig  # noqa: E402


# --- conservative failure classification ------------------------------------
#
# Only these patterns are treated as PERMANENT. Everything else is
# skipped on import. Adding a pattern here is a promise that the error
# will not go away by retrying the same expression under the same
# settings.
_PERMANENT_PATTERNS: tuple[str, ...] = (
    "incompatible unit",                      # BRAIN's unit check
    "simulation error",                       # {"status": "ERROR"}
    "complete response without 'alpha' id",   # malformed COMPLETE
    "alpha response missing 'is' block",      # malformed alpha fetch
)


def classify_failure(error: str | None) -> ErrorType | None:
    """Return PERMANENT if the error is recognized, else None (skip)."""
    if not error:
        return None
    lowered = error.lower()
    for pattern in _PERMANENT_PATTERNS:
        if pattern.lower() in lowered:
            return ErrorType.PERMANENT
    return None


# --- run DB scanning --------------------------------------------------------

def list_run_dbs(checkpoint_root: Path) -> list[Path]:
    """Return every .db under the checkpoint root except cache.db."""
    if not checkpoint_root.is_dir():
        return []
    return sorted(
        p for p in checkpoint_root.glob("*.db")
        if p.name != "cache.db"
    )


def iter_evaluations(db_path: Path):
    """Yield every row from a run DB's evaluations table."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT * FROM evaluations")
        for row in cur:
            yield row
    finally:
        conn.close()


# --- main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import historical run DBs into the evaluation cache.",
    )
    parser.add_argument(
        "--checkpoint-root", default="./checkpoints",
        help="directory containing run DBs (default: ./checkpoints)",
    )
    parser.add_argument(
        "--cache-path", default=None,
        help="cache DB path (default: <checkpoint-root>/cache.db)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be imported without writing anything",
    )
    parser.add_argument("--region", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--decay", type=int, default=None)
    parser.add_argument("--neutralization", default=None)
    parser.add_argument("--ttl-days", type=int, default=90)
    parser.add_argument(
        "--exclude", action="append", default=[],
        metavar="DB_NAME",
        help=(
            "skip this DB file (repeatable). Pass the filename, not "
            "the full path: --exclude first_real.db"
        ),
    )
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    cache_path = (
        Path(args.cache_path) if args.cache_path
        else checkpoint_root / "cache.db"
    )

    # Build the settings profile. Every imported row is assumed to
    # have been evaluated under this profile.
    base = BrainSettings()
    settings = BrainSettings(
        region=args.region or base.region,
        universe=args.universe or base.universe,
        delay=args.delay if args.delay is not None else base.delay,
        decay=args.decay if args.decay is not None else base.decay,
        neutralization=args.neutralization or base.neutralization,
    )
    cfg = EvaluationConfig(settings)
    settings_hash = cfg.settings_hash()
    settings_json = json.dumps(cfg.to_api_dict(), sort_keys=True)

    print(f"checkpoint root: {checkpoint_root}")
    print(f"cache path:      {cache_path}")
    print(f"settings:        {settings.to_api_dict()}")
    print(f"settings hash:   {settings_hash[:16]}...")
    print()

    db_paths = list_run_dbs(checkpoint_root)
    if args.exclude:
        excluded = set(args.exclude)
        before = len(db_paths)
        db_paths = [p for p in db_paths if p.name not in excluded]
        skipped = before - len(db_paths)
        print(f"excluding {skipped} DB(s): {sorted(excluded)}")
        print()
    if not db_paths:
        print("no run DBs to scan")
        return 0
    print(f"scanning {len(db_paths)} run DB(s)")
    print()

    # --- Open the cache (unless dry-run) ---
    cache: EvaluationCache | None = None
    existing_keys: set[str] = set()
    if not args.dry_run:
        cache = EvaluationCache(cache_path, ttl_days=args.ttl_days)
        # Load every existing key so we can skip already-imported rows
        # without bumping hit_count. Reading the private _conn is fine
        # for a one-off script.
        cur = cache._conn.execute(  # noqa: SLF001
            "SELECT cache_key FROM evaluation_cache"
        )
        existing_keys = {row[0] for row in cur.fetchall()}

    # --- Per-DB counters ---
    total_success = 0
    total_permanent = 0
    total_skipped_pending = 0
    total_skipped_transient = 0
    total_skipped_existing = 0

    try:
        for db_path in db_paths:
            n_success = 0
            n_permanent = 0
            n_skip_pending = 0
            n_skip_transient = 0
            n_skip_existing = 0

            entries: list[CacheEntry] = []

            for row in iter_evaluations(db_path):
                fingerprint = row["fingerprint"]
                status = row["status"]
                created_at = float(row["created_at"] or 0.0)

                key = cfg.cache_key(fingerprint)

                if key in existing_keys:
                    n_skip_existing += 1
                    continue

                if status == "SUCCESS":
                    fitness = row["fitness"]
                    if fitness is None:
                        n_skip_transient += 1
                        continue
                    metrics_raw = json.loads(row["metrics"] or "[]")
                    metrics = tuple(
                        (str(k), float(v)) for k, v in metrics_raw
                    )
                    entries.append(CacheEntry(
                        cache_key=key,
                        fingerprint=fingerprint,
                        settings_hash=settings_hash,
                        settings_json=settings_json,
                        status=CacheStatus.SUCCESS,
                        fitness=float(fitness),
                        metrics=metrics,
                        created_at=created_at,
                    ))
                    n_success += 1
                    existing_keys.add(key)

                elif status == "FAILED":
                    error_type = classify_failure(row["error"])
                    if error_type is None:
                        n_skip_transient += 1
                        continue
                    entries.append(CacheEntry(
                        cache_key=key,
                        fingerprint=fingerprint,
                        settings_hash=settings_hash,
                        settings_json=settings_json,
                        status=CacheStatus.FAILED,
                        error=str(row["error"]),
                        error_type=error_type,
                        created_at=created_at,
                    ))
                    n_permanent += 1
                    existing_keys.add(key)

                else:
                    # PENDING / SKIPPED / unknown
                    n_skip_pending += 1

            if cache is not None and entries:
                cache.put_many(entries)

            print(
                f"  {db_path.name:32}  "
                f"success={n_success:4}  permanent={n_permanent:3}  "
                f"skip_transient={n_skip_transient:3}  "
                f"skip_existing={n_skip_existing:4}  "
                f"skip_pending={n_skip_pending:3}"
            )
            total_success += n_success
            total_permanent += n_permanent
            total_skipped_pending += n_skip_pending
            total_skipped_transient += n_skip_transient
            total_skipped_existing += n_skip_existing

    finally:
        if cache is not None:
            cache.close()

    print()
    print("=== summary ===")
    print(f"  imported SUCCESS:           {total_success}")
    print(f"  imported PERMANENT failures: {total_permanent}")
    print(f"  skipped (transient/unknown): {total_skipped_transient}")
    print(f"  skipped (pending/skipped):   {total_skipped_pending}")
    print(f"  skipped (already cached):    {total_skipped_existing}")
    print()

    if args.dry_run:
        print("dry-run: nothing was written")
    else:
        print(f"wrote to {cache_path}")
        # Report cache totals.
        c = EvaluationCache(cache_path, ttl_days=args.ttl_days)
        try:
            s = c.stats()
        finally:
            c.close()
        print(
            f"cache now holds {s.total_entries} entries "
            f"({s.success_entries} success, {s.failed_entries} failed)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())