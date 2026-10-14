"""Cross-run evaluation cache.

Design decisions:
- Independent SQLite file, separate from every run's evaluation DB.
  See D-141.
- Table key is (schema_version, fingerprint, settings_hash) hashed
  into a single cache_key string. See D-140.
- Only reusable results are written: SUCCESS, and PERMANENT failures.
  TRANSIENT failures are never cached. See D-142.
- TTL is a soft expiry: entries older than ttl_days are ignored on
  read but not deleted. Explicit clear() removes them. See D-143.
- hit_count and last_hit_at are updated on every successful read, so
  cache value can be measured.
- in_flight columns are reserved for a future concurrency round and
  are unused in this first version. See D-144.
- WAL mode + busy_timeout so a future multi-process run does not hit
  "database is locked".
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from wqgp.evaluation_config import CACHE_SCHEMA_VERSION


log = logging.getLogger("wqgp.storage")


# --- errors -----------------------------------------------------------------

class CacheError(RuntimeError):
    """Base class for evaluation-cache failures."""


class IncompatibleCacheSchemaError(CacheError):
    """Raised when the cache DB's user_version is not what we expect."""


# --- enums ------------------------------------------------------------------

class CacheStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ErrorType(str, Enum):
    """Classification of a cached failure.

    Only PERMANENT failures are cached. TRANSIENT failures (network,
    429, 5xx) are not cached at all, so this enum does not need a
    TRANSIENT member. See D-142.
    """
    PERMANENT = "PERMANENT"


# --- entry ------------------------------------------------------------------

def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return True


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cached evaluation result."""
    cache_key: str
    fingerprint: str
    settings_hash: str
    settings_json: str
    status: CacheStatus
    fitness: float | None = None
    metrics: tuple[tuple[str, float], ...] = ()
    error: str | None = None
    error_type: ErrorType | None = None
    created_at: float = 0.0
    last_hit_at: float = 0.0
    hit_count: int = 0

    def __post_init__(self) -> None:
        if not self.cache_key:
            raise ValueError("cache_key must be non-empty")
        if not self.fingerprint:
            raise ValueError("fingerprint must be non-empty")
        if not self.settings_hash:
            raise ValueError("settings_hash must be non-empty")
        if not isinstance(self.status, CacheStatus):
            raise ValueError(
                f"status must be CacheStatus, "
                f"got {type(self.status).__name__}"
            )
        if self.fitness is not None and not _is_finite_number(self.fitness):
            raise ValueError(
                f"fitness must be a finite number, got {self.fitness!r}"
            )
        for k, v in self.metrics:
            if not isinstance(k, str) or not k:
                raise ValueError("metric name must be a non-empty string")
            if not _is_finite_number(v):
                raise ValueError(
                    f"metric {k!r} must be a finite number, got {v!r}"
                )
        if self.status is CacheStatus.SUCCESS:
            if self.fitness is None:
                raise ValueError("SUCCESS cache entry requires fitness")
            if self.error is not None or self.error_type is not None:
                raise ValueError(
                    "SUCCESS cache entry must not carry error/error_type"
                )
        if self.status is CacheStatus.FAILED:
            if not self.error or not self.error.strip():
                raise ValueError("FAILED cache entry requires error")
            if self.error_type is not ErrorType.PERMANENT:
                raise ValueError(
                    "only PERMANENT failures are cached; "
                    "FAILED entry must set error_type=PERMANENT"
                )

    @property
    def metrics_dict(self) -> dict[str, float]:
        return dict(self.metrics)


# --- stats ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CacheStats:
    """Read-only summary of the cache's current state."""
    total_entries: int
    success_entries: int
    failed_entries: int
    expired_entries: int
    total_hits: int
    oldest_created_at: float | None
    newest_created_at: float | None

    @property
    def avg_hits_per_entry(self) -> float:
        if self.total_entries == 0:
            return 0.0
        return self.total_hits / self.total_entries


# --- schema -----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_cache (
    cache_key TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    settings_hash TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error_type TEXT,
    fitness REAL,
    metrics TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at REAL NOT NULL,
    last_hit_at REAL NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    in_flight INTEGER NOT NULL DEFAULT 0,
    in_flight_since REAL
);
CREATE INDEX IF NOT EXISTS idx_cache_fingerprint
    ON evaluation_cache (fingerprint);
CREATE INDEX IF NOT EXISTS idx_cache_created
    ON evaluation_cache (created_at);
"""

_UPSERT = """
INSERT INTO evaluation_cache (
    cache_key, schema_version, fingerprint, settings_hash,
    settings_json, status, error_type, fitness, metrics, error,
    created_at, last_hit_at, hit_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
ON CONFLICT (cache_key) DO UPDATE SET
    schema_version = excluded.schema_version,
    fingerprint    = excluded.fingerprint,
    settings_hash  = excluded.settings_hash,
    settings_json  = excluded.settings_json,
    status         = excluded.status,
    error_type     = excluded.error_type,
    fitness        = excluded.fitness,
    metrics        = excluded.metrics,
    error          = excluded.error,
    created_at     = excluded.created_at
"""

_SELECT_COLS = (
    "cache_key, schema_version, fingerprint, settings_hash, "
    "settings_json, status, error_type, fitness, metrics, error, "
    "created_at, last_hit_at, hit_count"
)


class EvaluationCache:
    """Thread-safe SQLite cache for BRAIN evaluation results.

    Lifecycle:
        cache = EvaluationCache(path, ttl_days=90)
        entry = cache.get(cache_key)          # updates hit stats
        cache.put(entry)
        stats = cache.stats()
        removed = cache.clear(before_ts=None)
        cache.close()
    """

    __slots__ = ("_path", "_ttl_days", "_lock", "_conn")

    def __init__(
        self,
        path: Path | str,
        *,
        ttl_days: int = 90,
    ) -> None:
        if ttl_days < 1:
            raise ValueError("ttl_days must be >= 1")
        self._path = str(path)
        self._ttl_days = ttl_days
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()
        log.debug(
            "[storage] cache opened: %s (ttl=%d days)",
            self._path, self._ttl_days,
        )

    # --- setup ---------------------------------------------------------

    def _configure(self) -> None:
        cur = self._conn.cursor()
        if self._path != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
        self._conn.commit()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA user_version")
        current = int(cur.fetchone()[0])
        expected = CACHE_SCHEMA_VERSION
        if current == 0:
            cur.executescript(_SCHEMA)
            cur.execute(f"PRAGMA user_version={expected}")
            self._conn.commit()
        elif current != expected:
            raise IncompatibleCacheSchemaError(
                f"cache schema version {current} != expected {expected}"
            )

    # --- read/write ----------------------------------------------------

    def get(self, cache_key: str) -> CacheEntry | None:
        """Return the entry for `cache_key` if it exists and is fresh.

        A fresh hit also bumps hit_count and last_hit_at. An entry
        older than ttl_days is treated as absent but NOT deleted;
        call clear() to remove it. See D-143.
        """
        if not cache_key:
            raise ValueError("cache_key must be non-empty")

        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM evaluation_cache "
                f"WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            now = time.time()
            created_at = float(row["created_at"])
            if self._is_expired(created_at, now):
                return None

            self._conn.execute(
                "UPDATE evaluation_cache "
                "SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE cache_key = ?",
                (now, cache_key),
            )
            self._conn.commit()

        return self._row_to_entry(row)

    def put(self, entry: CacheEntry) -> None:
        """Insert or replace a cache entry."""
        with self._lock:
            self._conn.execute(_UPSERT, self._entry_to_params(entry))
            self._conn.commit()
        log.debug(
            "[storage] cache put: %s status=%s",
            entry.cache_key[:12], entry.status.value,
        )

    def put_many(self, entries: Iterable[CacheEntry]) -> int:
        rows = list(entries)
        if not rows:
            return 0
        params = [self._entry_to_params(e) for e in rows]
        with self._lock:
            self._conn.executemany(_UPSERT, params)
            self._conn.commit()
        return len(rows)

    # --- maintenance ---------------------------------------------------

    def stats(self) -> CacheStats:
        now = time.time()
        cutoff = now - self._ttl_days * 86400.0
        with self._lock:
            cur = self._conn.execute(
                "SELECT "
                "  COUNT(*) AS n, "
                "  SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS s, "
                "  SUM(CASE WHEN status = 'FAILED'  THEN 1 ELSE 0 END) AS f, "
                "  SUM(CASE WHEN created_at < ? THEN 1 ELSE 0 END) AS e, "
                "  COALESCE(SUM(hit_count), 0) AS h, "
                "  MIN(created_at) AS oldest, "
                "  MAX(created_at) AS newest "
                "FROM evaluation_cache",
                (cutoff,),
            )
            row = cur.fetchone()

        return CacheStats(
            total_entries=int(row["n"] or 0),
            success_entries=int(row["s"] or 0),
            failed_entries=int(row["f"] or 0),
            expired_entries=int(row["e"] or 0),
            total_hits=int(row["h"] or 0),
            oldest_created_at=(
                float(row["oldest"]) if row["oldest"] is not None else None
            ),
            newest_created_at=(
                float(row["newest"]) if row["newest"] is not None else None
            ),
        )

    def clear(self, *, before_ts: float | None = None) -> int:
        """Delete entries. None -> all; float -> older than that time."""
        with self._lock:
            if before_ts is None:
                cur = self._conn.execute("DELETE FROM evaluation_cache")
            else:
                cur = self._conn.execute(
                    "DELETE FROM evaluation_cache WHERE created_at < ?",
                    (before_ts,),
                )
            self._conn.commit()
            removed = int(cur.rowcount)
        log.info(
            "[storage] cache cleared: %d entries (before_ts=%s)",
            removed, before_ts,
        )
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        log.debug("[storage] cache closed: %s", self._path)

    # --- introspection -------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM evaluation_cache"
            )
            return int(cur.fetchone()[0])

    # --- internals -----------------------------------------------------

    def _is_expired(self, created_at: float, now: float) -> bool:
        age = now - created_at
        return age > self._ttl_days * 86400.0

    def _entry_to_params(self, e: CacheEntry) -> tuple[Any, ...]:
        now = time.time()
        created = e.created_at if e.created_at > 0 else now
        return (
            e.cache_key,
            CACHE_SCHEMA_VERSION,
            e.fingerprint,
            e.settings_hash,
            e.settings_json,
            e.status.value,
            e.error_type.value if e.error_type is not None else None,
            e.fitness,
            json.dumps(list(e.metrics)),
            e.error,
            created,
            e.last_hit_at if e.last_hit_at > 0 else created,
        )

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        raw_error_type = row["error_type"]
        error_type: ErrorType | None = None
        if raw_error_type is not None:
            error_type = ErrorType(raw_error_type)

        return CacheEntry(
            cache_key=row["cache_key"],
            fingerprint=row["fingerprint"],
            settings_hash=row["settings_hash"],
            settings_json=row["settings_json"],
            status=CacheStatus(row["status"]),
            fitness=row["fitness"],
            metrics=tuple(
                (k, v) for k, v in json.loads(row["metrics"])
            ),
            error=row["error"],
            error_type=error_type,
            created_at=float(row["created_at"]),
            last_hit_at=float(row["last_hit_at"]),
            hit_count=int(row["hit_count"]),
        )