"""Persistent index of BRAIN alphas.

Design decisions:
- Independent SQLite file (default: <checkpoint_root>/alphas.db),
  separate from both run DBs and the evaluation cache. See D-152.
- Primary key is BRAIN's alpha_id. There is no attempt to relate an
  API row back to a local AST fingerprint; the two identity systems
  cannot be reconciled without a BRAIN-expression parser we are not
  building. See D-153.
- Checks are stored as structured (name, result, limit, value,
  message) tuples, not as a blob, so they can be queried. See D-154.
- upsert_raw() accepts the raw API response dict directly. Extracting
  fields is this module's job, not the caller's.
- Thread-safe via a single connection + lock, same pattern as
  EvaluationStore (D-030).
- No TTL. Alphas live forever on BRAIN's side and here.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("wqgp.storage")


SCHEMA_VERSION = 1


class AlphaRegistryError(RuntimeError):
    """Base class for alpha-registry failures."""


class IncompatibleRegistrySchemaError(AlphaRegistryError):
    """Raised when the registry DB's user_version is not expected."""


# --- structured check ------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AlphaCheck:
    name: str
    result: str
    limit: float | None = None
    value: float | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("check name must be non-empty")

    @property
    def passed(self) -> bool:
        return self.result == "PASS"

    @property
    def failed(self) -> bool:
        return self.result == "FAIL"

    @property
    def pending(self) -> bool:
        return self.result == "PENDING"


# --- alpha record ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AlphaRecord:
    """One BRAIN alpha, as returned by /users/self/alphas."""
    alpha_id: str
    expression: str
    status: str

    region: str | None = None
    universe: str | None = None
    delay: int | None = None
    decay: int | None = None
    neutralization: str | None = None
    instrument_type: str | None = None

    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    drawdown: float | None = None
    returns: float | None = None
    margin: float | None = None

    checks: tuple[AlphaCheck, ...] = ()

    date_created: str | None = None
    fetched_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.alpha_id:
            raise ValueError("alpha_id must be non-empty")

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.failed)

    @property
    def n_pending(self) -> int:
        return sum(1 for c in self.checks if c.pending)

    def is_submittable(self) -> bool:
        """True iff no check has result=FAIL.

        PENDING checks (SELF_CORRELATION) do not block this predicate;
        they can only be resolved by a submission attempt.
        """
        return all(not c.failed for c in self.checks)

    def failed_check_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.failed)


# --- extraction from raw API dicts -----------------------------------------

def _safe_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    f = float(v)
    if not math.isfinite(f):
        return None
    return f


def _extract_code(alpha: dict) -> str:
    regular = alpha.get("regular")
    if isinstance(regular, dict):
        code = regular.get("code")
        return code if isinstance(code, str) else ""
    if isinstance(regular, str):
        return regular
    return ""


def extract_checks(alpha: dict) -> tuple[AlphaCheck, ...]:
    is_block = alpha.get("is") or {}
    checks = is_block.get("checks") or []
    out: list[AlphaCheck] = []
    for c in checks:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        result = c.get("result")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(result, str):
            result = "UNKNOWN"
        out.append(AlphaCheck(
            name=name,
            result=result,
            limit=_safe_float(c.get("limit")),
            value=_safe_float(c.get("value")),
            message=(
                c.get("message")
                if isinstance(c.get("message"), str) else None
            ),
        ))
    return tuple(out)


def extract_record(raw: dict) -> AlphaRecord:
    """Build an AlphaRecord from one raw API alpha dict."""
    is_block = raw.get("is") or {}
    settings = raw.get("settings") or {}
    now = time.time()
    return AlphaRecord(
        alpha_id=str(raw.get("id") or ""),
        expression=_extract_code(raw),
        status=str(raw.get("status") or "UNKNOWN"),
        region=settings.get("region"),
        universe=settings.get("universe"),
        delay=(
            settings.get("delay")
            if isinstance(settings.get("delay"), int) else None
        ),
        decay=(
            settings.get("decay")
            if isinstance(settings.get("decay"), int) else None
        ),
        neutralization=settings.get("neutralization"),
        instrument_type=settings.get("instrumentType"),
        sharpe=_safe_float(is_block.get("sharpe")),
        fitness=_safe_float(is_block.get("fitness")),
        turnover=_safe_float(is_block.get("turnover")),
        drawdown=_safe_float(is_block.get("drawdown")),
        returns=_safe_float(is_block.get("returns")),
        margin=_safe_float(is_block.get("margin")),
        checks=extract_checks(raw),
        date_created=(
            raw.get("dateCreated")
            if isinstance(raw.get("dateCreated"), str) else None
        ),
        fetched_at=now,
        updated_at=now,
    )


# --- schema -----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
    alpha_id TEXT PRIMARY KEY,
    expression TEXT NOT NULL,
    status TEXT NOT NULL,
    region TEXT,
    universe TEXT,
    delay INTEGER,
    decay INTEGER,
    neutralization TEXT,
    instrument_type TEXT,
    sharpe REAL,
    fitness REAL,
    turnover REAL,
    drawdown REAL,
    returns REAL,
    margin REAL,
    checks_json TEXT NOT NULL DEFAULT '[]',
    settings_json TEXT,
    metrics_json TEXT,
    raw_json TEXT,
    date_created TEXT,
    fetched_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alphas_status
    ON alphas (status);
CREATE INDEX IF NOT EXISTS idx_alphas_sharpe
    ON alphas (sharpe DESC);
CREATE INDEX IF NOT EXISTS idx_alphas_region
    ON alphas (region, universe, delay);
CREATE INDEX IF NOT EXISTS idx_alphas_date_created
    ON alphas (date_created);
"""

_UPSERT = """
INSERT INTO alphas (
    alpha_id, expression, status,
    region, universe, delay, decay, neutralization, instrument_type,
    sharpe, fitness, turnover, drawdown, returns, margin,
    checks_json, settings_json, metrics_json, raw_json,
    date_created, fetched_at, updated_at
) VALUES (
    ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?
)
ON CONFLICT (alpha_id) DO UPDATE SET
    expression      = excluded.expression,
    status          = excluded.status,
    region          = excluded.region,
    universe        = excluded.universe,
    delay           = excluded.delay,
    decay           = excluded.decay,
    neutralization  = excluded.neutralization,
    instrument_type = excluded.instrument_type,
    sharpe          = excluded.sharpe,
    fitness         = excluded.fitness,
    turnover        = excluded.turnover,
    drawdown        = excluded.drawdown,
    returns         = excluded.returns,
    margin          = excluded.margin,
    checks_json     = excluded.checks_json,
    settings_json   = excluded.settings_json,
    metrics_json    = excluded.metrics_json,
    raw_json        = excluded.raw_json,
    date_created    = excluded.date_created,
    updated_at      = excluded.updated_at
"""

_SELECT_COLS = (
    "alpha_id, expression, status, "
    "region, universe, delay, decay, neutralization, instrument_type, "
    "sharpe, fitness, turnover, drawdown, returns, margin, "
    "checks_json, settings_json, metrics_json, raw_json, "
    "date_created, fetched_at, updated_at"
)


class AlphaRegistry:
    """Thread-safe SQLite index of BRAIN alphas."""

    __slots__ = ("_path", "_lock", "_conn")

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()
        log.debug("[storage] alpha registry opened: %s", self._path)

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
        if current == 0:
            cur.executescript(_SCHEMA)
            cur.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()
        elif current != SCHEMA_VERSION:
            raise IncompatibleRegistrySchemaError(
                f"alpha registry schema version {current} != "
                f"{SCHEMA_VERSION}"
            )

    # --- write ---------------------------------------------------------

    def upsert_raw(self, raw: dict) -> None:
        """Insert or update from a raw API response dict."""
        if not isinstance(raw, dict):
            return
        aid = raw.get("id")
        if not isinstance(aid, str) or not aid:
            return
        rec = extract_record(raw)
        self.upsert(rec, raw=raw)

    def upsert(self, rec: AlphaRecord, raw: dict | None = None) -> None:
        params = self._record_to_params(rec, raw)
        with self._lock:
            self._conn.execute(_UPSERT, params)
            self._conn.commit()

    def upsert_many(
        self,
        items: Iterable[tuple[AlphaRecord, dict | None]],
    ) -> int:
        rows = list(items)
        if not rows:
            return 0
        params = [self._record_to_params(r, raw) for r, raw in rows]
        with self._lock:
            self._conn.executemany(_UPSERT, params)
            self._conn.commit()
        return len(rows)

    def upsert_raw_many(self, raws: Iterable[dict]) -> int:
        rows: list[tuple[AlphaRecord, dict]] = []
        for raw in raws:
            if not isinstance(raw, dict):
                continue
            if not isinstance(raw.get("id"), str) or not raw["id"]:
                continue
            rows.append((extract_record(raw), raw))
        return self.upsert_many(rows)

    # --- read ----------------------------------------------------------

    def get(self, alpha_id: str) -> AlphaRecord | None:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM alphas WHERE alpha_id = ?",
                (alpha_id,),
            )
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_submittable(
        self, *, min_sharpe: float | None = None,
    ) -> list[AlphaRecord]:
        sql = (
            f"SELECT {_SELECT_COLS} FROM alphas "
            f"WHERE checks_json NOT LIKE '%\"result\": \"FAIL\"%'"
        )
        params: list[Any] = []
        if min_sharpe is not None:
            sql += " AND sharpe IS NOT NULL AND sharpe >= ?"
            params.append(min_sharpe)
        sql += " ORDER BY sharpe DESC"
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        return [
            r for r in (self._row_to_record(row) for row in rows)
            if r.is_submittable()
        ]

    def list_by_status(self, status: str) -> list[AlphaRecord]:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM alphas "
                f"WHERE status = ? ORDER BY sharpe DESC",
                (status,),
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_failing_check(
        self, check_name: str,
    ) -> list[AlphaRecord]:
        needle = (
            f'%"name": "{check_name}", '
            f'"result": "FAIL"%'
        )
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM alphas "
                f"WHERE checks_json LIKE ? ORDER BY sharpe DESC",
                (needle,),
            )
            rows = cur.fetchall()
        # JSON ordering may vary; re-check in Python.
        return [
            r for r in (self._row_to_record(row) for row in rows)
            if check_name in r.failed_check_names()
        ]

    def top_by_sharpe(
        self, n: int, *, min_sharpe: float | None = None,
    ) -> list[AlphaRecord]:
        if n <= 0:
            return []
        sql = f"SELECT {_SELECT_COLS} FROM alphas WHERE sharpe IS NOT NULL"
        params: list[Any] = []
        if min_sharpe is not None:
            sql += " AND sharpe >= ?"
            params.append(min_sharpe)
        sql += " ORDER BY sharpe DESC LIMIT ?"
        params.append(n)
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_expression_substring(
        self, s: str,
    ) -> list[AlphaRecord]:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM alphas "
                f"WHERE expression LIKE ? ORDER BY sharpe DESC",
                (f"%{s}%",),
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM alphas")
            return int(cur.fetchone()[0])

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM alphas GROUP BY status"
            )
            return {
                row["status"]: int(row["n"]) for row in cur.fetchall()
            }

    def count_by_failed_check(self) -> dict[str, int]:
        """How many alphas fail each check."""
        with self._lock:
            cur = self._conn.execute(
                f"SELECT checks_json FROM alphas"
            )
            rows = cur.fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            for c in json.loads(row["checks_json"] or "[]"):
                if isinstance(c, dict) and c.get("result") == "FAIL":
                    n = c.get("name")
                    if isinstance(n, str):
                        counts[n] = counts.get(n, 0) + 1
        return counts

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        log.debug("[storage] alpha registry closed: %s", self._path)

    def __len__(self) -> int:
        return self.count()

    def __enter__(self) -> "AlphaRegistry":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --- internals -----------------------------------------------------

    def _record_to_params(
        self, r: AlphaRecord, raw: dict | None,
    ) -> tuple[Any, ...]:
        now = time.time()
        created = r.fetched_at if r.fetched_at > 0 else now
        updated = r.updated_at if r.updated_at > 0 else now
        settings_json = None
        metrics_json = None
        if raw is not None:
            settings_json = json.dumps(
                raw.get("settings") or {}, sort_keys=True,
            )
            metrics_json = json.dumps(
                raw.get("is") or {}, sort_keys=True,
            )
        return (
            r.alpha_id, r.expression, r.status,
            r.region, r.universe, r.delay, r.decay,
            r.neutralization, r.instrument_type,
            r.sharpe, r.fitness, r.turnover,
            r.drawdown, r.returns, r.margin,
            json.dumps([
                {
                    "name": c.name,
                    "result": c.result,
                    "limit": c.limit,
                    "value": c.value,
                    "message": c.message,
                }
                for c in r.checks
            ]),
            settings_json,
            metrics_json,
            json.dumps(raw) if raw is not None else None,
            r.date_created,
            created, updated,
        )

    def _row_to_record(self, row: sqlite3.Row) -> AlphaRecord:
        checks_raw = json.loads(row["checks_json"] or "[]")
        checks = tuple(
            AlphaCheck(
                name=str(c.get("name") or ""),
                result=str(c.get("result") or "UNKNOWN"),
                limit=_safe_float(c.get("limit")),
                value=_safe_float(c.get("value")),
                message=(
                    c.get("message")
                    if isinstance(c.get("message"), str) else None
                ),
            )
            for c in checks_raw
            if isinstance(c, dict)
        )
        return AlphaRecord(
            alpha_id=row["alpha_id"],
            expression=row["expression"],
            status=row["status"],
            region=row["region"],
            universe=row["universe"],
            delay=row["delay"],
            decay=row["decay"],
            neutralization=row["neutralization"],
            instrument_type=row["instrument_type"],
            sharpe=row["sharpe"],
            fitness=row["fitness"],
            turnover=row["turnover"],
            drawdown=row["drawdown"],
            returns=row["returns"],
            margin=row["margin"],
            checks=checks,
            date_created=row["date_created"],
            fetched_at=float(row["fetched_at"]),
            updated_at=float(row["updated_at"]),
        )