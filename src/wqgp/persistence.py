"""SQLite-backed evaluation record store.

Design decisions:
- run_id rules shared with checkpoint via _ids.py. See D-026.
- EvaluationRecord is a frozen value object. `fitness` is a first-class
  field because it is the sort key; other metrics go into `metrics` as a
  hashable tuple of pairs. See D-027.
- SQLite with WAL + PRAGMA user_version. See D-028.
- Upsert semantics: created_at preserved, updated_at refreshed. See D-029.
- One connection guarded by a lock; not a pool. See D-030.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from wqgp._ids import validate_run_id


SCHEMA_VERSION = 1


class PersistenceError(RuntimeError):
    """Base class for evaluation-store failures."""


class IncompatibleSchemaError(PersistenceError):
    """Raised when the DB's user_version does not match SCHEMA_VERSION."""


class EvaluationStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One evaluation result.

    Invariants (enforced in __post_init__):
      - run_id passes _ids.validate_run_id
      - generation >= 0
      - fingerprint, expression non-empty
      - status is EvaluationStatus
      - metrics keys non-empty str, values int/float (bool excluded)
      - SUCCESS requires fitness
      - FAILED requires non-empty error
    """
    run_id: str
    generation: int
    fingerprint: str
    expression: str
    status: EvaluationStatus
    fitness: float | None = None
    metrics: tuple[tuple[str, float], ...] = ()
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        if not self.fingerprint:
            raise ValueError("fingerprint must be non-empty")
        if not self.expression or not self.expression.strip():
            raise ValueError("expression must be non-empty")
        if not isinstance(self.status, EvaluationStatus):
            raise ValueError(
                f"status must be EvaluationStatus, "
                f"got {type(self.status).__name__}"
            )
        for k, v in self.metrics:
            if not isinstance(k, str) or not k:
                raise ValueError("metric name must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"metric {k!r} must be int/float, got {type(v).__name__}"
                )
        if self.status is EvaluationStatus.SUCCESS and self.fitness is None:
            raise ValueError("SUCCESS record requires fitness")
        if self.status is EvaluationStatus.FAILED:
            if not self.error or not self.error.strip():
                raise ValueError("FAILED record requires non-empty error")

    @property
    def metrics_dict(self) -> dict[str, float]:
        return dict(self.metrics)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    expression TEXT NOT NULL,
    status TEXT NOT NULL,
    fitness REAL,
    metrics TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (run_id, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_eval_run_gen
    ON evaluations (run_id, generation);
CREATE INDEX IF NOT EXISTS idx_eval_run_fitness
    ON evaluations (run_id, fitness DESC);
CREATE INDEX IF NOT EXISTS idx_eval_run_status
    ON evaluations (run_id, status);
"""


_UPSERT = """
INSERT INTO evaluations (
    run_id, generation, fingerprint, expression, status,
    fitness, metrics, error, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (run_id, fingerprint) DO UPDATE SET
    generation = excluded.generation,
    expression = excluded.expression,
    status = excluded.status,
    fitness = excluded.fitness,
    metrics = excluded.metrics,
    error = excluded.error,
    updated_at = excluded.updated_at
"""


_SELECT_COLS = (
    "run_id, generation, fingerprint, expression, status, "
    "fitness, metrics, error, created_at, updated_at"
)


class EvaluationStore:
    """Thread-safe SQLite store for evaluation records.

    All operations serialize on an internal lock. SQLite itself allows
    one writer + many readers in WAL mode, but since this class holds a
    single connection, we do not benefit from that until a connection
    pool is introduced. See D-030.
    """

    __slots__ = ("_path", "_lock", "_conn")

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()

    # --- setup ---

    def _configure(self) -> None:
        cur = self._conn.cursor()
        if self._path != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
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
            raise IncompatibleSchemaError(
                f"database schema version {current} != "
                f"expected {SCHEMA_VERSION}"
            )

    # --- row helpers ---

    def _to_row_params(
        self, r: EvaluationRecord, now: float
    ) -> tuple[Any, ...]:
        created = r.created_at if r.created_at > 0 else now
        updated = r.updated_at if r.updated_at > 0 else now
        return (
            r.run_id,
            r.generation,
            r.fingerprint,
            r.expression,
            r.status.value,
            r.fitness,
            json.dumps(list(r.metrics)),
            r.error,
            created,
            updated,
        )

    def _row_to_record(self, row: sqlite3.Row) -> EvaluationRecord:
        return EvaluationRecord(
            run_id=row["run_id"],
            generation=int(row["generation"]),
            fingerprint=row["fingerprint"],
            expression=row["expression"],
            status=EvaluationStatus(row["status"]),
            fitness=row["fitness"],
            metrics=tuple(
                (k, v) for k, v in json.loads(row["metrics"])
            ),
            error=row["error"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    # --- write ---

    def record(self, rec: EvaluationRecord) -> None:
        """Upsert a single record."""
        with self._lock:
            self._conn.execute(
                _UPSERT, self._to_row_params(rec, time.time())
            )
            self._conn.commit()

    def record_many(self, recs: Iterable[EvaluationRecord]) -> int:
        """Upsert many records in one transaction. Returns count written."""
        rows = list(recs)
        if not rows:
            return 0
        now = time.time()
        params = [self._to_row_params(r, now) for r in rows]
        with self._lock:
            self._conn.executemany(_UPSERT, params)
            self._conn.commit()
        return len(rows)

    # --- read ---

    def get(
        self, run_id: str, fingerprint: str
    ) -> EvaluationRecord | None:
        validate_run_id(run_id)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM evaluations "
                f"WHERE run_id = ? AND fingerprint = ?",
                (run_id, fingerprint),
            )
            row = cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_by_generation(
        self, run_id: str, generation: int
    ) -> list[EvaluationRecord]:
        validate_run_id(run_id)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM evaluations "
                f"WHERE run_id = ? AND generation = ? "
                f"ORDER BY fingerprint",
                (run_id, generation),
            )
            return [self._row_to_record(r) for r in cur.fetchall()]

    def top_by_fitness(
        self,
        run_id: str,
        k: int,
        *,
        status: EvaluationStatus = EvaluationStatus.SUCCESS,
        min_generation: int | None = None,
    ) -> list[EvaluationRecord]:
        validate_run_id(run_id)
        if k <= 0:
            return []
        params: list[Any] = [run_id, status.value]
        where = "run_id = ? AND status = ? AND fitness IS NOT NULL"
        if min_generation is not None:
            where += " AND generation >= ?"
            params.append(min_generation)
        params.append(k)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_SELECT_COLS} FROM evaluations "
                f"WHERE {where} "
                f"ORDER BY fitness DESC, fingerprint "
                f"LIMIT ?",
                params,
            )
            return [self._row_to_record(r) for r in cur.fetchall()]

    def count_by_status(self, run_id: str) -> dict[str, int]:
        validate_run_id(run_id)
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM evaluations "
                "WHERE run_id = ? GROUP BY status",
                (run_id,),
            )
            return {row["status"]: int(row["n"]) for row in cur.fetchall()}

    def list_run_ids(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT run_id FROM evaluations ORDER BY run_id"
            )
            return [row["run_id"] for row in cur.fetchall()]

    # --- maintenance ---

    def delete_run(self, run_id: str) -> int:
        validate_run_id(run_id)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM evaluations WHERE run_id = ?", (run_id,)
            )
            self._conn.commit()
            return int(cur.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- dunders ---

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM evaluations")
            return int(cur.fetchone()[0])

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()