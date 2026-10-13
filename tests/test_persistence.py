"""Tests for the SQLite evaluation store. Phase 2 Step 3 gate."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from wqgp.persistence import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStatus,
    EvaluationStore,
    IncompatibleSchemaError,
)


@pytest.fixture
def store(tmp_path: Path):
    s = EvaluationStore(tmp_path / "eval.db")
    yield s
    s.close()


def _ok(
    *,
    run_id: str = "run_a",
    generation: int = 0,
    fingerprint: str = "fp1",
    expression: str = "add(close, 1)",
    fitness: float = 1.5,
    metrics: tuple[tuple[str, float], ...] = (("sharpe", 2.0),),
) -> EvaluationRecord:
    return EvaluationRecord(
        run_id=run_id,
        generation=generation,
        fingerprint=fingerprint,
        expression=expression,
        status=EvaluationStatus.SUCCESS,
        fitness=fitness,
        metrics=metrics,
    )


def _fail(
    *,
    run_id: str = "run_a",
    fingerprint: str = "fp1",
    error: str = "syntax error",
) -> EvaluationRecord:
    return EvaluationRecord(
        run_id=run_id,
        generation=0,
        fingerprint=fingerprint,
        expression="bad expr",
        status=EvaluationStatus.FAILED,
        error=error,
    )


# --- basic round-trip ---

def test_record_and_get(store: EvaluationStore) -> None:
    store.record(_ok())
    got = store.get("run_a", "fp1")
    assert got is not None
    assert got.fitness == 1.5
    assert got.metrics_dict == {"sharpe": 2.0}
    assert got.status is EvaluationStatus.SUCCESS


def test_get_unknown_returns_none(store: EvaluationStore) -> None:
    assert store.get("run_a", "ghost") is None


def test_record_assigns_timestamps(store: EvaluationStore) -> None:
    store.record(_ok())
    got = store.get("run_a", "fp1")
    assert got is not None
    assert got.created_at > 0
    assert got.updated_at > 0


def test_metrics_roundtrip_multiple(store: EvaluationStore) -> None:
    m = (("sharpe", 1.7), ("turnover", 0.42), ("returns", 0.03))
    store.record(_ok(metrics=m))
    got = store.get("run_a", "fp1")
    assert got is not None
    assert set(got.metrics) == set(m)


# --- upsert semantics (D-029) ---

def test_upsert_updates_fields(store: EvaluationStore) -> None:
    store.record(_ok(fitness=1.0))
    first = store.get("run_a", "fp1")
    assert first is not None
    created = first.created_at

    store.record(_ok(fitness=2.0, generation=3))
    got = store.get("run_a", "fp1")
    assert got is not None
    assert got.fitness == 2.0
    assert got.generation == 3
    assert got.created_at == created          # preserved
    assert got.updated_at >= created          # refreshed


def test_upsert_only_affects_matching_run(store: EvaluationStore) -> None:
    store.record(_ok(run_id="alpha", fitness=1.0))
    store.record(_ok(run_id="beta", fingerprint="fp1", fitness=9.0))
    assert store.get("alpha", "fp1").fitness == 1.0
    assert store.get("beta", "fp1").fitness == 9.0


def test_upsert_can_transition_pending_to_success(
    store: EvaluationStore,
) -> None:
    pending = EvaluationRecord(
        run_id="run_a",
        generation=0,
        fingerprint="fp1",
        expression="add(close, 1)",
        status=EvaluationStatus.PENDING,
    )
    store.record(pending)
    store.record(_ok(fitness=3.0))
    got = store.get("run_a", "fp1")
    assert got is not None
    assert got.status is EvaluationStatus.SUCCESS
    assert got.fitness == 3.0


# --- batch ---

def test_record_many(store: EvaluationStore) -> None:
    recs = [
        _ok(fingerprint=f"fp{i}", fitness=float(i)) for i in range(5)
    ]
    assert store.record_many(recs) == 5
    assert len(store) == 5


def test_record_many_empty_returns_zero(store: EvaluationStore) -> None:
    assert store.record_many([]) == 0


# --- queries ---

def test_list_by_generation(store: EvaluationStore) -> None:
    store.record(_ok(fingerprint="fp1", generation=0))
    store.record(_ok(fingerprint="fp2", generation=0))
    store.record(_ok(fingerprint="fp3", generation=1))
    gen0 = store.list_by_generation("run_a", 0)
    assert {r.fingerprint for r in gen0} == {"fp1", "fp2"}


def test_list_by_generation_sorted_by_fingerprint(
    store: EvaluationStore,
) -> None:
    for fp in ("fpB", "fpA", "fpC"):
        store.record(_ok(fingerprint=fp))
    got = store.list_by_generation("run_a", 0)
    assert [r.fingerprint for r in got] == ["fpA", "fpB", "fpC"]


def test_top_by_fitness(store: EvaluationStore) -> None:
    for i, fp in enumerate(("fp1", "fp2", "fp3")):
        store.record(_ok(fingerprint=fp, fitness=float(i)))
    top = store.top_by_fitness("run_a", 2)
    assert [r.fingerprint for r in top] == ["fp3", "fp2"]


def test_top_by_fitness_zero_returns_empty(store: EvaluationStore) -> None:
    store.record(_ok())
    assert store.top_by_fitness("run_a", 0) == []


def test_top_by_fitness_min_generation(store: EvaluationStore) -> None:
    store.record(_ok(fingerprint="fp1", generation=0, fitness=100.0))
    store.record(_ok(fingerprint="fp2", generation=5, fitness=1.0))
    top = store.top_by_fitness("run_a", 10, min_generation=3)
    assert [r.fingerprint for r in top] == ["fp2"]


def test_top_by_fitness_excludes_failed(store: EvaluationStore) -> None:
    store.record(_ok(fitness=1.0))
    store.record(_fail(fingerprint="bad"))
    top = store.top_by_fitness("run_a", 10)
    assert [r.fingerprint for r in top] == ["fp1"]


def test_count_by_status(store: EvaluationStore) -> None:
    store.record(_ok(fingerprint="fp1"))
    store.record(_ok(fingerprint="fp2"))
    store.record(_fail(fingerprint="fp3"))
    counts = store.count_by_status("run_a")
    assert counts.get("SUCCESS") == 2
    assert counts.get("FAILED") == 1


def test_list_run_ids(store: EvaluationStore) -> None:
    store.record(_ok(run_id="beta"))
    store.record(_ok(run_id="alpha"))
    assert store.list_run_ids() == ["alpha", "beta"]


def test_delete_run(store: EvaluationStore) -> None:
    store.record(_ok(run_id="doomed"))
    store.record(_ok(run_id="keeper"))
    assert store.delete_run("doomed") == 1
    assert store.list_run_ids() == ["keeper"]


# --- persistence across reopen ---

def test_reopen_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "eval.db"
    with EvaluationStore(path) as s:
        s.record(_ok(fitness=4.2))
    with EvaluationStore(path) as s:
        got = s.get("run_a", "fp1")
        assert got is not None
        assert got.fitness == 4.2


def test_schema_version_persisted(tmp_path: Path) -> None:
    path = tmp_path / "eval.db"
    EvaluationStore(path).close()
    conn = sqlite3.connect(str(path))
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert v == SCHEMA_VERSION


def test_schema_version_mismatch_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "eval.db"
    EvaluationStore(path).close()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    conn.close()
    with pytest.raises(IncompatibleSchemaError):
        EvaluationStore(path)


# --- record invariants ---

def test_success_requires_fitness() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="add(close,1)",
            status=EvaluationStatus.SUCCESS,
        )


def test_failed_requires_error() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="add(close,1)",
            status=EvaluationStatus.FAILED,
        )


def test_pending_can_have_neither() -> None:
    r = EvaluationRecord(
        run_id="run_a",
        generation=0,
        fingerprint="fp",
        expression="add(close,1)",
        status=EvaluationStatus.PENDING,
    )
    assert r.fitness is None
    assert r.error is None


def test_negative_generation_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=-1,
            fingerprint="fp",
            expression="x",
            status=EvaluationStatus.PENDING,
        )


def test_empty_expression_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="   ",
            status=EvaluationStatus.PENDING,
        )


def test_bad_run_id_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="../etc",
            generation=0,
            fingerprint="fp",
            expression="x",
            status=EvaluationStatus.PENDING,
        )


def test_string_status_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="x",
            status="SUCCESS",  # type: ignore[arg-type]
        )


def test_bool_metric_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="x",
            status=EvaluationStatus.SUCCESS,
            fitness=1.0,
            metrics=(("ok", True),),  # type: ignore[arg-type]
        )


def test_empty_metric_name_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationRecord(
            run_id="run_a",
            generation=0,
            fingerprint="fp",
            expression="x",
            status=EvaluationStatus.SUCCESS,
            fitness=1.0,
            metrics=(("", 1.0),),
        )