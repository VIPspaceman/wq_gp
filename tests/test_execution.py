"""Tests for the concurrent executor. Phase 2 Step 4 gate."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from wqgp.budget import Budget, BudgetTracker
from wqgp.execution import (
    Evaluator,
    Executor,
    ExecutorError,
    ExecutorStats,
    Task,
    TaskResult,
    TaskStatus,
)
from wqgp.persistence import EvaluationStatus, EvaluationStore


@pytest.fixture
def store(tmp_path: Path):
    s = EvaluationStore(tmp_path / "eval.db")
    yield s
    s.close()


def _task(i: int = 0, generation: int = 0) -> Task:
    return Task(
        fingerprint=f"fp{i}",
        expression=f"add(close, {i})",
        generation=generation,
    )


def _ok_evaluator(fitness_fn=None) -> Evaluator:
    def ev(task: Task) -> TaskResult:
        fit = float(fitness_fn(task)) if fitness_fn else 1.0
        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.SUCCESS,
            fitness=fit,
            metrics=(("sharpe", fit / 2),),
        )
    return ev


def _tracker(total: int = 1000) -> BudgetTracker:
    return BudgetTracker(Budget(total_simulations=total))


# --- Task value object ---

def test_task_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError):
        Task(fingerprint="", expression="x", generation=0)


def test_task_rejects_empty_expression() -> None:
    with pytest.raises(ValueError):
        Task(fingerprint="fp", expression="   ", generation=0)


def test_task_rejects_negative_generation() -> None:
    with pytest.raises(ValueError):
        Task(fingerprint="fp", expression="x", generation=-1)


# --- TaskResult value object ---

def test_task_result_success_requires_fitness() -> None:
    with pytest.raises(ValueError):
        TaskResult(fingerprint="fp", status=TaskStatus.SUCCESS)


def test_task_result_failed_requires_error() -> None:
    with pytest.raises(ValueError):
        TaskResult(fingerprint="fp", status=TaskStatus.FAILED)


def test_task_result_rejects_bool_metric() -> None:
    with pytest.raises(ValueError):
        TaskResult(
            fingerprint="fp",
            status=TaskStatus.SUCCESS,
            fitness=1.0,
            metrics=(("ok", True),),  # type: ignore[arg-type]
        )


def test_task_result_skipped_is_legal_without_extras() -> None:
    r = TaskResult(fingerprint="fp", status=TaskStatus.SKIPPED)
    assert r.fitness is None
    assert r.error is None


# --- basic drain ---

def test_single_task_drain(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    try:
        assert ex.submit(_task(0))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.submitted == 1
    assert stats.completed == 1
    assert stats.failed == 0
    rec = store.get("run_a", "fp0")
    assert rec is not None
    assert rec.status is EvaluationStatus.SUCCESS
    assert rec.fitness == 1.0


def test_many_tasks_drain(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(fitness_fn=lambda t: int(t.fingerprint[2:])),
        tracker=_tracker(),
        n_workers=4,
    )
    ex.start()
    try:
        for i in range(20):
            assert ex.submit(_task(i))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.submitted == 20
    assert stats.completed == 20
    assert len(store) == 20
    top = store.top_by_fitness("run_a", 3)
    assert [r.fingerprint for r in top] == ["fp19", "fp18", "fp17"]


# --- failure paths ---

def test_evaluator_exception_becomes_failed_record(
    store: EvaluationStore,
) -> None:
    def boom(task: Task) -> TaskResult:
        raise RuntimeError("brain said no")

    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=boom,
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    try:
        assert ex.submit(_task(0))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.failed == 1
    rec = store.get("run_a", "fp0")
    assert rec is not None
    assert rec.status is EvaluationStatus.FAILED
    assert "RuntimeError" in (rec.error or "")
    assert "brain said no" in (rec.error or "")


def test_one_failure_does_not_kill_pipeline(store: EvaluationStore) -> None:
    def fail_on_odd(task: Task) -> TaskResult:
        n = int(task.fingerprint[2:])
        if n % 2 == 1:
            raise ValueError(f"odd {n}")
        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.SUCCESS,
            fitness=float(n),
        )

    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=fail_on_odd,
        tracker=_tracker(),
        n_workers=3,
    )
    ex.start()
    try:
        for i in range(6):
            assert ex.submit(_task(i))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.completed == 3
    assert stats.failed == 3
    assert len(store) == 6


def test_evaluator_returning_failed_status(store: EvaluationStore) -> None:
    def ev(task: Task) -> TaskResult:
        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.FAILED,
            error="soft failure",
        )

    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=ev,
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    try:
        assert ex.submit(_task(0))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.failed == 1
    rec = store.get("run_a", "fp0")
    assert rec is not None
    assert rec.status is EvaluationStatus.FAILED
    assert rec.error == "soft failure"


# --- budget integration (D-032) ---

def test_budget_exhaustion_rejects_submission(
    store: EvaluationStore,
) -> None:
    tracker = _tracker(total=3)
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=tracker,
        n_workers=2,
    )
    ex.start()
    try:
        assert ex.submit(_task(0))
        assert ex.submit(_task(1))
        assert ex.submit(_task(2))
        assert not ex.submit(_task(3))   # over budget
        assert not ex.submit(_task(4))
        stats = ex.run_until_drained()
    finally:
        ex.shutdown()

    assert stats.submitted == 3
    assert stats.skipped == 2
    assert stats.budget_exhausted is True
    assert len(store) == 3


def test_budget_consumed_at_submit_not_completion(
    store: EvaluationStore,
) -> None:
    """Two workers must not both see 'last slot'. D-032."""
    tracker = _tracker(total=2)
    started = threading.Event()
    release = threading.Event()

    def slow(task: Task) -> TaskResult:
        started.set()
        release.wait(timeout=2.0)
        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.SUCCESS,
            fitness=1.0,
        )

    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=slow,
        tracker=tracker,
        n_workers=2,
    )
    ex.start()
    try:
        assert ex.submit(_task(0))
        assert ex.submit(_task(1))
        # Both workers may be mid-evaluation; the third submit must be
        # rejected based on the submit-time counter, regardless of
        # whether the first two have completed.
        assert not ex.submit(_task(2))
        release.set()
        ex.run_until_drained()
    finally:
        release.set()
        ex.shutdown()


# --- concurrency ---

def test_workers_run_concurrently(store: EvaluationStore) -> None:
    peak = 0
    current = 0
    lock = threading.Lock()

    def ev(task: Task) -> TaskResult:
        nonlocal peak, current
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.05)
        with lock:
            current -= 1
        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.SUCCESS,
            fitness=1.0,
        )

    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=ev,
        tracker=_tracker(),
        n_workers=4,
    )
    ex.start()
    try:
        for i in range(8):
            assert ex.submit(_task(i))
        ex.run_until_drained()
    finally:
        ex.shutdown()

    assert peak >= 2, f"expected concurrency, peak={peak}"


# --- lifecycle ---

def test_submit_before_start_raises(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=1,
    )
    with pytest.raises(ExecutorError):
        ex.submit(_task(0))


def test_run_until_drained_before_start_raises(
    store: EvaluationStore,
) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=1,
    )
    with pytest.raises(ExecutorError):
        ex.run_until_drained()


def test_start_is_idempotent(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    ex.start()  # must not spawn a second batch
    try:
        assert ex.submit(_task(0))
        ex.run_until_drained()
    finally:
        ex.shutdown()


def test_shutdown_is_idempotent(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    ex.shutdown()
    ex.shutdown()  # must not raise


def test_n_workers_must_be_positive(store: EvaluationStore) -> None:
    with pytest.raises(ValueError):
        Executor(
            run_id="run_a",
            store=store,
            evaluator=_ok_evaluator(),
            tracker=_tracker(),
            n_workers=0,
        )


# --- stats immutability ---

def test_stats_snapshot_is_frozen() -> None:
    s = ExecutorStats(
        submitted=1, completed=1, failed=0, skipped=0,
        budget_exhausted=False,
    )
    with pytest.raises(Exception):
        s.submitted = 5  # type: ignore[misc]


def test_stats_returns_fresh_snapshot(store: EvaluationStore) -> None:
    ex = Executor(
        run_id="run_a",
        store=store,
        evaluator=_ok_evaluator(),
        tracker=_tracker(),
        n_workers=2,
    )
    ex.start()
    try:
        s0 = ex.stats()
        assert s0.submitted == 0
        assert ex.submit(_task(0))
        ex.run_until_drained()
        s1 = ex.stats()
        assert s1.submitted == 1
        assert s0.submitted == 0  # snapshot not mutated retroactively
    finally:
        ex.shutdown()