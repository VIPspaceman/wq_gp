"""Concurrent task execution for a GP run.

Design decisions:
- Budget is consumed at SUBMIT time, not at completion. See D-032.
- Evaluator is a Protocol, not an ABC. See D-033.
- Worker swallows all exceptions into a FAILED record. See D-034.
- Task / TaskResult are frozen value objects; TaskStatus is local. See D-035.
- stdlib queue.Queue + sentinel shutdown. See D-036.
- Stats exposed as a frozen snapshot. See D-037.

This module does NOT talk to BRAIN. It runs any callable that maps
`Task -> TaskResult`. Wiring the real BRAIN client is Phase 3's job.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from wqgp.budget import BudgetExhausted, BudgetTracker
from wqgp.persistence import EvaluationRecord, EvaluationStatus, EvaluationStore


class TaskStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of work: evaluate a single alpha expression.

    Invariants:
      - fingerprint non-empty
      - expression non-empty
      - generation >= 0
    """
    fingerprint: str
    expression: str
    generation: int

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("task fingerprint must be non-empty")
        if not self.expression or not self.expression.strip():
            raise ValueError("task expression must be non-empty")
        if self.generation < 0:
            raise ValueError("task generation must be >= 0")


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Result returned by an Evaluator.

    Invariants:
      - SUCCESS requires fitness
      - FAILED requires non-empty error
      - metrics values must be numeric (bool excluded)
    """
    fingerprint: str
    status: TaskStatus
    fitness: float | None = None
    metrics: tuple[tuple[str, float], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("result fingerprint must be non-empty")
        if not isinstance(self.status, TaskStatus):
            raise ValueError(
                f"status must be TaskStatus, got {type(self.status).__name__}"
            )
        for k, v in self.metrics:
            if not isinstance(k, str) or not k:
                raise ValueError("metric name must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"metric {k!r} must be int/float, got {type(v).__name__}"
                )
        if self.status is TaskStatus.SUCCESS and self.fitness is None:
            raise ValueError("SUCCESS TaskResult requires fitness")
        if self.status is TaskStatus.FAILED and (
            not self.error or not self.error.strip()
        ):
            raise ValueError("FAILED TaskResult requires non-empty error")


class Evaluator(Protocol):
    """Callable: Task -> TaskResult.

    Implementations must be thread-safe; the Executor runs them from
    multiple worker threads concurrently.
    """

    def __call__(self, task: Task) -> TaskResult: ...


@dataclass(frozen=True, slots=True)
class ExecutorStats:
    """Immutable snapshot of executor counters at one instant."""
    submitted: int
    completed: int
    failed: int
    skipped: int
    budget_exhausted: bool


@dataclass
class _MutableStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    budget_exhausted: bool = False

    def freeze(self) -> ExecutorStats:
        return ExecutorStats(
            submitted=self.submitted,
            completed=self.completed,
            failed=self.failed,
            skipped=self.skipped,
            budget_exhausted=self.budget_exhausted,
        )


_TASK_STATUS_TO_EVAL = {
    TaskStatus.SUCCESS: EvaluationStatus.SUCCESS,
    TaskStatus.FAILED: EvaluationStatus.FAILED,
    TaskStatus.SKIPPED: EvaluationStatus.SKIPPED,
}


class ExecutorError(RuntimeError):
    """Raised on executor lifecycle misuse (e.g. submit before start)."""


class Executor:
    """Thread-pool executor with budget-aware submission.

    Lifecycle:
        ex = Executor(...)
        ex.start()
        for task in tasks:
            if not ex.submit(task):
                break                # budget exhausted
        ex.run_until_drained()
        ex.shutdown()

    `submit()` returns False if the budget check raised; in that case the
    task is dropped and a `skipped` counter is incremented. The caller
    decides whether to stop producing or continue with a fresh budget.
    """

    __slots__ = (
        "_run_id",
        "_store",
        "_evaluator",
        "_tracker",
        "_n_workers",
        "_queue",
        "_workers",
        "_stats",
        "_stats_lock",
        "_started",
    )

    def __init__(
        self,
        *,
        run_id: str,
        store: EvaluationStore,
        evaluator: Evaluator,
        tracker: BudgetTracker,
        n_workers: int = 4,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self._run_id = run_id
        self._store = store
        self._evaluator = evaluator
        self._tracker = tracker
        self._n_workers = n_workers
        self._queue: queue.Queue[Task | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._stats = _MutableStats()
        self._stats_lock = threading.Lock()
        self._started = False

    # --- lifecycle ---

    def start(self) -> None:
        if self._started:
            return
        for i in range(self._n_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"wqgp-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        self._started = True

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop all workers. Idempotent."""
        if not self._started:
            return
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=timeout)
        self._workers.clear()
        self._started = False

    # --- submission ---

    def submit(self, task: Task) -> bool:
        """Enqueue a task if budget allows. Returns False if not."""
        if not self._started:
            raise ExecutorError("submit() called before start()")
        try:
            self._tracker.consume_simulation(1)
        except BudgetExhausted:
            with self._stats_lock:
                self._stats.skipped += 1
                self._stats.budget_exhausted = True
            return False
        self._queue.put(task)
        with self._stats_lock:
            self._stats.submitted += 1
        return True

    def run_until_drained(self) -> ExecutorStats:
        """Block until every submitted task has been processed."""
        if not self._started:
            raise ExecutorError("run_until_drained() before start()")
        self._queue.join()
        return self.stats()

    def stats(self) -> ExecutorStats:
        with self._stats_lock:
            return self._stats.freeze()

    # --- worker ---

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._process(task)
            finally:
                self._queue.task_done()

    def _process(self, task: Task) -> None:
        try:
            result = self._evaluator(task)
        except Exception as e:  # noqa: BLE001 -- intentional, see D-034
            self._record_failure(
                task, f"{type(e).__name__}: {e}"
            )
            return

        try:
            rec = EvaluationRecord(
                run_id=self._run_id,
                generation=task.generation,
                fingerprint=task.fingerprint,
                expression=task.expression,
                status=_TASK_STATUS_TO_EVAL[result.status],
                fitness=result.fitness,
                metrics=result.metrics,
                error=result.error,
            )
            self._store.record(rec)
        except Exception as e:  # noqa: BLE001
            # Store write failed: log as a failed task, keep the pipeline
            # alive. Losing one record beats killing a worker.
            self._record_failure(
                task, f"store write failed: {type(e).__name__}: {e}"
            )
            return

        with self._stats_lock:
            if result.status is TaskStatus.SUCCESS:
                self._stats.completed += 1
            elif result.status is TaskStatus.FAILED:
                self._stats.failed += 1
            else:
                self._stats.skipped += 1

    def _record_failure(self, task: Task, error: str) -> None:
        try:
            self._store.record(
                EvaluationRecord(
                    run_id=self._run_id,
                    generation=task.generation,
                    fingerprint=task.fingerprint,
                    expression=task.expression,
                    status=EvaluationStatus.FAILED,
                    error=error,
                )
            )
        except Exception:  # noqa: BLE001
            # If even this write fails, we cannot do anything useful.
            # Do NOT re-raise: a dead worker thread is worse than a lost
            # record. The stats counter still reflects the failure.
            pass
        with self._stats_lock:
            self._stats.failed += 1