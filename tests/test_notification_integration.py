"""Coordinator + Notifier integration tests. Phase 4 Step 3 gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wqgp.brain.settings import BrainSettings
from wqgp.budget import Budget
from wqgp.coordinator import Coordinator, RunConfig
from wqgp.evaluation_config import EvaluationConfig
from wqgp.execution import Task, TaskResult, TaskStatus
from wqgp.field import FieldDef
from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.selection import SelectionConfig
from wqgp.types import DataType, ParamDomain, SemanticKind


class RecordingNotifier:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def on_run_start(self, event: RunStartEvent) -> None:
        self.events.append(event)

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        self.events.append(event)

    def on_run_end(self, event: RunEndEvent) -> None:
        self.events.append(event)


class RaisingNotifier:
    def on_run_start(self, event: RunStartEvent) -> None:
        raise RuntimeError("boom on_start")

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        raise RuntimeError("boom on_gen")

    def on_run_end(self, event: RunEndEvent) -> None:
        raise RuntimeError("boom on_end")


def _matrix_field(fid: str) -> FieldDef:
    return FieldDef(fid, DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _add() -> OperatorDef:
    return OperatorDef(
        "add",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _sub() -> OperatorDef:
    return OperatorDef(
        "sub", (DataType.MATRIX, DataType.MATRIX), DataType.MATRIX,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _pool() -> Pool:
    return Pool(
        fields=FieldPool([
            _matrix_field("close"),
            _matrix_field("open"),
            _matrix_field("high"),
            _matrix_field("low"),
            _matrix_field("volume"),
            _matrix_field("vwap"),
        ]),
        operators=OperatorPool([_add(), _sub(), _ts_mean()]),
    )


def _config(
    tmp_path: Path,
    *,
    run_id: str = "run_notif",
    max_generations: int = 2,
    pop_size: int = 4,
    total_sims: int = 100,
    seed: int = 42,
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        pool=_pool(),
        budget=Budget(total_simulations=total_sims),
        selection=SelectionConfig(
            population_size=pop_size,
            elite_count=1,
            tournament_size=2,
            crossover_rate=0.5,
            max_breeding_attempts=200,
        ),
        seed=seed,
        evaluation_config=EvaluationConfig(BrainSettings()),
        max_generations=max_generations,
        checkpoint_root=tmp_path / "ckpt",
        evaluation_db=tmp_path / "eval.db",
        n_workers=2,
        seed_count=0,
    )


def _deterministic_evaluator(task: Task) -> TaskResult:
    fit = float(sum(ord(c) for c in task.expression) % 100) / 10.0
    return TaskResult(
        fingerprint=task.fingerprint,
        status=TaskStatus.SUCCESS,
        fitness=fit,
        metrics=(("sharpe", fit / 2.0),),
    )


def test_events_emitted_in_order(tmp_path: Path) -> None:
    rec = RecordingNotifier()
    cfg = _config(tmp_path, max_generations=2)
    Coordinator(cfg, _deterministic_evaluator, rec).run()

    kinds = [type(e).__name__ for e in rec.events]
    assert kinds[0] == "RunStartEvent"
    assert kinds[-1] == "RunEndEvent"
    assert all(k == "GenerationEndEvent" for k in kinds[1:-1])
    assert len(kinds) >= 3


def test_run_start_reports_new_vs_resuming(tmp_path: Path) -> None:
    rec = RecordingNotifier()
    cfg = _config(tmp_path, max_generations=1)
    Coordinator(cfg, _deterministic_evaluator, rec).run()

    start = rec.events[0]
    assert isinstance(start, RunStartEvent)
    assert start.run_id == "run_notif"
    assert start.generation == 0
    assert start.resuming is False


def test_run_start_reports_resuming(tmp_path: Path) -> None:
    cfg1 = _config(tmp_path, max_generations=2)
    Coordinator(cfg1, _deterministic_evaluator).run()

    rec = RecordingNotifier()
    cfg2 = _config(tmp_path, max_generations=3)
    Coordinator(cfg2, _deterministic_evaluator, rec).run()

    start = rec.events[0]
    assert isinstance(start, RunStartEvent)
    assert start.resuming is True
    assert start.generation == 1


def test_generation_end_contains_fitness(tmp_path: Path) -> None:
    rec = RecordingNotifier()
    cfg = _config(tmp_path, max_generations=1)
    Coordinator(cfg, _deterministic_evaluator, rec).run()

    gen_events = [
        e for e in rec.events if isinstance(e, GenerationEndEvent)
    ]
    assert len(gen_events) >= 1
    first = gen_events[0]
    assert first.evaluated_count >= 1
    assert first.best_fitness is not None
    assert first.total_simulations_used >= 1


def test_run_end_reports_reason(tmp_path: Path) -> None:
    rec = RecordingNotifier()
    cfg = _config(tmp_path, max_generations=2)
    Coordinator(cfg, _deterministic_evaluator, rec).run()

    end = rec.events[-1]
    assert isinstance(end, RunEndEvent)
    assert end.run_id == "run_notif"
    assert end.stopped_reason in {
        "max_generations", "budget_exhausted", "no_progress",
    }
    assert end.population_size >= 1


def test_no_notifier_means_null_notifier(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_generations=1)
    result = Coordinator(cfg, _deterministic_evaluator).run()
    assert result.stopped_reason in {
        "max_generations", "budget_exhausted",
    }


def test_raising_notifier_does_not_break_run(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_generations=2)
    result = Coordinator(
        cfg, _deterministic_evaluator, RaisingNotifier()
    ).run()
    assert len(result.population) >= 1
    assert result.total_simulations_used >= 1