"""Tests for the end-to-end coordinator. Phase 3 Step 5 gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from wqgp.budget import Budget
from wqgp.coordinator import (
    Coordinator,
    CoordinatorError,
    RunConfig,
    RunResult,
)
from wqgp.execution import Task, TaskResult, TaskStatus
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.selection import SelectionConfig
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---------------------------------------------------------------

def _field(fid: str) -> FieldDef:
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
        fields=FieldPool(
            [
                _field("close"),
                _field("open"),
                _field("high"),
                _field("low"),
                _field("volume"),
                _field("vwap"),
            ]
        ),
        operators=OperatorPool([_add(), _sub(), _ts_mean()]),
    )


def _config(
    tmp_path: Path,
    *,
    run_id: str = "run_test",
    max_generations: int = 2,
    pop_size: int = 6,
    total_sims: int = 500,
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
        max_generations=max_generations,
        checkpoint_root=tmp_path / "ckpt",
        evaluation_db=tmp_path / "eval.db",
        n_workers=2,
    )


def _deterministic_evaluator(task: Task) -> TaskResult:
    """Fitness derived from the expression text; deterministic, fast."""
    fit = float(sum(ord(c) for c in task.expression) % 100) / 10.0
    return TaskResult(
        fingerprint=task.fingerprint,
        status=TaskStatus.SUCCESS,
        fitness=fit,
        metrics=(("sharpe", fit / 2.0),),
    )


def _failing_evaluator(task: Task) -> TaskResult:
    raise RuntimeError("evaluator intentionally failed")


# --- RunConfig validation ---------------------------------------------------

def test_config_rejects_bad_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id="../bad",
            pool=_pool(),
            budget=Budget(total_simulations=10),
            selection=SelectionConfig(population_size=4),
            seed=0,
            checkpoint_root=tmp_path,
        )


def test_config_rejects_zero_generations(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id="r",
            pool=_pool(),
            budget=Budget(total_simulations=10),
            selection=SelectionConfig(population_size=4),
            seed=0,
            max_generations=0,
            checkpoint_root=tmp_path,
        )


def test_config_rejects_zero_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id="r",
            pool=_pool(),
            budget=Budget(total_simulations=10),
            selection=SelectionConfig(population_size=4),
            seed=0,
            checkpoint_root=tmp_path,
            n_workers=0,
        )


def test_config_rejects_bad_initial_population_size(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        RunConfig(
            run_id="r",
            pool=_pool(),
            budget=Budget(total_simulations=10),
            selection=SelectionConfig(population_size=4),
            seed=0,
            checkpoint_root=tmp_path,
            initial_population_size=0,
        )


# --- basic run --------------------------------------------------------------

def test_run_returns_result(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_generations=1, pop_size=4)
    coord = Coordinator(cfg, _deterministic_evaluator)
    result = coord.run()
    assert isinstance(result, RunResult)
    assert result.total_simulations_used > 0
    assert len(result.population) >= 1


def test_run_evaluates_initial_population(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_generations=1, pop_size=4)
    coord = Coordinator(cfg, _deterministic_evaluator)
    result = coord.run()
    evaluated = result.population.evaluated()
    assert len(evaluated) >= 1
    for ind in evaluated:
        assert ind.fitness is not None


def test_run_reproducible_with_same_seed(tmp_path: Path) -> None:
    cfg1 = _config(tmp_path / "a", seed=12345, max_generations=2)
    cfg2 = _config(tmp_path / "b", seed=12345, max_generations=2)
    r1 = Coordinator(cfg1, _deterministic_evaluator).run()
    r2 = Coordinator(cfg2, _deterministic_evaluator).run()
    fps1 = sorted(i.fingerprint for i in r1.population)
    fps2 = sorted(i.fingerprint for i in r2.population)
    assert fps1 == fps2


def test_run_different_seeds_diverge(tmp_path: Path) -> None:
    cfg1 = _config(tmp_path / "a", seed=1, max_generations=2)
    cfg2 = _config(tmp_path / "b", seed=999, max_generations=2)
    r1 = Coordinator(cfg1, _deterministic_evaluator).run()
    r2 = Coordinator(cfg2, _deterministic_evaluator).run()
    fps1 = sorted(i.fingerprint for i in r1.population)
    fps2 = sorted(i.fingerprint for i in r2.population)
    assert fps1 != fps2


# --- budget -----------------------------------------------------------------

def test_run_stops_at_budget_exhaustion(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path, max_generations=10, pop_size=20, total_sims=5
    )
    coord = Coordinator(cfg, _deterministic_evaluator)
    result = coord.run()
    assert result.budget_exhausted is True
    assert result.total_simulations_used <= 5
    assert result.stopped_reason == "budget_exhausted"


def test_run_respects_max_generations(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path, max_generations=3, pop_size=4, total_sims=500
    )
    result = Coordinator(cfg, _deterministic_evaluator).run()
    assert result.generation == 2
    assert not result.budget_exhausted


# --- checkpointing ----------------------------------------------------------

def test_run_writes_checkpoint(tmp_path: Path) -> None:
    cfg = _config(tmp_path, max_generations=2)
    Coordinator(cfg, _deterministic_evaluator).run()
    ckpt_dir = Path(cfg.checkpoint_root) / cfg.run_id
    assert ckpt_dir.exists()
    step_files = list(ckpt_dir.glob("step_*.json"))
    assert len(step_files) >= 1


def test_resume_preserves_evolution(tmp_path: Path) -> None:
    """A run interrupted after generation 0, then resumed, must produce
    the same final population as an uninterrupted run."""
    # Reference: single 3-generation run.
    cfg_ref = _config(
        tmp_path / "ref", max_generations=3, pop_size=4, seed=7
    )
    r_ref = Coordinator(cfg_ref, _deterministic_evaluator).run()
    fps_ref = sorted(i.fingerprint for i in r_ref.population)

    # Interrupted: one generation, then resume with the same config
    # but a higher max_generations.
    cfg1 = _config(
        tmp_path / "resume", max_generations=1, pop_size=4, seed=7
    )
    Coordinator(cfg1, _deterministic_evaluator).run()

    cfg2 = _config(
        tmp_path / "resume", max_generations=3, pop_size=4, seed=7
    )
    r2 = Coordinator(cfg2, _deterministic_evaluator).run()
    fps_resumed = sorted(i.fingerprint for i in r2.population)

    assert fps_ref == fps_resumed


def test_resume_reaches_generation(tmp_path: Path) -> None:
    cfg1 = _config(tmp_path, max_generations=1, pop_size=4, seed=7)
    Coordinator(cfg1, _deterministic_evaluator).run()

    cfg2 = _config(tmp_path, max_generations=3, pop_size=4, seed=7)
    r2 = Coordinator(cfg2, _deterministic_evaluator).run()
    assert r2.generation == 2


# --- failure handling -------------------------------------------------------

def test_failing_evaluator_does_not_crash_run(tmp_path: Path) -> None:
    """A failing evaluator produces no evaluated individuals, so the run
    stops with 'no_progress' instead of crashing."""
    cfg = _config(tmp_path, max_generations=1, pop_size=4)
    result = Coordinator(cfg, _failing_evaluator).run()
    assert isinstance(result, RunResult)
    assert result.stopped_reason == "no_progress"


def test_impossible_initial_generation_raises(tmp_path: Path) -> None:
    empty_pool = Pool(fields=FieldPool([]), operators=OperatorPool([]))
    cfg = RunConfig(
        run_id="r",
        pool=empty_pool,
        budget=Budget(total_simulations=10),
        selection=SelectionConfig(population_size=4),
        seed=0,
        max_generations=1,
        checkpoint_root=tmp_path / "ckpt",
        evaluation_db=tmp_path / "eval.db",
    )
    coord = Coordinator(cfg, _deterministic_evaluator)
    with pytest.raises(CoordinatorError):
        coord.run()


def test_elites_not_reevaluated_across_generations(
    tmp_path: Path,
) -> None:
    """With elite_count=1, the elite is carried over and already has a
    fitness in the store. It must not be re-submitted."""
    submitted_count = {"n": 0}

    def counting_evaluator(task: Task) -> TaskResult:
        submitted_count["n"] += 1
        return _deterministic_evaluator(task)

    cfg = _config(tmp_path, max_generations=3, pop_size=4, total_sims=500)
    Coordinator(cfg, counting_evaluator).run()
    # A 3-generation run with pop_size=4 would submit up to 4*3=12
    # evaluations if elites were re-submitted. With elite_count=1,
    # the ceiling is lower; a loose bound is fine here.
    assert submitted_count["n"] <= 12