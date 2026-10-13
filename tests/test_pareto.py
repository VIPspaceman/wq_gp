"""Tests for Pareto multi-objective selection. Phase 4 Step 6 gate."""
from __future__ import annotations

import math
import random

import pytest

from wqgp.ast import ASTNode
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.pareto import (
    DEFAULT_OBJECTIVES,
    Objective,
    ObjectiveDirection,
    ParetoConfig,
    ParetoRanking,
    ParetoSelector,
)
from wqgp.population import Individual, Population
from wqgp.types import DataType, SemanticKind


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


def _ast(fid: str) -> ASTNode:
    return ASTNode.field_ref(_field(fid))


def _ind(
    fid: str,
    *,
    sharpe: float | None = None,
    turnover: float | None = None,
    fitness: float | None = None,
    generation: int = 0,
) -> Individual:
    metrics: list[tuple[str, float]] = []
    if sharpe is not None:
        metrics.append(("sharpe", sharpe))
    if turnover is not None:
        metrics.append(("turnover", turnover))
    if fitness is None and sharpe is not None:
        fitness = sharpe
    if fitness is None:
        return Individual.from_ast(_ast(fid), generation=generation)
    return Individual.from_ast(
        _ast(fid),
        fitness=fitness,
        metrics=tuple(metrics),
        status=EvaluationStatus.SUCCESS,
        generation=generation,
    )


# --- Objective --------------------------------------------------------------

def test_objective_rejects_empty_metric() -> None:
    with pytest.raises(ValueError):
        Objective("")


def test_objective_default_direction_is_maximize() -> None:
    assert (
        Objective("sharpe").direction is ObjectiveDirection.MAXIMIZE
    )


def test_objective_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        Objective("sharpe", "UP")  # type: ignore[arg-type]


def test_objective_extract_from_metrics() -> None:
    ind = _ind("a", sharpe=1.5, turnover=0.3)
    assert Objective("sharpe").extract(ind) == 1.5
    assert Objective("turnover").extract(ind) == 0.3


def test_objective_extract_sharpe_falls_back_to_fitness() -> None:
    """When metrics do not carry sharpe, use the Individual's fitness."""
    ind = _ind("a", fitness=2.0)  # no metrics
    assert Objective("sharpe").extract(ind) == 2.0


def test_objective_extract_missing_maximize_is_neg_inf() -> None:
    ind = _ind("a", sharpe=1.0)
    assert Objective("missing").extract(ind) == -math.inf


def test_objective_extract_missing_minimize_is_pos_inf() -> None:
    ind = _ind("a", sharpe=1.0)
    obj = Objective("missing", ObjectiveDirection.MINIMIZE)
    assert obj.extract(ind) == math.inf


# --- ParetoConfig -----------------------------------------------------------

def test_config_rejects_empty_objectives() -> None:
    with pytest.raises(ValueError):
        ParetoConfig(())


def test_config_rejects_duplicate_metric() -> None:
    with pytest.raises(ValueError):
        ParetoConfig((
            Objective("sharpe"),
            Objective("sharpe", ObjectiveDirection.MINIMIZE),
        ))


def test_config_default_has_sharpe_and_turnover() -> None:
    cfg = ParetoConfig.default()
    names = [o.metric for o in cfg.objectives]
    assert "sharpe" in names
    assert "turnover" in names


def test_default_objectives_constant() -> None:
    assert len(DEFAULT_OBJECTIVES) == 2


# --- prepare ---------------------------------------------------------------

def test_prepare_empty_population() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    ranking = selector.prepare(Population.empty())
    assert dict(ranking.front_index) == {}
    assert dict(ranking.crowding) == {}


def test_prepare_single_individual() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    pop = Population([_ind("a", sharpe=1.0, turnover=0.3)])
    ranking = selector.prepare(pop)
    ind = pop.all()[0]
    assert ranking.front_of(ind.fingerprint) == 0
    assert ranking.crowding_of(ind.fingerprint) == math.inf


def test_prepare_two_nondominated_individuals_same_front() -> None:
    """A: high sharpe, high turnover. B: low sharpe, low turnover.
    Neither dominates the other; both are in front 0."""
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=2.0, turnover=0.5)
    b = _ind("b", sharpe=1.0, turnover=0.1)
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    assert ranking.front_of(a.fingerprint) == 0
    assert ranking.front_of(b.fingerprint) == 0


def test_prepare_dominated_individual_in_later_front() -> None:
    """A dominates B: A has higher sharpe and lower turnover."""
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=2.0, turnover=0.1)
    b = _ind("b", sharpe=1.0, turnover=0.5)
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    assert ranking.front_of(a.fingerprint) == 0
    assert ranking.front_of(b.fingerprint) == 1


def test_prepare_three_layer_fronts() -> None:
    """A dominates B dominates C on both objectives."""
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=3.0, turnover=0.1)
    b = _ind("b", sharpe=2.0, turnover=0.3)
    c = _ind("c", sharpe=1.0, turnover=0.5)
    pop = Population([a, b, c])
    ranking = selector.prepare(pop)
    assert ranking.front_of(a.fingerprint) == 0
    assert ranking.front_of(b.fingerprint) == 1
    assert ranking.front_of(c.fingerprint) == 2


def test_prepare_identical_individuals_share_front_zero() -> None:
    """No dominance between equal-valued individuals."""
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=1.0, turnover=0.3)
    b = _ind("b", sharpe=1.0, turnover=0.3)
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    assert ranking.front_of(a.fingerprint) == 0
    assert ranking.front_of(b.fingerprint) == 0


# --- crowding --------------------------------------------------------------

def test_crowding_boundary_individuals_get_infinity() -> None:
    """Within a single front, the min and max on each objective get inf.

    The three individuals must be mutually non-dominating so they end
    up in the same front. A naive "all sharpe decreasing, turnover
    increasing" pattern produces three separate fronts (a dominates b
    dominates c), and each single-member front trivially gets
    crowding=inf. See D-131.
    """
    selector = ParetoSelector(ParetoConfig.default())
    # Mutually non-dominating: high sharpe comes with high turnover.
    a = _ind("a", sharpe=3.0, turnover=0.5)
    b = _ind("b", sharpe=2.0, turnover=0.3)
    c = _ind("c", sharpe=1.0, turnover=0.1)
    pop = Population([a, b, c])
    ranking = selector.prepare(pop)

    # All three in the same front.
    assert ranking.front_of(a.fingerprint) == 0
    assert ranking.front_of(b.fingerprint) == 0
    assert ranking.front_of(c.fingerprint) == 0

    # a and c are boundaries on both objectives -> inf.
    assert ranking.crowding_of(a.fingerprint) == math.inf
    assert ranking.crowding_of(c.fingerprint) == math.inf
    # b is interior on both -> finite.
    assert math.isfinite(ranking.crowding_of(b.fingerprint))


def test_crowding_interior_accumulates_both_objectives() -> None:
    """The interior point's crowding is the sum of its normalized gaps
    across every objective.

    With mutually non-dominating inputs, all three live in front 0, and
    b is interior on both axes:

      sharpe   ascending [1.0, 2.0, 3.0], range 2.0
               b's gap = (3.0 - 1.0) / 2.0 = 1.0
      turnover ascending [0.1, 0.3, 0.5], range 0.4
               b's gap = (0.5 - 0.1) / 0.4 = 1.0
      total = 2.0
    """
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=3.0, turnover=0.5)
    b = _ind("b", sharpe=2.0, turnover=0.3)
    c = _ind("c", sharpe=1.0, turnover=0.1)
    pop = Population([a, b, c])
    ranking = selector.prepare(pop)
    assert abs(ranking.crowding_of(b.fingerprint) - 2.0) < 1e-9


# --- select_elites ----------------------------------------------------------

def test_select_elites_zero() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    pop = Population([_ind("a", sharpe=1.0, turnover=0.3)])
    ranking = selector.prepare(pop)
    assert selector.select_elites(pop, ranking, 0) == ()


def test_select_elites_negative_raises() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    pop = Population([_ind("a", sharpe=1.0, turnover=0.3)])
    ranking = selector.prepare(pop)
    with pytest.raises(ValueError):
        selector.select_elites(pop, ranking, -1)


def test_select_elites_prefers_front_zero() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=2.0, turnover=0.1)  # front 0
    b = _ind("b", sharpe=1.0, turnover=0.5)  # front 1
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    elites = selector.select_elites(pop, ranking, 1)
    assert len(elites) == 1
    assert elites[0].fingerprint == a.fingerprint


def test_select_elites_fills_across_fronts() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=3.0, turnover=0.1)   # front 0
    b = _ind("b", sharpe=2.0, turnover=0.5)   # front 0
    c = _ind("c", sharpe=1.0, turnover=0.2)   # front 1
    d = _ind("d", sharpe=0.5, turnover=0.9)   # front 2
    pop = Population([a, b, c, d])
    ranking = selector.prepare(pop)
    elites = selector.select_elites(pop, ranking, 3)
    fps = {e.fingerprint for e in elites}
    assert a.fingerprint in fps
    assert b.fingerprint in fps
    assert c.fingerprint in fps
    # d is left out.
    assert d.fingerprint not in fps


def test_select_elites_deterministic() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=3.0, turnover=0.1)
    b = _ind("b", sharpe=3.0, turnover=0.1)
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    e1 = selector.select_elites(pop, ranking, 1)
    e2 = selector.select_elites(pop, ranking, 1)
    assert e1 == e2


# --- tournament_select ------------------------------------------------------

def test_tournament_empty_population_raises() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    ranking = selector.prepare(Population.empty())
    with pytest.raises(ValueError):
        selector.tournament_select(
            Population.empty(), ranking, 3, random.Random(0)
        )


def test_tournament_rejects_zero_k() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    pop = Population([_ind("a", sharpe=1.0, turnover=0.3)])
    ranking = selector.prepare(pop)
    with pytest.raises(ValueError):
        selector.tournament_select(pop, ranking, 0, random.Random(0))


def test_tournament_uses_all_when_k_exceeds_size() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    a = _ind("a", sharpe=2.0, turnover=0.1)   # front 0
    b = _ind("b", sharpe=1.0, turnover=0.5)   # front 1
    pop = Population([a, b])
    ranking = selector.prepare(pop)
    # k=100 >= 2 -> all sampled -> a wins (front 0)
    for seed in range(10):
        picked = selector.tournament_select(
            pop, ranking, 100, random.Random(seed)
        )
        assert picked.fingerprint == a.fingerprint


def test_tournament_reproducible() -> None:
    selector = ParetoSelector(ParetoConfig.default())
    pop = Population([
        _ind("a", sharpe=1.0, turnover=0.5),
        _ind("b", sharpe=2.0, turnover=0.3),
        _ind("c", sharpe=3.0, turnover=0.1),
    ])
    ranking = selector.prepare(pop)
    a = selector.tournament_select(pop, ranking, 2, random.Random(0))
    b = selector.tournament_select(pop, ranking, 2, random.Random(0))
    assert a.fingerprint == b.fingerprint


# --- integration with next_generation -------------------------------------

def test_selection_config_accepts_pareto_mode() -> None:
    from wqgp.selection import SelectionConfig
    cfg = SelectionConfig(population_size=4, mode="pareto")
    assert cfg.mode == "pareto"


def test_selection_config_rejects_bad_mode() -> None:
    from wqgp.selection import SelectionConfig
    with pytest.raises(ValueError):
        SelectionConfig(population_size=4, mode="fancy")


def test_selection_config_rejects_bad_objectives_shape() -> None:
    from wqgp.selection import SelectionConfig
    with pytest.raises(ValueError):
        SelectionConfig(
            population_size=4,
            mode="pareto",
            pareto_objectives=(("sharpe",),),  # type: ignore[arg-type]
        )


def test_selection_config_default_mode_is_single() -> None:
    from wqgp.selection import SelectionConfig
    assert SelectionConfig(population_size=4).mode == "single"


def test_next_generation_pareto_produces_population() -> None:
    from wqgp.crossover import Crossover
    from wqgp.mutation import Mutator
    from wqgp.pool import FieldPool, OperatorPool, Pool
    from wqgp.selection import SelectionConfig, next_generation

    pool = Pool(
        fields=FieldPool([
            _field("a"), _field("b"), _field("c"),
            _field("d"), _field("e"), _field("f"),
        ]),
        operators=OperatorPool([_add()]),
    )
    parents = Population([
        _ind("a", sharpe=2.0, turnover=0.1),
        _ind("b", sharpe=1.5, turnover=0.2),
        _ind("c", sharpe=1.0, turnover=0.4),
        _ind("d", sharpe=0.8, turnover=0.5),
    ])
    config = SelectionConfig(
        population_size=6,
        elite_count=2,
        tournament_size=2,
        crossover_rate=0.5,
        max_breeding_attempts=200,
        mode="pareto",
    )
    rng = random.Random(0)
    result = next_generation(
        parents,
        Mutator(pool),
        Crossover(pool),
        config,
        rng,
    )
    assert len(result) >= 1
    # Every result must have a unique fingerprint.
    fps = [i.fingerprint for i in result]
    assert len(fps) == len(set(fps))


def test_next_generation_pareto_deterministic() -> None:
    from wqgp.crossover import Crossover
    from wqgp.mutation import Mutator
    from wqgp.pool import FieldPool, OperatorPool, Pool
    from wqgp.selection import SelectionConfig, next_generation

    pool = Pool(
        fields=FieldPool([
            _field("a"), _field("b"), _field("c"),
            _field("d"), _field("e"),
        ]),
        operators=OperatorPool([_add()]),
    )
    parents = Population([
        _ind("a", sharpe=2.0, turnover=0.1),
        _ind("b", sharpe=1.5, turnover=0.2),
        _ind("c", sharpe=1.0, turnover=0.4),
    ])
    config = SelectionConfig(
        population_size=5,
        elite_count=1,
        tournament_size=2,
        crossover_rate=0.5,
        max_breeding_attempts=200,
        mode="pareto",
    )

    def run_once():
        return next_generation(
            parents,
            Mutator(pool),
            Crossover(pool),
            config,
            random.Random(12345),
        )

    r1 = run_once()
    r2 = run_once()
    assert sorted(i.fingerprint for i in r1) == sorted(
        i.fingerprint for i in r2
    )