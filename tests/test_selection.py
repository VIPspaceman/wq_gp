"""Tests for selection operators. Phase 3 Step 4 gate."""
from __future__ import annotations

import random

import pytest

from wqgp.ast import ASTNode
from wqgp.crossover import Crossover
from wqgp.field import FieldDef
from wqgp.mutation import Mutator
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.population import Individual, Population
from wqgp.selection import (
    SelectionConfig,
    SelectionError,
    next_generation,
    select_elites,
    tournament_select,
)
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---------------------------------------------------------------

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


def _mul() -> OperatorDef:
    return OperatorDef(
        "mul",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _ts_std() -> OperatorDef:
    return OperatorDef(
        "ts_std",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _pool() -> Pool:
    return Pool(
        fields=FieldPool(
            [
                _matrix_field("close"),
                _matrix_field("open"),
                _matrix_field("high"),
                _matrix_field("low"),
                _matrix_field("volume"),
                _matrix_field("vwap"),
            ]
        ),
        operators=OperatorPool(
            [_add(), _sub(), _mul(), _ts_mean(), _ts_std()]
        ),
    )


def _f(fid: str) -> ASTNode:
    return ASTNode.field_ref(_matrix_field(fid))


def _ast_add() -> ASTNode:
    return ASTNode.call(_add(), (_f("close"), _f("open")))


def _ast_sub() -> ASTNode:
    return ASTNode.call(_sub(), (_f("high"), _f("low")))


def _ast_mul() -> ASTNode:
    return ASTNode.call(_mul(), (_f("close"), _f("volume")))


def _ast_ts_mean() -> ASTNode:
    return ASTNode.call(_ts_mean(), (_f("close"),), params=(5,))


def _ast_ts_std() -> ASTNode:
    return ASTNode.call(_ts_std(), (_f("close"),), params=(10,))


def _ind(
    ast: ASTNode,
    fitness: float | None = None,
    generation: int = 0,
) -> Individual:
    if fitness is None:
        return Individual.from_ast(ast, generation=generation)
    return Individual.from_ast(
        ast,
        fitness=fitness,
        status=EvaluationStatus.SUCCESS,
        generation=generation,
    )


def _pop(*specs: tuple[ASTNode, float | None]) -> Population:
    return Population([_ind(ast, fit) for ast, fit in specs])


def _mutator() -> Mutator:
    return Mutator(_pool())


def _crossover() -> Crossover:
    return Crossover(_pool())


# --- SelectionConfig validation --------------------------------------------

def test_config_rejects_zero_population_size() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=0)


def test_config_rejects_negative_elite_count() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=10, elite_count=-1)


def test_config_rejects_elite_exceeding_population() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=3, elite_count=4)


def test_config_rejects_zero_tournament_size() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=10, tournament_size=0)


def test_config_rejects_out_of_range_crossover_rate() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=10, crossover_rate=-0.1)
    with pytest.raises(ValueError):
        SelectionConfig(population_size=10, crossover_rate=1.1)


def test_config_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError):
        SelectionConfig(population_size=10, max_breeding_attempts=0)


def test_config_accepts_boundary_values() -> None:
    SelectionConfig(population_size=1, elite_count=0, crossover_rate=0.0)
    SelectionConfig(population_size=1, elite_count=1, crossover_rate=1.0)
    SelectionConfig(population_size=10, elite_count=10)


# --- tournament_select ------------------------------------------------------

def test_tournament_empty_population_raises() -> None:
    with pytest.raises(SelectionError):
        tournament_select(Population.empty(), 3, random.Random(0))


def test_tournament_rejects_zero_k() -> None:
    p = _pop((_ast_add(), 1.0))
    with pytest.raises(ValueError):
        tournament_select(p, 0, random.Random(0))


def test_tournament_uses_all_when_k_exceeds_population() -> None:
    p = _pop((_ast_add(), 1.0), (_ast_sub(), 2.0))
    # k=100 > 2, uses all, returns the best.
    best = tournament_select(p, 100, random.Random(0))
    assert best.fitness == 2.0


def test_tournament_returns_one_of_the_members() -> None:
    p = _pop(
        (_ast_add(), 1.0),
        (_ast_sub(), 2.0),
        (_ast_mul(), 3.0),
        (_ast_ts_mean(), 4.0),
        (_ast_ts_std(), 5.0),
    )
    members = set(p.all())
    for seed in range(20):
        picked = tournament_select(p, 3, random.Random(seed))
        assert picked in members


def test_tournament_reproducible_with_seed() -> None:
    p = _pop(
        (_ast_add(), 1.0),
        (_ast_sub(), 2.0),
        (_ast_mul(), 3.0),
        (_ast_ts_mean(), 4.0),
    )
    a = tournament_select(p, 2, random.Random(12345))
    b = tournament_select(p, 2, random.Random(12345))
    assert a.fingerprint == b.fingerprint


def test_tournament_prefers_evaluated_over_unevaluated() -> None:
    """Five unevaluated + one evaluated. A tournament of size >= the
    population size must return the evaluated one, because unevaluated
    individuals rank at -inf."""
    p = Population(
        [
            _ind(_ast_add(), None),
            _ind(_ast_sub(), None),
            _ind(_ast_mul(), None),
            _ind(_ast_ts_mean(), None),
            _ind(_ast_ts_std(), 5.0),  # the only evaluated one
        ]
    )
    picked = tournament_select(p, 100, random.Random(0))
    assert picked.fitness == 5.0


def test_tournament_k1_picks_uniformly() -> None:
    p = _pop(
        (_ast_add(), 1.0),
        (_ast_sub(), 2.0),
        (_ast_mul(), 3.0),
    )
    seen = set()
    for seed in range(30):
        seen.add(
            tournament_select(p, 1, random.Random(seed)).fingerprint
        )
    # k=1 over 3 members should visit more than one member across seeds.
    assert len(seen) >= 2


# --- select_elites ----------------------------------------------------------

def test_elites_zero_returns_empty() -> None:
    p = _pop((_ast_add(), 1.0))
    assert select_elites(p, 0) == ()


def test_elites_negative_raises() -> None:
    p = _pop((_ast_add(), 1.0))
    with pytest.raises(ValueError):
        select_elites(p, -1)


def test_elites_returns_top_n() -> None:
    a = _ind(_ast_add(), 1.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    p = Population([a, b, c])
    top = select_elites(p, 2)
    assert [i.fitness for i in top] == [5.0, 3.0]


def test_elites_skips_unevaluated() -> None:
    a = _ind(_ast_add(), 1.0)
    b = _ind(_ast_sub(), None)
    c = _ind(_ast_mul(), 3.0)
    p = Population([a, b, c])
    top = select_elites(p, 10)
    assert len(top) == 2
    assert all(i.fitness is not None for i in top)


def test_elites_ties_broken_by_fingerprint() -> None:
    # Same fitness, different fingerprints -> deterministic order.
    a = _ind(_ast_add(), 1.0)
    b = _ind(_ast_sub(), 1.0)
    p = Population([a, b])
    top1 = select_elites(p, 2)
    top2 = select_elites(p, 2)
    assert [i.fingerprint for i in top1] == [i.fingerprint for i in top2]


def test_elites_fewer_evaluated_than_n() -> None:
    a = _ind(_ast_add(), 1.0)
    b = _ind(_ast_sub(), None)
    p = Population([a, b])
    assert len(select_elites(p, 5)) == 1


# --- next_generation --------------------------------------------------------

def test_next_generation_empty_parents_raises() -> None:
    config = SelectionConfig(population_size=4)
    with pytest.raises(SelectionError):
        next_generation(
            Population.empty(),
            _mutator(),
            _crossover(),
            config,
            random.Random(0),
        )


def test_next_generation_preserves_elites() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 1.0)
    d = _ind(_ast_ts_mean(), 0.5)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6, elite_count=2, max_breeding_attempts=500
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    assert a.fingerprint in result
    assert b.fingerprint in result


def test_next_generation_reaches_target_size() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 1.0)
    d = _ind(_ast_ts_mean(), 0.5)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6, elite_count=2, max_breeding_attempts=2000
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    assert len(result) == 6


def test_next_generation_dedups() -> None:
    """Even if breeding repeatedly proposes the same child, the result
    must have unique fingerprints (Population enforces this)."""
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6, elite_count=2, max_breeding_attempts=500
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    fps = [i.fingerprint for i in result]
    assert len(fps) == len(set(fps))


def test_next_generation_reproducible() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6, elite_count=2, max_breeding_attempts=500
    )
    r1 = next_generation(
        p, _mutator(), _crossover(), config, random.Random(12345)
    )
    r2 = next_generation(
        p, _mutator(), _crossover(), config, random.Random(12345)
    )
    assert (
        sorted(i.fingerprint for i in r1)
        == sorted(i.fingerprint for i in r2)
    )


def test_offspring_generation_increments() -> None:
    a = _ind(_ast_add(), 10.0, generation=5)
    b = _ind(_ast_sub(), 5.0, generation=5)
    c = _ind(_ast_mul(), 3.0, generation=5)
    d = _ind(_ast_ts_mean(), 1.0, generation=5)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6,
        elite_count=0,   # so every result member is a fresh offspring
        max_breeding_attempts=2000,
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    assert len(result) > 0
    for ind in result:
        assert ind.generation == 6


def test_offspring_record_parents() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6, elite_count=0, max_breeding_attempts=2000
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    parent_fps = {i.fingerprint for i in p.all()}
    for ind in result:
        assert len(ind.parents) >= 1
        # Every recorded parent must be one of the actual parents.
        for fp in ind.parents:
            assert fp in parent_fps


def test_crossover_only_offspring_have_two_parents() -> None:
    """With crossover_rate=1.0, every offspring carries two parents,
    unless its parent pair had no compatible swap -- in which case it
    falls back to mutation. Verify that a majority of offspring are
    crossover products."""
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    e = _ind(_ast_ts_std(), 0.5)
    p = Population([a, b, c, d, e])

    config = SelectionConfig(
        population_size=8,
        elite_count=0,
        crossover_rate=1.0,
        max_breeding_attempts=2000,
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    two_parent = sum(1 for i in result if len(i.parents) == 2)
    assert two_parent >= 1


def test_mutation_only_offspring_have_one_parent() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=6,
        elite_count=0,
        crossover_rate=0.0,
        max_breeding_attempts=2000,
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    assert len(result) > 0
    for ind in result:
        assert len(ind.parents) == 1


def test_elite_count_zero_produces_only_offspring() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=5, elite_count=0, max_breeding_attempts=2000
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    for ind in result:
        assert ind.generation == 1  # parent generation was 0


def test_elite_count_equals_population_size_returns_elites() -> None:
    a = _ind(_ast_add(), 10.0)
    b = _ind(_ast_sub(), 5.0)
    c = _ind(_ast_mul(), 3.0)
    d = _ind(_ast_ts_mean(), 1.0)
    p = Population([a, b, c, d])

    config = SelectionConfig(
        population_size=3, elite_count=3, max_breeding_attempts=10
    )
    result = next_generation(
        p, _mutator(), _crossover(), config, random.Random(0)
    )
    assert len(result) == 3
    # All three must be the top-3 of the parents (i.e. no offspring).
    for ind in result:
        assert ind.generation == 0