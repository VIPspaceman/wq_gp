"""Tests for individuals and populations. Phase 3 Step 1 gate."""
from __future__ import annotations

import random

import pytest

from wqgp.ast import ASTNode, NodeKind
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.population import (
    DuplicateIndividualError,
    GenerationError,
    Individual,
    Population,
    RandomAstBuilder,
)
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---------------------------------------------------------------

def _close() -> FieldDef:
    return FieldDef("close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _open() -> FieldDef:
    return FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _sector() -> FieldDef:
    return FieldDef(
        "sector", DataType.GROUP, SemanticKind.CATEGORICAL, "pv1"
    )


def _add() -> OperatorDef:
    return OperatorDef(
        "add",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=True,
        associative=True,
        symmetric=True,
    )


def _sub() -> OperatorDef:
    return OperatorDef(
        "sub",
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _group_rank() -> OperatorDef:
    return OperatorDef(
        "group_rank",
        (DataType.MATRIX, DataType.GROUP),
        DataType.MATRIX,
    )


def _pool() -> Pool:
    return Pool(
        fields=FieldPool([_close(), _open(), _sector()]),
        operators=OperatorPool([_add(), _sub(), _ts_mean(), _group_rank()]),
    )


def _add_ast() -> ASTNode:
    return ASTNode.call(
        _add(),
        (ASTNode.field_ref(_close()), ASTNode.field_ref(_open())),
    )


def _sub_ast() -> ASTNode:
    return ASTNode.call(
        _sub(),
        (ASTNode.field_ref(_close()), ASTNode.field_ref(_open())),
    )


def _count_nodes(node: ASTNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


# --- Individual -------------------------------------------------------------

def test_individual_from_ast_computes_fingerprint() -> None:
    ast = _add_ast()
    ind = Individual.from_ast(ast)
    assert ind.fingerprint == ast.fingerprint()
    assert ind.ast is ast
    assert ind.fitness is None
    assert not ind.is_evaluated()


def test_individual_equality_by_fingerprint() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_add_ast())
    assert a == b
    assert hash(a) == hash(b)


def test_individual_equality_ignores_fitness() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(
        _add_ast(), fitness=1.0, status=EvaluationStatus.SUCCESS
    )
    assert a == b
    assert hash(a) == hash(b)


def test_individual_equality_ignores_commutative_order() -> None:
    """add(close, open) and add(open, close) share a fingerprint, so the
    Individuals must compare equal."""
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(
        ASTNode.call(
            _add(),
            (ASTNode.field_ref(_open()), ASTNode.field_ref(_close())),
        )
    )
    assert a == b


def test_individual_inequality_for_different_fingerprint() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_sub_ast())
    assert a != b


def test_individual_not_equal_to_other_types() -> None:
    ind = Individual.from_ast(_add_ast())
    assert (ind == "abc") is False
    assert (ind == 42) is False


def test_individual_rejects_empty_fingerprint() -> None:
    with pytest.raises(ValueError):
        Individual(ast=_add_ast(), fingerprint="")


def test_individual_rejects_negative_generation() -> None:
    with pytest.raises(ValueError):
        Individual(ast=_add_ast(), fingerprint="fp", generation=-1)


def test_individual_success_requires_fitness() -> None:
    with pytest.raises(ValueError):
        Individual(
            ast=_add_ast(),
            fingerprint="fp",
            status=EvaluationStatus.SUCCESS,
        )


def test_individual_rejects_bool_metric() -> None:
    with pytest.raises(ValueError):
        Individual(
            ast=_add_ast(),
            fingerprint="fp",
            metrics=(("ok", True),),  # type: ignore[arg-type]
        )


def test_individual_rejects_empty_metric_name() -> None:
    with pytest.raises(ValueError):
        Individual(ast=_add_ast(), fingerprint="fp", metrics=(("", 1.0),))


def test_individual_with_evaluation_produces_new_instance() -> None:
    a = Individual.from_ast(_add_ast())
    b = a.with_evaluation(fitness=1.5, metrics=(("sharpe", 2.0),))
    assert b is not a
    assert b.fingerprint == a.fingerprint
    assert b.ast is a.ast
    assert b.fitness == 1.5
    assert b.metrics == (("sharpe", 2.0),)
    assert b.status is EvaluationStatus.SUCCESS
    assert a.is_evaluated() is False
    assert b.is_evaluated() is True


def test_individual_is_immutable() -> None:
    a = Individual.from_ast(_add_ast())
    with pytest.raises(AttributeError):
        a.fitness = 1.0  # type: ignore[misc]


# --- Population -------------------------------------------------------------

def test_population_empty() -> None:
    p = Population.empty()
    assert len(p) == 0
    assert p.all() == ()
    assert list(p) == []


def test_population_basic() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_sub_ast())
    p = Population([a, b])
    assert len(p) == 2
    assert a.fingerprint in p
    assert b.fingerprint in p
    assert p.by_fingerprint(a.fingerprint) is a
    assert p.contains(a.fingerprint)


def test_population_rejects_duplicate_fingerprint() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_add_ast())  # same fingerprint
    with pytest.raises(DuplicateIndividualError):
        Population([a, b])


def test_population_commutative_add_dedupes() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(
        ASTNode.call(
            _add(),
            (ASTNode.field_ref(_open()), ASTNode.field_ref(_close())),
        )
    )
    with pytest.raises(DuplicateIndividualError):
        Population([a, b])


def test_population_iteration_sorted_by_fingerprint() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_sub_ast())
    p = Population([a, b])
    fps = [i.fingerprint for i in p]
    assert fps == sorted(fps)


def test_population_by_fingerprint_unknown_returns_none() -> None:
    p = Population.empty()
    assert p.by_fingerprint("nope") is None
    assert "nope" not in p
    assert 42 not in p


def test_population_contains_individual() -> None:
    a = Individual.from_ast(_add_ast())
    p = Population([a])
    assert a in p
    b = Individual.from_ast(_sub_ast())
    assert b not in p


def test_population_equality_is_order_independent() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_sub_ast())
    p1 = Population([a, b])
    p2 = Population([b, a])
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_population_not_equal_to_other_types() -> None:
    p = Population.empty()
    assert (p == "abc") is False
    assert (p == 42) is False


def test_population_concat_dedupes() -> None:
    a = Individual.from_ast(_add_ast())
    b = Individual.from_ast(_sub_ast())
    p1 = Population([a])
    p2 = Population([a, b])  # a duplicated across p1 and p2
    merged = Population.concat(p1, p2)
    assert len(merged) == 2


def test_population_concat_empty() -> None:
    p = Population.concat(Population.empty(), Population.empty())
    assert len(p) == 0


def test_population_evaluated() -> None:
    a = Individual.from_ast(
        _add_ast(), fitness=1.0, status=EvaluationStatus.SUCCESS
    )
    b = Individual.from_ast(_sub_ast())  # unevaluated
    p = Population([a, b])
    ev = p.evaluated()
    assert len(ev) == 1
    assert ev[0] == a


def test_population_with_fitness_at_least() -> None:
    a = Individual.from_ast(
        _add_ast(), fitness=1.0, status=EvaluationStatus.SUCCESS
    )
    b = Individual.from_ast(
        _sub_ast(), fitness=5.0, status=EvaluationStatus.SUCCESS
    )
    p = Population([a, b])
    assert len(p.with_fitness_at_least(2.0)) == 1
    assert len(p.with_fitness_at_least(0.0)) == 2
    assert len(p.with_fitness_at_least(10.0)) == 0


# --- RandomAstBuilder -------------------------------------------------------

def test_builder_produces_matrix_ast() -> None:
    b = RandomAstBuilder(_pool(), max_depth=3, max_nodes=10)
    ast = b.build(DataType.MATRIX, random.Random(42))
    assert ast.data_type is DataType.MATRIX


def test_builder_reproducible_with_same_seed() -> None:
    b1 = RandomAstBuilder(_pool(), max_depth=3, max_nodes=10)
    b2 = RandomAstBuilder(_pool(), max_depth=3, max_nodes=10)
    a1 = b1.build(DataType.MATRIX, random.Random(12345))
    a2 = b2.build(DataType.MATRIX, random.Random(12345))
    assert a1.fingerprint() == a2.fingerprint()


def test_builder_different_seed_different_ast() -> None:
    b = RandomAstBuilder(_pool(), max_depth=4, max_nodes=20)
    a1 = b.build(DataType.MATRIX, random.Random(1))
    a2 = b.build(DataType.MATRIX, random.Random(999))
    assert a1.fingerprint() != a2.fingerprint()


def test_builder_empty_pool_raises() -> None:
    pool = Pool(fields=FieldPool([]), operators=OperatorPool([]))
    b = RandomAstBuilder(pool)
    with pytest.raises(GenerationError):
        b.build(DataType.MATRIX, random.Random(0))


def test_builder_no_matrix_field_or_op_raises() -> None:
    # Only a GROUP field; no operator outputs MATRIX. Nothing to build.
    pool = Pool(
        fields=FieldPool([_sector()]),
        operators=OperatorPool([]),
    )
    b = RandomAstBuilder(pool)
    with pytest.raises(GenerationError):
        b.build(DataType.MATRIX, random.Random(0))


def test_builder_max_depth_zero_yields_leaf() -> None:
    b = RandomAstBuilder(_pool(), max_depth=0, max_nodes=10)
    for seed in range(20):
        ast = b.build(DataType.MATRIX, random.Random(seed))
        assert ast.kind is NodeKind.FIELD_REF


def test_builder_max_nodes_respected() -> None:
    b = RandomAstBuilder(_pool(), max_depth=5, max_nodes=3)
    for seed in range(30):
        ast = b.build(DataType.MATRIX, random.Random(seed))
        assert _count_nodes(ast) <= 3


def test_builder_scalar_leaf_is_literal_when_no_scalar_fields() -> None:
    b = RandomAstBuilder(_pool(), max_depth=0, max_nodes=1)
    for seed in range(20):
        ast = b.build(DataType.SCALAR, random.Random(seed))
        assert ast.kind is NodeKind.LITERAL


def test_builder_generated_asts_are_valid() -> None:
    """ASTNode.__post_init__ validates type compatibility. If the builder
    returns an AST, that AST has already passed validation."""
    b = RandomAstBuilder(_pool(), max_depth=4, max_nodes=15)
    seen_calls = False
    for seed in range(50):
        ast = b.build(DataType.MATRIX, random.Random(seed))
        assert ast.fingerprint()
        if ast.kind is NodeKind.CALL:
            seen_calls = True
    assert seen_calls, "builder never produced a CALL node in 50 seeds"


def test_builder_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        RandomAstBuilder(_pool(), max_depth=-1)
    with pytest.raises(ValueError):
        RandomAstBuilder(_pool(), max_nodes=0)
    with pytest.raises(ValueError):
        RandomAstBuilder(_pool(), max_operator_attempts=0)