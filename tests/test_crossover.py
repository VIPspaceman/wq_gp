"""Tests for subtree crossover. Phase 3 Step 3 gate."""
from __future__ import annotations

import random

import pytest

from wqgp.ast import ASTNode, NodeKind, tree_depth, tree_size
from wqgp.crossover import (
    Crossover,
    CrossoverError,
    subtree_crossover,
)
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---------------------------------------------------------------

def _matrix_field(fid: str) -> FieldDef:
    return FieldDef(fid, DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _group_field(fid: str) -> FieldDef:
    return FieldDef(fid, DataType.GROUP, SemanticKind.CATEGORICAL, "pv1")


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
        commutative=True,
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
        "group_rank", (DataType.MATRIX, DataType.GROUP), DataType.MATRIX,
    )


def _pool() -> Pool:
    return Pool(
        fields=FieldPool(
            [
                _matrix_field("close"),
                _matrix_field("open"),
                _matrix_field("high"),
                _matrix_field("low"),
                _group_field("sector"),
            ]
        ),
        operators=OperatorPool(
            [_add(), _sub(), _mul(), _ts_mean(), _group_rank()]
        ),
    )


def _f(fid: str) -> ASTNode:
    return ASTNode.field_ref(_matrix_field(fid))


def _add_close_open() -> ASTNode:
    return ASTNode.call(_add(), (_f("close"), _f("open")))


def _sub_high_low() -> ASTNode:
    return ASTNode.call(_sub(), (_f("high"), _f("low")))


def _ts_mean_close() -> ASTNode:
    return ASTNode.call(_ts_mean(), (_f("close"),), params=(5,))


# --- basic crossover --------------------------------------------------------

def test_crossover_returns_distinct_child() -> None:
    a = _add_close_open()
    b = _sub_high_low()
    c = Crossover(_pool())
    child = c.crossover(a, b, random.Random(0))
    assert child.fingerprint() != a.fingerprint()
    assert child.fingerprint() != b.fingerprint()


def test_crossover_child_has_root_operator_of_a() -> None:
    """Root is never swapped (D-055), so the child's root op matches A's."""
    a = _add_close_open()
    b = _sub_high_low()
    c = Crossover(_pool())
    for seed in range(20):
        child = c.crossover(a, b, random.Random(seed))
        assert child.op_def is not None
        assert child.op_def.name == "add"


def test_crossover_commutative_normalization_applies() -> None:
    """If the swapped-in subtree is just a reordering under commutative
    normalization, the dispatcher must reject it and try again."""
    a = ASTNode.call(_add(), (_f("close"), _f("open")))
    b = ASTNode.call(_sub(), (_f("open"), _f("close")))
    c = Crossover(_pool())
    child = c.crossover(a, b, random.Random(0))
    # add(close, open) and add(open, close) share fingerprint; dispatcher
    # must not return either of those.
    assert child.fingerprint() != a.fingerprint()


def test_crossover_reproducible() -> None:
    a = _add_close_open()
    b = _sub_high_low()
    c1 = Crossover(_pool())
    c2 = Crossover(_pool())
    r1 = c1.crossover(a, b, random.Random(12345))
    r2 = c2.crossover(a, b, random.Random(12345))
    assert r1.fingerprint() == r2.fingerprint()


def test_crossover_identical_parents_rejected() -> None:
    a = _add_close_open()
    b = _add_close_open()  # same fingerprint
    c = Crossover(_pool())
    with pytest.raises(CrossoverError):
        c.crossover(a, b, random.Random(0))


# --- type compatibility (D-052) ---------------------------------------------

def test_crossover_only_swaps_same_type() -> None:
    """A = group_rank(close, sector); B = group_rank(open, industry).

    Both parents have a MATRIX arg and a GROUP arg at the same
    positions. A valid crossover must mix the two parents: either
    group_rank(open, sector) or group_rank(close, industry). A child
    with a MATRIX where a GROUP is expected would fail AST validation,
    so only same-type swaps are candidates.

    Two positions differ between A and B, which is the minimum needed
    for at least one same-type swap to produce a child distinct from
    both parents. See D-057.
    """
    a = ASTNode.call(
        _group_rank(),
        (_f("close"), ASTNode.field_ref(_group_field("sector"))),
    )
    b = ASTNode.call(
        _group_rank(),
        (_f("open"), ASTNode.field_ref(_group_field("industry"))),
    )
    pool = Pool(
        fields=FieldPool(
            [
                _matrix_field("close"),
                _matrix_field("open"),
                _matrix_field("high"),
                _matrix_field("low"),
                _group_field("sector"),
                _group_field("industry"),
            ]
        ),
        operators=OperatorPool(
            [_add(), _sub(), _mul(), _ts_mean(), _group_rank()]
        ),
    )
    c = Crossover(pool)
    for seed in range(20):
        child = c.crossover(a, b, random.Random(seed))
        assert child.op_def is not None
        assert child.op_def.name == "group_rank"
        assert child.fingerprint() != a.fingerprint()
        assert child.fingerprint() != b.fingerprint()
        # The child must mix: its arg0 and arg1 come from different parents.
        arg0 = child.children[0].field_def.field_id  # type: ignore[union-attr]
        arg1 = child.children[1].field_def.field_id  # type: ignore[union-attr]
        assert (arg0, arg1) in {
            ("open", "sector"),
            ("close", "industry"),
        }


def test_subtree_crossover_returns_none_when_no_type_match() -> None:
    """No compatible non-root pair exists: A's non-root is MATRIX,
    B's only node is its GROUP root, which is forbidden (D-055)."""
    a = ASTNode.call(_ts_mean(), (_f("close"),), params=(5,))
    b = ASTNode.field_ref(_group_field("sector"))
    # b has no non-root nodes at all -> no pair.
    assert subtree_crossover(a, b, random.Random(0)) is None


def test_subtree_crossover_none_for_leaf_parents() -> None:
    a = _f("close")
    b = _f("open")
    # Both parents are single nodes, so a_nodes and b_nodes are empty
    # after filtering out roots. No swap is possible.
    assert subtree_crossover(a, b, random.Random(0)) is None


# --- depth / size limits (D-054) --------------------------------------------

def test_crossover_respects_max_depth() -> None:
    """Every child has depth >= A's depth: a leaf-for-leaf swap preserves
    depth, and grafted subtrees only increase it. So a max_depth smaller
    than A's own depth rejects every candidate. See D-056."""
    a = _add_close_open()          # depth 2, size 3
    b = _sub_high_low()            # depth 2, size 3
    # max_depth=1 < depth(A)=2: every candidate is rejected.
    c = Crossover(_pool(), max_depth=1, max_nodes=100, max_attempts=5)
    with pytest.raises(CrossoverError):
        c.crossover(a, b, random.Random(0))


def test_crossover_respects_max_nodes() -> None:
    """Every child has size >= A's size for the same reason. A max_nodes
    smaller than A's own size rejects every candidate. See D-056."""
    a = _add_close_open()          # size 3
    b = _sub_high_low()            # size 3
    # max_nodes=2 < size(A)=3: every candidate is rejected.
    c = Crossover(_pool(), max_depth=100, max_nodes=2, max_attempts=5)
    with pytest.raises(CrossoverError):
        c.crossover(a, b, random.Random(0))


def test_crossover_output_within_limits_when_feasible() -> None:
    a = _add_close_open()
    b = _sub_high_low()
    c = Crossover(_pool(), max_depth=3, max_nodes=10)
    for seed in range(20):
        child = c.crossover(a, b, random.Random(seed))
        assert tree_depth(child) <= 3
        assert tree_size(child) <= 10


# --- root preservation (D-055) ----------------------------------------------

def test_crossover_never_swaps_root() -> None:
    """Even when B's root is a MATRIX-typed node and A's non-root slots
    are MATRIX-typed, the child's root must stay A's root operator."""
    a = _add_close_open()
    b = _sub_high_low()
    for seed in range(20):
        c = Crossover(_pool())
        child = c.crossover(a, b, random.Random(seed))
        assert child.op_def is not None
        assert child.op_def.name == "add"


# --- dispatcher lifecycle ---------------------------------------------------

def test_crossover_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        Crossover(_pool(), max_attempts=0)
    with pytest.raises(ValueError):
        Crossover(_pool(), max_depth=0)
    with pytest.raises(ValueError):
        Crossover(_pool(), max_nodes=0)


def test_subtree_crossover_rejects_bad_limits() -> None:
    a = _add_close_open()
    b = _sub_high_low()
    with pytest.raises(ValueError):
        subtree_crossover(a, b, random.Random(0), max_depth=0)
    with pytest.raises(ValueError):
        subtree_crossover(a, b, random.Random(0), max_nodes=0)


# --- deep tree integration --------------------------------------------------

def test_crossover_deep_trees_produces_valid_child() -> None:
    inner_a = ASTNode.call(_ts_mean(), (_f("close"),), params=(10,))
    a = ASTNode.call(_add(), (inner_a, _f("open")))

    inner_b = ASTNode.call(_ts_mean(), (_f("high"),), params=(20,))
    b = ASTNode.call(_sub(), (inner_b, _f("low")))

    c = Crossover(_pool(), max_depth=6, max_nodes=20)
    for seed in range(20):
        child = c.crossover(a, b, random.Random(seed))
        # ASTNode.__post_init__ would have raised if the child were
        # type-invalid; reaching here means the tree is well-formed.
        assert child.data_type is DataType.MATRIX
        assert child.fingerprint() != a.fingerprint()
        assert child.fingerprint() != b.fingerprint()