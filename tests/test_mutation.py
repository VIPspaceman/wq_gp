"""Tests for mutation operators. Phase 3 Step 2 gate."""
from __future__ import annotations

import random

import pytest

from wqgp.ast import ASTNode, NodeKind
from wqgp.field import FieldDef
from wqgp.mutation import (
    DEFAULT_STRATEGIES,
    MutationError,
    Mutator,
    mutate_field,
    mutate_operator,
    mutate_subtree,
    mutate_window,
)
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
    # Same parameter name as ts_mean -> swappable under D-048.
    return OperatorDef(
        "ts_std",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _ts_decay() -> OperatorDef:
    # Different parameter name -> NOT swappable with ts_mean under D-048.
    return OperatorDef(
        "ts_decay",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("halflife", ParamDomain.int_range(1, 512)),),
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
                _matrix_field("volume"),
                _group_field("sector"),
            ]
        ),
        operators=OperatorPool(
            [
                _add(), _sub(), _mul(),
                _ts_mean(), _ts_std(), _ts_decay(),
                _group_rank(),
            ]
        ),
    )


def _minimal_pool() -> Pool:
    """The smallest pool under which no mutation strategy can succeed
    for a single field-ref AST.

    - No operators: blocks window / operator / subtree-growth paths.
    - One field only: blocks field swap.
    - No operators also means the subtree builder can only reproduce
      the same leaf, which the dispatcher rejects (D-047).
    """
    return Pool(
        fields=FieldPool([_matrix_field("close")]),
        operators=OperatorPool([]),
    )


def _close() -> ASTNode:
    return ASTNode.field_ref(_matrix_field("close"))


def _open() -> ASTNode:
    return ASTNode.field_ref(_matrix_field("open"))


def _add_close_open() -> ASTNode:
    return ASTNode.call(_add(), (_close(), _open()))


def _ts_mean_close_5() -> ASTNode:
    return ASTNode.call(_ts_mean(), (_close(),), params=(5,))


# --- iter helpers (indirectly via strategies) ------------------------------

def test_iter_nodes_visible_through_strategy() -> None:
    """Sanity: the AST has at least the nodes the strategies expect."""
    ast = _add_close_open()
    assert ast.kind is NodeKind.CALL
    assert len(ast.children) == 2


# --- mutate_window ---------------------------------------------------------

def test_window_mutation_changes_param() -> None:
    ast = _ts_mean_close_5()
    result = mutate_window(ast, _pool(), random.Random(0))
    assert result is not None
    assert result.params != ast.params
    assert result.params[0] != 5


def test_window_mutation_returns_none_on_no_windowed_call() -> None:
    ast = _add_close_open()
    assert mutate_window(ast, _pool(), random.Random(0)) is None


def test_window_mutation_preserves_children() -> None:
    ast = _ts_mean_close_5()
    result = mutate_window(ast, _pool(), random.Random(42))
    assert result is not None
    assert result.children[0] == ast.children[0]


def test_window_mutation_preserves_type() -> None:
    ast = _ts_mean_close_5()
    result = mutate_window(ast, _pool(), random.Random(1))
    assert result is not None
    assert result.data_type is DataType.MATRIX


# --- mutate_field ----------------------------------------------------------

def test_field_mutation_changes_field() -> None:
    ast = _close()
    result = mutate_field(ast, _pool(), random.Random(0))
    assert result is not None
    assert result.kind is NodeKind.FIELD_REF
    assert result.field_def is not None
    assert result.field_def.field_id != "close"


def test_field_mutation_returns_none_on_unique_field() -> None:
    ast = _close()
    assert mutate_field(ast, _minimal_pool(), random.Random(0)) is None


def test_field_mutation_preserves_type() -> None:
    ast = _close()
    for seed in range(20):
        result = mutate_field(ast, _pool(), random.Random(seed))
        assert result is not None
        assert result.data_type is DataType.MATRIX


def test_field_mutation_on_group_field_stays_group() -> None:
    ast = ASTNode.field_ref(_group_field("sector"))
    # No other GROUP field -> returns None.
    assert mutate_field(ast, _pool(), random.Random(0)) is None


# --- mutate_operator -------------------------------------------------------

def test_operator_mutation_swaps_operator() -> None:
    ast = _add_close_open()
    for seed in range(20):
        result = mutate_operator(ast, _pool(), random.Random(seed))
        if result is not None:
            assert result.op_def is not None
            assert result.op_def.name != "add"
            return
    pytest.fail("mutate_operator never found an alternative in 20 seeds")


def test_operator_mutation_keeps_children() -> None:
    ast = _add_close_open()
    result = mutate_operator(ast, _pool(), random.Random(0))
    assert result is not None
    assert result.children == ast.children


def test_operator_mutation_respects_param_name_match() -> None:
    """ts_mean -> ts_std is legal (same param name). ts_mean -> ts_decay
    is NOT legal (different param name). See D-048."""
    ast = _ts_mean_close_5()
    for seed in range(30):
        result = mutate_operator(ast, _pool(), random.Random(seed))
        if result is not None:
            assert result.op_def is not None
            assert result.op_def.name in {"ts_std"}, (
                f"unexpected operator {result.op_def.name!r}"
            )
            return
    pytest.fail("mutate_operator never produced ts_std in 30 seeds")


def test_operator_mutation_none_when_no_alternative() -> None:
    ast = ASTNode.call(_sub(), (_close(), _open()))
    # pool has only sub in the minimal pool; nothing to swap to.
    assert mutate_operator(ast, _minimal_pool(), random.Random(0)) is None


def test_operator_mutation_rejects_bad_input_types() -> None:
    """group_rank takes (MATRIX, GROUP); adding a MATRIX-only sibling is
    not a candidate, so with only MATRIX children nothing matches."""
    ast = ASTNode.call(
        _group_rank(),
        (_close(), ASTNode.field_ref(_group_field("sector"))),
    )
    # pool has no other (MATRIX, GROUP) operator -> None.
    assert mutate_operator(ast, _pool(), random.Random(0)) is None


# --- mutate_subtree --------------------------------------------------------

def test_subtree_mutation_changes_fingerprint() -> None:
    ast = _add_close_open()
    result = mutate_subtree(ast, _pool(), random.Random(0))
    assert result is not None
    assert result.fingerprint() != ast.fingerprint()


def test_subtree_mutation_preserves_root_type() -> None:
    ast = _ts_mean_close_5()
    for seed in range(20):
        result = mutate_subtree(ast, _pool(), random.Random(seed))
        if result is not None:
            assert result.data_type is DataType.MATRIX


def test_subtree_mutation_returns_none_on_empty_pool() -> None:
    empty = Pool(fields=FieldPool([]), operators=OperatorPool([]))
    ast = _add_close_open()
    assert mutate_subtree(ast, empty, random.Random(0)) is None


# --- Mutator dispatcher ----------------------------------------------------

def test_mutator_produces_distinct_fingerprint() -> None:
    m = Mutator(_pool())
    ast = _add_close_open()
    result = m.mutate(ast, random.Random(0))
    assert result.fingerprint() != ast.fingerprint()


def test_mutator_reproducible() -> None:
    m1 = Mutator(_pool())
    m2 = Mutator(_pool())
    ast = _add_close_open()
    r1 = m1.mutate(ast, random.Random(12345))
    r2 = m2.mutate(ast, random.Random(12345))
    assert r1.fingerprint() == r2.fingerprint()


def test_mutator_raises_when_impossible() -> None:
    m = Mutator(_minimal_pool())
    ast = _close()  # single field-ref; nothing to swap, shrink, or grow
    with pytest.raises(MutationError):
        m.mutate(ast, random.Random(0))


def test_mutator_rejects_empty_strategy_list() -> None:
    with pytest.raises(ValueError):
        Mutator(_pool(), strategies=())


def test_mutator_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError):
        Mutator(_pool(), max_attempts=0)


def test_mutator_single_strategy() -> None:
    m = Mutator(_pool(), strategies=(mutate_field,))
    ast = _close()
    result = m.mutate(ast, random.Random(0))
    assert result.kind is NodeKind.FIELD_REF
    assert result.fingerprint() != ast.fingerprint()


def test_mutator_single_strategy_raises_when_no_candidates() -> None:
    # Only window strategy, but AST has no windowed call.
    m = Mutator(
        _pool(), strategies=(mutate_window,), max_attempts=3
    )
    ast = _add_close_open()
    with pytest.raises(MutationError):
        m.mutate(ast, random.Random(0))


def test_default_strategies_cover_all_four() -> None:
    assert mutate_window in DEFAULT_STRATEGIES
    assert mutate_field in DEFAULT_STRATEGIES
    assert mutate_operator in DEFAULT_STRATEGIES
    assert mutate_subtree in DEFAULT_STRATEGIES


# --- integration: mutation + fingerprint normalization ---------------------

def test_mutating_commutative_children_does_not_create_duplicate() -> None:
    """add(close, open) and add(open, close) share a fingerprint, so a
    field mutation that just swaps them must not be reported as a
    distinct mutation."""
    ast1 = ASTNode.call(_add(), (_close(), _open()))
    ast2 = ASTNode.call(_add(), (_open(), _close()))
    assert ast1.fingerprint() == ast2.fingerprint()
    # Mutator on ast1 must return an AST with a *different* fingerprint
    # than the shared original fingerprint.
    m = Mutator(_pool())
    result = m.mutate(ast1, random.Random(0))
    assert result.fingerprint() != ast1.fingerprint()


def test_mutation_of_deep_tree_preserves_types() -> None:
    """Build a 3-level tree, mutate, verify the result is still valid
    (ASTNode.__post_init__ would raise if not)."""
    inner = ASTNode.call(_ts_mean(), (_close(),), params=(10,))
    mid = ASTNode.call(_add(), (inner, _open()))
    root = ASTNode.call(_ts_mean(), (mid,), params=(5,))

    m = Mutator(_pool())
    for seed in range(30):
        result = m.mutate(root, random.Random(seed))
        assert result.data_type is DataType.MATRIX
        assert result.fingerprint() != root.fingerprint()