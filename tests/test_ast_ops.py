"""Tests for AST simplification. D-164 gate."""
from __future__ import annotations

import pytest

from wqgp.ast import ASTNode, NodeKind
from wqgp.ast_ops import (
    collect_fields,
    count_distinct_fields,
    simplify_ast,
)
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.types import DataType, SemanticKind


def _field(fid: str) -> FieldDef:
    return FieldDef(fid, DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _binary(name: str, *, commutative: bool = False) -> OperatorDef:
    return OperatorDef(
        name,
        (DataType.MATRIX, DataType.MATRIX),
        DataType.MATRIX,
        commutative=commutative,
    )


A = ASTNode.field_ref(_field("a"))
B = ASTNode.field_ref(_field("b"))
C = ASTNode.field_ref(_field("c"))
ZERO = ASTNode.literal(0)
ONE = ASTNode.literal(1)
TWO = ASTNode.literal(2)

ADD = _binary("add", commutative=True)
SUBTRACT = _binary("subtract")
MULTIPLY = _binary("multiply", commutative=True)
DIVIDE = _binary("divide")
MAX = _binary("max", commutative=True)
MIN = _binary("min", commutative=True)


# --- leaves are returned unchanged -----------------------------------------

def test_leaf_field_unchanged() -> None:
    assert simplify_ast(A).fingerprint() == A.fingerprint()


def test_leaf_literal_unchanged() -> None:
    assert simplify_ast(ZERO).fingerprint() == ZERO.fingerprint()


# --- max / min -------------------------------------------------------------

def test_max_same_field_collapses() -> None:
    node = ASTNode.call(MAX, (A, A))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_min_same_field_collapses() -> None:
    node = ASTNode.call(MIN, (A, A))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_max_different_fields_unchanged() -> None:
    node = ASTNode.call(MAX, (A, B))
    assert simplify_ast(node).fingerprint() == node.fingerprint()


def test_min_different_fields_unchanged() -> None:
    node = ASTNode.call(MIN, (A, B))
    assert simplify_ast(node).fingerprint() == node.fingerprint()


def test_max_uses_commutative_normalization() -> None:
    """max(add(a, b), add(b, a)) is max(x, x) semantically."""
    inner_ab = ASTNode.call(ADD, (A, B))
    inner_ba = ASTNode.call(ADD, (B, A))
    node = ASTNode.call(MAX, (inner_ab, inner_ba))
    result = simplify_ast(node)
    assert result.fingerprint() == inner_ab.fingerprint()


# --- add / subtract zero ---------------------------------------------------

def test_add_zero_left_collapses() -> None:
    node = ASTNode.call(ADD, (ZERO, A))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_add_zero_right_collapses() -> None:
    node = ASTNode.call(ADD, (A, ZERO))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_add_non_zero_unchanged() -> None:
    node = ASTNode.call(ADD, (ONE, A))
    assert simplify_ast(node).fingerprint() == node.fingerprint()


def test_subtract_zero_collapses() -> None:
    node = ASTNode.call(SUBTRACT, (A, ZERO))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_subtract_non_zero_unchanged() -> None:
    node = ASTNode.call(SUBTRACT, (A, ONE))
    assert simplify_ast(node).fingerprint() == node.fingerprint()


# --- multiply / divide one -------------------------------------------------

def test_multiply_one_left_collapses() -> None:
    node = ASTNode.call(MULTIPLY, (ONE, A))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_multiply_one_right_collapses() -> None:
    node = ASTNode.call(MULTIPLY, (A, ONE))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_multiply_non_one_unchanged() -> None:
    node = ASTNode.call(MULTIPLY, (TWO, A))
    assert simplify_ast(node).fingerprint() == node.fingerprint()


def test_divide_one_collapses() -> None:
    node = ASTNode.call(DIVIDE, (A, ONE))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


# --- bottom-up cascade -----------------------------------------------------

def test_nested_max_collapses() -> None:
    """max(max(a, a), a) -> a in one call."""
    inner = ASTNode.call(MAX, (A, A))
    outer = ASTNode.call(MAX, (inner, A))
    assert simplify_ast(outer).fingerprint() == A.fingerprint()


def test_add_inside_max_collapses() -> None:
    """max(add(a, 0), a) -> a."""
    inner = ASTNode.call(ADD, (A, ZERO))
    node = ASTNode.call(MAX, (inner, A))
    assert simplify_ast(node).fingerprint() == A.fingerprint()


def test_deep_nesting_collapses() -> None:
    """multiply(add(a, 0), 1) -> a."""
    inner = ASTNode.call(ADD, (A, ZERO))
    outer = ASTNode.call(MULTIPLY, (inner, ONE))
    assert simplify_ast(outer).fingerprint() == A.fingerprint()


def test_simplification_does_not_introduce_new_nodes() -> None:
    """Node count must be non-increasing under simplification."""
    node = ASTNode.call(MAX, (ASTNode.call(ADD, (A, ZERO)), A))
    result = simplify_ast(node)
    assert result.fingerprint() == A.fingerprint()


# --- collect_fields / count_distinct_fields -------------------------------

def test_collect_single_field() -> None:
    assert collect_fields(A) == frozenset({"a"})


def test_collect_literal_is_empty() -> None:
    assert collect_fields(ZERO) == frozenset()


def test_collect_two_fields() -> None:
    node = ASTNode.call(ADD, (A, B))
    assert collect_fields(node) == frozenset({"a", "b"})


def test_collect_dedup() -> None:
    node = ASTNode.call(ADD, (A, A))
    assert collect_fields(node) == frozenset({"a"})


def test_count_one() -> None:
    assert count_distinct_fields(A) == 1


def test_count_zero() -> None:
    assert count_distinct_fields(ZERO) == 0


def test_count_two() -> None:
    node = ASTNode.call(ADD, (A, B))
    assert count_distinct_fields(node) == 2


def test_count_three_nested() -> None:
    inner = ASTNode.call(ADD, (B, C))
    node = ASTNode.call(ADD, (A, inner))
    assert count_distinct_fields(node) == 3


def test_count_after_redundant_wrapper_is_one() -> None:
    """The motivating case for D-164: an expression that looks
    multi-field but is single-field after simplification."""
    inner = ASTNode.call(MAX, (A, A))
    outer = ASTNode.call(MAX, (inner, A))
    # Raw tree has one field, but it appears three times.
    assert count_distinct_fields(outer) == 1
    # Simplification makes that obvious at the structural level too.
    assert simplify_ast(outer).fingerprint() == A.fingerprint()


def test_original_node_is_not_mutated() -> None:
    """simplify_ast returns a new tree; the input is untouched."""
    inner = ASTNode.call(ADD, (A, ZERO))
    original_fp = inner.fingerprint()
    _ = simplify_ast(inner)
    assert inner.fingerprint() == original_fp