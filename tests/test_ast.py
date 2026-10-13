"""Tests for ASTNode and fingerprint normalization. Phase 1 gate."""
from __future__ import annotations

import pytest

from wqgp.ast import ASTNode, NodeKind
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---

CLOSE = FieldDef(
    field_id="close",
    data_type=DataType.MATRIX,
    semantic_kind=SemanticKind.NUMERIC,
    dataset="pv1",
)

SECTOR = FieldDef(
    field_id="sector",
    data_type=DataType.GROUP,
    semantic_kind=SemanticKind.CATEGORICAL,
    dataset="pv1",
)

ADD = OperatorDef(
    name="add",
    input_types=(DataType.MATRIX, DataType.MATRIX),
    output_type=DataType.MATRIX,
    commutative=True,
    associative=True,
    symmetric=True,
)

SUB = OperatorDef(
    name="sub",
    input_types=(DataType.MATRIX, DataType.MATRIX),
    output_type=DataType.MATRIX,
)

MATMUL = OperatorDef(
    name="matmul",
    input_types=(DataType.MATRIX, DataType.MATRIX),
    output_type=DataType.MATRIX,
    associative=True,   # but NOT commutative
)

TS_MEAN = OperatorDef(
    name="ts_mean",
    input_types=(DataType.MATRIX,),
    output_type=DataType.MATRIX,
    param_domains=(("window", ParamDomain.int_range(1, 512, default=5)),),
)

GROUP_RANK = OperatorDef(
    name="group_rank",
    input_types=(DataType.MATRIX, DataType.GROUP),
    output_type=DataType.MATRIX,
)

MIN3 = OperatorDef(
    name="min3",
    input_types=(DataType.MATRIX, DataType.MATRIX, DataType.MATRIX),
    output_type=DataType.MATRIX,
    commutative=True,
    symmetric=True,
)


def _lit(v): return ASTNode.literal(v)
def _fld(f): return ASTNode.field_ref(f)
def _close(): return _fld(CLOSE)


# --- LITERAL ---

def test_literal_int_float_str() -> None:
    assert _lit(5).data_type is DataType.SCALAR
    assert _lit(0.5).data_type is DataType.SCALAR
    assert _lit("add").data_type is DataType.SCALAR


def test_literal_rejects_bool() -> None:
    with pytest.raises(ValueError):
        ASTNode.literal(True)  # type: ignore[arg-type]


def test_literal_rejects_missing_value() -> None:
    with pytest.raises(ValueError):
        ASTNode(kind=NodeKind.LITERAL)


def test_literal_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ASTNode(kind=NodeKind.LITERAL, value=5, children=(_lit(1),))


# --- FIELD_REF ---

def test_field_ref_data_type() -> None:
    assert _fld(CLOSE).data_type is DataType.MATRIX
    assert _fld(SECTOR).data_type is DataType.GROUP


def test_field_ref_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        ASTNode(kind=NodeKind.FIELD_REF)


def test_field_ref_rejects_extra_fields() -> None:
    with pytest.raises(ValueError):
        ASTNode(kind=NodeKind.FIELD_REF, field_def=CLOSE, value=5)


# --- CALL validation ---

def test_call_happy_path() -> None:
    node = ASTNode.call(ADD, (_close(), _close()))
    assert node.data_type is DataType.MATRIX
    assert node.op_def is ADD


def test_call_wrong_arity_rejected() -> None:
    with pytest.raises(ValueError):
        ASTNode.call(ADD, (_close(),))


def test_call_wrong_type_rejected() -> None:
    with pytest.raises(ValueError):
        # group_rank wants (MATRIX, GROUP); passing (MATRIX, MATRIX)
        ASTNode.call(GROUP_RANK, (_close(), _close()))


def test_call_missing_param_rejected() -> None:
    with pytest.raises(ValueError):
        ASTNode.call(TS_MEAN, (_close(),))


def test_call_param_out_of_domain_rejected() -> None:
    with pytest.raises(ValueError):
        ASTNode.call(TS_MEAN, (_close(),), params=(0,))     # window >= 1
    with pytest.raises(ValueError):
        ASTNode.call(TS_MEAN, (_close(),), params=(9999,))  # window <= 512


def test_call_param_happy_path() -> None:
    node = ASTNode.call(TS_MEAN, (_close(),), params=(5,))
    assert node.params == (5,)


# --- SCALAR broadcast (D-011) ---

def test_scalar_broadcast_into_numeric_position() -> None:
    node = ASTNode.call(ADD, (_close(), _lit(1)))
    assert node.data_type is DataType.MATRIX


def test_scalar_does_not_broadcast_into_group() -> None:
    with pytest.raises(ValueError):
        ASTNode.call(GROUP_RANK, (_close(), _lit(5)))


# --- fingerprint: commutative binary (D-001 anchor) ---

def test_commutative_binary_sorts_children() -> None:
    a, b = _close(), _fld(SECTOR)  # types differ but ADD accepts MATRIX only
    # use two MATRIX fields
    open_ = FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    a = _close()
    b = _fld(open_)
    n1 = ASTNode.call(ADD, (a, b))
    n2 = ASTNode.call(ADD, (b, a))
    assert n1.fingerprint() == n2.fingerprint()
    assert n1.canonical() == n2.canonical()


def test_commutative_does_not_recurse() -> None:
    """Sorting at one level must not destroy structure at lower levels."""
    open_ = FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    high = FieldDef("high", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    # sub is non-commutative; add(close, sub(open, high)) must not equal
    # add(close, sub(high, open))
    inner1 = ASTNode.call(SUB, (_fld(open_), _fld(high)))
    inner2 = ASTNode.call(SUB, (_fld(high), _fld(open_)))
    n1 = ASTNode.call(ADD, (_close(), inner1))
    n2 = ASTNode.call(ADD, (_close(), inner2))
    assert n1.fingerprint() != n2.fingerprint()


# --- fingerprint: non-commutative binary ---

def test_non_commutative_binary_does_not_sort() -> None:
    open_ = FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    a, b = _close(), _fld(open_)
    n1 = ASTNode.call(SUB, (a, b))
    n2 = ASTNode.call(SUB, (b, a))
    assert n1.fingerprint() != n2.fingerprint()


# --- fingerprint: params are never reordered ---

def test_params_not_reordered() -> None:
    n1 = ASTNode.call(TS_MEAN, (_close(),), params=(5,))
    n2 = ASTNode.call(TS_MEAN, (_close(),), params=(10,))
    assert n1.fingerprint() != n2.fingerprint()


def test_params_are_not_part_of_child_sorting() -> None:
    """group_rank(close, sector) must not equal group_rank(sector, close)
    even though it has two children -- because group_rank is NOT
    commutative."""
    with pytest.raises(ValueError):
        ASTNode.call(GROUP_RANK, (_fld(SECTOR), _close()))
    n = ASTNode.call(GROUP_RANK, (_close(), _fld(SECTOR)))
    assert n.fingerprint()  # just sanity


# --- fingerprint: associative without commutative (D-008) ---

def test_associative_but_not_commutative_does_not_sort() -> None:
    open_ = FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    a, b = _close(), _fld(open_)
    n1 = ASTNode.call(MATMUL, (a, b))
    n2 = ASTNode.call(MATMUL, (b, a))
    assert n1.fingerprint() != n2.fingerprint()


# --- fingerprint: symmetric n-ary ---

def test_symmetric_nary_sorts_all_children() -> None:
    open_ = FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    high = FieldDef("high", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    a, b, c = _close(), _fld(open_), _fld(high)
    n1 = ASTNode.call(MIN3, (a, b, c))
    n2 = ASTNode.call(MIN3, (c, a, b))
    n3 = ASTNode.call(MIN3, (b, c, a))
    assert n1.fingerprint() == n2.fingerprint() == n3.fingerprint()


# --- fingerprint: 1.0 vs 1 normalization (D-012) ---

def test_integer_valued_float_normalized_in_fingerprint() -> None:
    """D-012 applies where both forms are legal -- FLOAT_RANGE params.

    On INT_RANGE params, `contains` already rejects 5.0 at construction
    (see test_int_range_rejects_bool_and_float), so the fingerprint layer
    never sees a float there.
    """
    scale = OperatorDef(
        name="scale",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains=(("k", ParamDomain.float_range(0.0, 10.0)),),
    )
    n1 = ASTNode.call(scale, (_close(),), params=(5,))     # int  form
    n2 = ASTNode.call(scale, (_close(),), params=(5.0,))   # float form
    assert n1.fingerprint() == n2.fingerprint()


def test_non_integer_float_kept_distinct() -> None:
    # use a float-range param on a synthetic op
    OP = OperatorDef(
        name="scale",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains=(("k", ParamDomain.float_range(0.0, 1.0)),),
    )
    n1 = ASTNode.call(OP, (_close(),), params=(0.5,))
    n2 = ASTNode.call(OP, (_close(),), params=(0.25,))
    assert n1.fingerprint() != n2.fingerprint()


# --- fingerprint: canonical stability ---

def test_fingerprint_is_deterministic() -> None:
    n = ASTNode.call(ADD, (_close(), _lit(1)))
    assert n.fingerprint() == n.fingerprint()
    assert len(n.fingerprint()) == 64  # sha256 hex


def test_canonical_example() -> None:
    n = ASTNode.call(TS_MEAN, (_close(),), params=(5,))
    assert n.canonical() == "C(ts_mean|F(close)|window=5)"


# --- frozen + hashable + equality ---

def test_ast_node_is_frozen_and_hashable() -> None:
    n = ASTNode.call(ADD, (_close(), _lit(1)))
    with pytest.raises(Exception):
        n.kind = NodeKind.LITERAL  # type: ignore[misc]
    assert hash(n) == hash(ASTNode.call(ADD, (_close(), _lit(1))))
    # structural equality ignores commutativity; use fingerprint for that
    n1 = ASTNode.call(ADD, (_close(), _lit(1)))
    n2 = ASTNode.call(ADD, (_lit(1), _close()))
    assert n1 != n2
    assert n1.fingerprint() == n2.fingerprint()