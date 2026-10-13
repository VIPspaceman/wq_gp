"""Tests for OperatorDef. Phase 1 gate: all must pass before moving on."""
from __future__ import annotations

import pytest

from wqgp.operator import OperatorDef
from wqgp.types import DataType, ParamDomain


# --- helpers: representative operators -------------------------------------

def _add() -> OperatorDef:
    return OperatorDef(
        name="add",
        input_types=(DataType.MATRIX, DataType.MATRIX),
        output_type=DataType.MATRIX,
        commutative=True,
        associative=True,
        symmetric=True,
    )


def _sub() -> OperatorDef:
    return OperatorDef(
        name="sub",
        input_types=(DataType.MATRIX, DataType.MATRIX),
        output_type=DataType.MATRIX,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        name="ts_mean",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512, default=5)),),
    )


def _group_rank() -> OperatorDef:
    return OperatorDef(
        name="group_rank",
        input_types=(DataType.MATRIX, DataType.GROUP),
        output_type=DataType.MATRIX,
    )


# --- happy paths ---

def test_add_happy_path() -> None:
    op = _add()
    assert op.name == "add"
    assert op.arity == 2
    assert op.output_type is DataType.MATRIX
    assert op.commutative and op.associative and op.symmetric
    assert op.param_domains == ()
    assert op.total_args == 2


def test_sub_happy_path() -> None:
    op = _sub()
    assert op.arity == 2
    assert not op.commutative
    assert not op.associative
    assert not op.symmetric


def test_ts_mean_with_param() -> None:
    op = _ts_mean()
    assert op.arity == 1
    assert op.param_names == ("window",)
    assert op.total_args == 2
    assert op.param_domain("window").contains(5)
    assert not op.param_domain("window").contains(0)


def test_group_rank_with_group_arg() -> None:
    op = _group_rank()
    assert op.arity == 2
    assert op.accepts((DataType.MATRIX, DataType.GROUP))
    assert not op.accepts((DataType.MATRIX, DataType.MATRIX))


# --- input_types normalization: single, union ---

def test_single_type_normalized_to_frozenset() -> None:
    op = _sub()
    for pos in op.input_types:
        assert isinstance(pos, frozenset)
        assert pos == frozenset({DataType.MATRIX})


def test_union_input_types() -> None:
    op = OperatorDef(
        name="rank",
        input_types=((DataType.MATRIX, DataType.VECTOR),),
        output_type=DataType.MATRIX,
    )
    assert op.input_types == (frozenset({DataType.MATRIX, DataType.VECTOR}),)
    assert op.accepts((DataType.MATRIX,))
    assert op.accepts((DataType.VECTOR,))
    assert not op.accepts((DataType.GROUP,))


def test_mixed_single_and_union_inputs() -> None:
    op = OperatorDef(
        name="vec_ops",
        input_types=(
            (DataType.MATRIX, DataType.VECTOR),
            DataType.SCALAR,
        ),
        output_type=DataType.VECTOR,
    )
    assert op.accepts((DataType.VECTOR, DataType.SCALAR))
    assert op.accepts((DataType.MATRIX, DataType.SCALAR))
    assert not op.accepts((DataType.GROUP, DataType.SCALAR))


# --- symmetric n-ary ---

def test_symmetric_nary() -> None:
    op = OperatorDef(
        name="min3",
        input_types=(
            DataType.MATRIX, DataType.MATRIX, DataType.MATRIX,
        ),
        output_type=DataType.MATRIX,
        commutative=True,
        symmetric=True,
    )
    assert op.arity == 3
    assert op.symmetric


# --- D-008: associative does NOT imply commutative ---

def test_associative_not_commutative_is_legal() -> None:
    """Matmul / string concat: associative, not commutative.
    This must NOT raise. See DECISIONS.md D-008."""
    op = OperatorDef(
        name="matmul",
        input_types=(DataType.MATRIX, DataType.MATRIX),
        output_type=DataType.MATRIX,
        associative=True,
        commutative=False,
    )
    assert op.associative
    assert not op.commutative


# --- name invariants ---

def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef("", (DataType.MATRIX,), DataType.MATRIX)


def test_whitespace_only_name_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef("  ", (DataType.MATRIX,), DataType.MATRIX)


def test_name_with_surrounding_whitespace_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef(" add ", (DataType.MATRIX,), DataType.MATRIX)


# --- input_types invariants ---

def test_empty_input_types_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef("noargs", (), DataType.MATRIX)


def test_empty_union_position_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef("bad", ((),), DataType.MATRIX)


def test_invalid_input_type_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef("bad", ("MATRIX",), DataType.MATRIX)  # type: ignore[arg-type]


# --- flag invariants ---

@pytest.mark.parametrize(
    "flag",
    ["commutative", "associative", "symmetric"],
)
def test_flags_require_arity_at_least_2(flag: str) -> None:
    kwargs = {flag: True}
    with pytest.raises(ValueError):
        OperatorDef(
            "unary",
            (DataType.MATRIX,),
            DataType.MATRIX,
            **kwargs,
        )


# --- param_domains: dict vs tuple input ---

def test_param_domains_accepts_dict() -> None:
    op = OperatorDef(
        name="ts_mean",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains={"window": ParamDomain.int_range(1, 512)},
    )
    assert op.param_names == ("window",)


def test_param_domains_accepts_tuple_of_pairs() -> None:
    op = _ts_mean()
    assert isinstance(op.param_domains, tuple)
    assert op.param_domains[0][0] == "window"


def test_dict_param_domains_preserve_insertion_order() -> None:
    op = OperatorDef(
        name="foo",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains={
            "window": ParamDomain.int_range(1, 512),
            "k": ParamDomain.float_range(0.0, 1.0),
        },
    )
    assert op.param_names == ("window", "k")


# --- param_domains invariants ---

def test_empty_param_name_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef(
            name="foo",
            input_types=(DataType.MATRIX,),
            output_type=DataType.MATRIX,
            param_domains=(("", ParamDomain.int_range(1, 5)),),
        )


def test_duplicate_param_name_rejected() -> None:
    with pytest.raises(ValueError):
        OperatorDef(
            name="foo",
            input_types=(DataType.MATRIX,),
            output_type=DataType.MATRIX,
            param_domains=(
                ("window", ParamDomain.int_range(1, 5)),
                ("window", ParamDomain.int_range(1, 10)),
            ),
        )


def test_param_domain_lookup_missing_raises() -> None:
    op = _ts_mean()
    with pytest.raises(KeyError):
        op.param_domain("nonexistent")


# --- accepts() ---

def test_accepts_false_on_wrong_arity() -> None:
    op = _sub()
    assert not op.accepts((DataType.MATRIX,))
    assert not op.accepts((DataType.MATRIX, DataType.MATRIX, DataType.MATRIX))


def test_accepts_false_on_wrong_type() -> None:
    op = _group_rank()
    assert not op.accepts((DataType.GROUP, DataType.GROUP))


# --- frozen + hashable + equality ---

def test_operator_is_frozen_and_hashable() -> None:
    op = _sub()
    with pytest.raises(Exception):
        op.name = "mul"  # type: ignore[misc]
    assert hash(op) == hash(_sub())
    assert op == _sub()