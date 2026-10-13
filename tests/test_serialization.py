"""Tests for JSON serialization primitives. Phase 2 Step 2 gate."""
from __future__ import annotations

import pytest

from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.serialization import (
    SERIALIZATION_SCHEMA_VERSION,
    SerializationError,
    UnsupportedSchemaError,
    field_from_dict,
    field_to_dict,
    operator_from_dict,
    operator_to_dict,
    param_domain_from_dict,
    param_domain_to_dict,
    validate_schema_version,
)
from wqgp.types import DataType, ParamDomain, ParamKind, SemanticKind


# --- ParamDomain round-trip ---

def test_param_domain_int_roundtrip() -> None:
    d = ParamDomain.int_range(1, 512, default=5)
    assert param_domain_from_dict(param_domain_to_dict(d)) == d


def test_param_domain_float_roundtrip() -> None:
    d = ParamDomain.float_range(0.0, 1.0, default=0.5)
    assert param_domain_from_dict(param_domain_to_dict(d)) == d


def test_param_domain_choice_roundtrip() -> None:
    d = ParamDomain.choice(("add", "sub", "mul"), default="add")
    assert param_domain_from_dict(param_domain_to_dict(d)) == d


def test_param_domain_missing_kind_rejected() -> None:
    with pytest.raises(SerializationError):
        param_domain_from_dict({"min_value": 1, "max_value": 5})


def test_param_domain_unknown_kind_rejected() -> None:
    with pytest.raises(SerializationError):
        param_domain_from_dict({"kind": "BOGUS"})


def test_param_domain_bad_invariants_wrapped() -> None:
    # min > max -> ParamDomain raises ValueError -> wrapped
    with pytest.raises(SerializationError):
        param_domain_from_dict(
            {"kind": "INT_RANGE", "min_value": 10, "max_value": 1}
        )


# --- FieldDef round-trip ---

def test_field_roundtrip_with_unit() -> None:
    f = FieldDef(
        "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1",
        coverage=0.98, delay=1, unit="USD",
    )
    assert field_from_dict(field_to_dict(f)) == f


def test_field_roundtrip_without_unit() -> None:
    f = FieldDef("sector", DataType.GROUP, SemanticKind.CATEGORICAL, "pv1")
    assert field_from_dict(field_to_dict(f)) == f


def test_field_missing_required_key_rejected() -> None:
    with pytest.raises(SerializationError):
        field_from_dict({"field_id": "x", "data_type": "MATRIX"})


def test_field_unknown_data_type_rejected() -> None:
    with pytest.raises(SerializationError):
        field_from_dict({
            "field_id": "x", "data_type": "BOGUS",
            "semantic_kind": "NUMERIC", "dataset": "pv1",
        })


def test_field_invariant_wrapped() -> None:
    # CATEGORICAL + non-GROUP -> FieldDef raises -> wrapped
    with pytest.raises(SerializationError):
        field_from_dict({
            "field_id": "x", "data_type": "MATRIX",
            "semantic_kind": "CATEGORICAL", "dataset": "pv1",
        })


# --- OperatorDef round-trip ---

def _add() -> OperatorDef:
    return OperatorDef(
        name="add",
        input_types=(DataType.MATRIX, DataType.MATRIX),
        output_type=DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        name="ts_mean",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512, default=5)),),
    )


def _with_union() -> OperatorDef:
    return OperatorDef(
        name="rank",
        input_types=((DataType.MATRIX, DataType.VECTOR),),
        output_type=DataType.MATRIX,
    )


def test_operator_simple_roundtrip() -> None:
    assert operator_from_dict(operator_to_dict(_add())) == _add()


def test_operator_with_param_roundtrip() -> None:
    assert operator_from_dict(operator_to_dict(_ts_mean())) == _ts_mean()


def test_operator_with_union_roundtrip() -> None:
    assert operator_from_dict(operator_to_dict(_with_union())) == _with_union()


def test_operator_union_serialized_as_sorted_list() -> None:
    d = operator_to_dict(_with_union())
    assert d["input_types"] == [["MATRIX", "VECTOR"]]


def test_operator_single_type_serialized_as_string() -> None:
    d = operator_to_dict(_add())
    assert d["input_types"] == ["MATRIX", "MATRIX"]


def test_operator_flags_omitted_when_false() -> None:
    op = OperatorDef(
        name="sub",
        input_types=(DataType.MATRIX, DataType.MATRIX),
        output_type=DataType.MATRIX,
    )
    d = operator_to_dict(op)
    assert "commutative" not in d
    assert "associative" not in d
    assert "symmetric" not in d
    assert "deterministic" not in d


def test_operator_non_deterministic_roundtrip() -> None:
    op = OperatorDef(
        name="rand",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        deterministic=False,
    )
    assert operator_from_dict(operator_to_dict(op)).deterministic is False


def test_operator_missing_key_rejected() -> None:
    with pytest.raises(SerializationError):
        operator_from_dict({"name": "add", "input_types": ["MATRIX"]})


def test_operator_empty_union_rejected() -> None:
    with pytest.raises(SerializationError):
        operator_from_dict({
            "name": "x", "input_types": [[]], "output_type": "MATRIX",
        })


def test_operator_invalid_input_type_kind_rejected() -> None:
    with pytest.raises(SerializationError):
        operator_from_dict({
            "name": "x", "input_types": [5], "output_type": "MATRIX",
        })


def test_operator_unknown_output_type_rejected() -> None:
    with pytest.raises(SerializationError):
        operator_from_dict({
            "name": "x", "input_types": ["MATRIX"], "output_type": "BOGUS",
        })


def test_operator_param_domain_order_preserved() -> None:
    op = OperatorDef(
        name="foo",
        input_types=(DataType.MATRIX,),
        output_type=DataType.MATRIX,
        param_domains=(
            ("window", ParamDomain.int_range(1, 512)),
            ("k", ParamDomain.float_range(0.0, 1.0)),
        ),
    )
    d = operator_to_dict(op)
    assert list(d["param_domains"].keys()) == ["window", "k"]
    assert operator_from_dict(d).param_names == ("window", "k")


# --- schema version ---

def test_validate_schema_version_ok() -> None:
    assert validate_schema_version(
        {"schema_version": SERIALIZATION_SCHEMA_VERSION}
    ) == SERIALIZATION_SCHEMA_VERSION


def test_validate_schema_version_missing() -> None:
    with pytest.raises(SerializationError):
        validate_schema_version({})


def test_validate_schema_version_mismatch() -> None:
    with pytest.raises(UnsupportedSchemaError):
        validate_schema_version({"schema_version": 999})