"""Tests for the core type system. Phase 1 gate: all must pass before moving on."""
from __future__ import annotations

import pytest

from wqgp.types import DataType, ParamDomain, ParamKind, SemanticKind


# --- DataType / SemanticKind: they must stay separate ---

def test_data_type_has_no_categorical() -> None:
    """CATEGORICAL must NEVER leak into DataType. See DECISIONS.md D-001."""
    names = {m.name for m in DataType}
    assert "CATEGORICAL" not in names
    assert names == {"MATRIX", "VECTOR", "GROUP", "UNIVERSE", "SCALAR"}


def test_semantic_kind_carries_categorical() -> None:
    assert SemanticKind.CATEGORICAL.value == "CATEGORICAL"
    # str enum -> JSON-friendly
    assert isinstance(SemanticKind.CATEGORICAL, str)


def test_data_type_json_round_trip() -> None:
    # str, Enum means we can compare directly to the platform vocabulary.
    assert DataType.GROUP == "GROUP"
    assert DataType("MATRIX") is DataType.MATRIX


# --- ParamDomain: constructors + invariants ---

def test_int_range_happy_path() -> None:
    d = ParamDomain.int_range(1, 512, default=5)
    assert d.kind is ParamKind.INT_RANGE
    assert d.contains(5)
    assert d.contains(1)
    assert d.contains(512)


def test_int_range_rejects_bool_and_float() -> None:
    d = ParamDomain.int_range(1, 10)
    # bool is an int subclass -- must be rejected explicitly.
    assert not d.contains(True)
    assert not d.contains(3.0)
    assert not d.contains(0)
    assert not d.contains(11)


def test_float_range_happy_path() -> None:
    d = ParamDomain.float_range(0.0, 1.0, default=0.5)
    assert d.contains(0.5)
    assert d.contains(0)
    assert not d.contains(1.1)


def test_choice_happy_path() -> None:
    d = ParamDomain.choice(("add", "sub"), default="add")
    assert d.contains("add")
    assert not d.contains("mul")
    assert not d.contains(1)


# --- ParamDomain: invariant enforcement ---

def test_range_requires_bounds() -> None:
    with pytest.raises(ValueError):
        ParamDomain(kind=ParamKind.INT_RANGE)


def test_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        ParamDomain.int_range(10, 1)


def test_range_rejects_out_of_range_default() -> None:
    with pytest.raises(ValueError):
        ParamDomain.int_range(1, 10, default=99)


def test_choice_requires_options() -> None:
    with pytest.raises(ValueError):
        ParamDomain(kind=ParamKind.CHOICE)


def test_choice_rejects_unknown_default() -> None:
    with pytest.raises(ValueError):
        ParamDomain.choice(("a", "b"), default="c")


# --- ParamDomain: frozen + hashable + equality ---

def test_param_domain_is_frozen_and_hashable() -> None:
    d = ParamDomain.int_range(1, 10)
    with pytest.raises(Exception):
        d.min_value = 5  # type: ignore[misc]
    assert hash(d) == hash(ParamDomain.int_range(1, 10))
    assert d == ParamDomain.int_range(1, 10)