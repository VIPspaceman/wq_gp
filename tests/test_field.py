"""Tests for FieldDef. Phase 1 gate: all must pass before moving on."""
from __future__ import annotations

import pytest

from wqgp.field import FieldDef
from wqgp.types import DataType, SemanticKind


# --- happy paths ---

def test_numeric_matrix_field() -> None:
    f = FieldDef(
        field_id="close",
        data_type=DataType.MATRIX,
        semantic_kind=SemanticKind.NUMERIC,
        dataset="pv1",
        coverage=0.98,
        delay=1,
        unit="USD",
    )
    assert f.field_id == "close"
    assert not f.is_categorical
    assert f.unit == "USD"


def test_categorical_group_field() -> None:
    f = FieldDef(
        field_id="sector",
        data_type=DataType.GROUP,
        semantic_kind=SemanticKind.CATEGORICAL,
        dataset="pv1",
        coverage=1.0,
        delay=1,
    )
    assert f.is_categorical
    assert f.unit is None


def test_defaults() -> None:
    f = FieldDef(
        field_id="volume",
        data_type=DataType.MATRIX,
        semantic_kind=SemanticKind.NUMERIC,
        dataset="pv1",
    )
    assert f.coverage == 1.0
    assert f.delay == 1
    assert f.unit is None


# --- invariant: field_id ---

def test_empty_field_id_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef("", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def test_whitespace_only_field_id_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef("   ", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def test_field_id_with_surrounding_whitespace_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef(" close ", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


# --- invariant: dataset ---

def test_empty_dataset_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef("close", DataType.MATRIX, SemanticKind.NUMERIC, "")


# --- invariant: coverage ---

def test_coverage_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef(
            "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1",
            coverage=-0.01,
        )


def test_coverage_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef(
            "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1",
            coverage=1.01,
        )


# --- invariant: delay ---

def test_negative_delay_rejected() -> None:
    with pytest.raises(ValueError):
        FieldDef(
            "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1",
            delay=-1,
        )


def test_zero_delay_allowed() -> None:
    f = FieldDef(
        "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1", delay=0
    )
    assert f.delay == 0


# --- invariant: CATEGORICAL => GROUP (one-way) ---

def test_categorical_must_be_group() -> None:
    with pytest.raises(ValueError):
        FieldDef(
            "sector", DataType.MATRIX, SemanticKind.CATEGORICAL, "pv1"
        )
    with pytest.raises(ValueError):
        FieldDef(
            "sector", DataType.VECTOR, SemanticKind.CATEGORICAL, "pv1"
        )


def test_group_can_be_non_categorical() -> None:
    """One-way constraint: GROUP does NOT require CATEGORICAL.
    See DECISIONS.md D-006."""
    f = FieldDef(
        "bucket", DataType.GROUP, SemanticKind.NUMERIC, "pv1"
    )
    assert not f.is_categorical


# --- frozen + hashable + equality ---

def test_field_is_frozen_and_hashable() -> None:
    f = FieldDef("close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    with pytest.raises(Exception):
        f.field_id = "open"  # type: ignore[misc]
    assert hash(f) == hash(
        FieldDef("close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")
    )
    assert f == FieldDef(
        "close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1"
    )