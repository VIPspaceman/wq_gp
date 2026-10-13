"""Tests for field/operator pools. Phase 2 Step 2 gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.pool import FieldPool, OperatorPool, Pool, PoolError
from wqgp.types import DataType, ParamDomain, SemanticKind


# --- fixtures ---

def _close() -> FieldDef:
    return FieldDef("close", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _open() -> FieldDef:
    return FieldDef("open", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _volume() -> FieldDef:
    return FieldDef("volume", DataType.MATRIX, SemanticKind.NUMERIC, "pv1")


def _sector() -> FieldDef:
    return FieldDef("sector", DataType.GROUP, SemanticKind.CATEGORICAL, "pv1")


def _add() -> OperatorDef:
    return OperatorDef(
        "add", (DataType.MATRIX, DataType.MATRIX), DataType.MATRIX,
        commutative=True, associative=True, symmetric=True,
    )


def _sub() -> OperatorDef:
    return OperatorDef(
        "sub", (DataType.MATRIX, DataType.MATRIX), DataType.MATRIX,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean", (DataType.MATRIX,), DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _group_rank() -> OperatorDef:
    return OperatorDef(
        "group_rank", (DataType.MATRIX, DataType.GROUP), DataType.MATRIX,
    )


# --- FieldPool ---

def test_field_pool_by_id() -> None:
    p = FieldPool([_close(), _open()])
    assert p.by_id("close").field_id == "close"


def test_field_pool_by_id_unknown_raises() -> None:
    p = FieldPool([_close()])
    with pytest.raises(KeyError):
        p.by_id("nope")


def test_field_pool_by_type() -> None:
    p = FieldPool([_close(), _open(), _volume(), _sector()])
    assert {f.field_id for f in p.by_type(DataType.MATRIX)} == {
        "close", "open", "volume",
    }
    assert {f.field_id for f in p.by_type(DataType.GROUP)} == {"sector"}
    assert p.by_type(DataType.VECTOR) == ()


def test_field_pool_by_type_sorted() -> None:
    p = FieldPool([_volume(), _close(), _open()])
    ids = [f.field_id for f in p.by_type(DataType.MATRIX)]
    assert ids == ["close", "open", "volume"]


def test_field_pool_all_sorted() -> None:
    p = FieldPool([_volume(), _close(), _open(), _sector()])
    assert [f.field_id for f in p.all()] == [
        "close", "open", "sector", "volume",
    ]


def test_field_pool_duplicate_rejected() -> None:
    with pytest.raises(PoolError):
        FieldPool([_close(), _close()])


def test_field_pool_len_and_contains() -> None:
    p = FieldPool([_close(), _open()])
    assert len(p) == 2
    assert "close" in p
    assert "nope" not in p
    assert 5 not in p


def test_field_pool_iter_is_deterministic() -> None:
    p = FieldPool([_volume(), _close(), _open()])
    assert [f.field_id for f in p] == [f.field_id for f in p]


# --- OperatorPool ---

def test_operator_pool_by_name() -> None:
    p = OperatorPool([_add(), _sub()])
    assert p.by_name("add").name == "add"


def test_operator_pool_by_arity() -> None:
    p = OperatorPool([_add(), _sub(), _ts_mean(), _group_rank()])
    assert {op.name for op in p.by_arity(1)} == {"ts_mean"}
    assert {op.name for op in p.by_arity(2)} == {
        "add", "sub", "group_rank",
    }


def test_operator_pool_by_output_type() -> None:
    p = OperatorPool([_add(), _sub()])
    assert {op.name for op in p.by_output_type(DataType.MATRIX)} == {
        "add", "sub",
    }
    assert p.by_output_type(DataType.SCALAR) == ()


def test_operator_pool_all_sorted() -> None:
    p = OperatorPool([_ts_mean(), _add(), _group_rank(), _sub()])
    assert [op.name for op in p.all()] == [
        "add", "group_rank", "sub", "ts_mean",
    ]


def test_operator_pool_duplicate_rejected() -> None:
    with pytest.raises(PoolError):
        OperatorPool([_add(), _add()])


# --- Pool: dict round-trip ---

def _make_pool() -> Pool:
    return Pool(
        fields=FieldPool([_close(), _open(), _volume(), _sector()]),
        operators=OperatorPool([_add(), _sub(), _ts_mean(), _group_rank()]),
    )


def test_pool_dict_roundtrip() -> None:
    p = _make_pool()
    rebuilt = Pool.from_dict(p.to_dict())
    assert rebuilt == p


def test_pool_to_dict_is_deterministic() -> None:
    p1 = _make_pool()
    p2 = Pool(
        fields=FieldPool([_sector(), _volume(), _open(), _close()]),
        operators=OperatorPool([_group_rank(), _ts_mean(), _sub(), _add()]),
    )
    # different insertion order -> identical serialized dict
    assert p1.to_dict() == p2.to_dict()


def test_pool_from_dict_missing_schema_version_rejected() -> None:
    p = _make_pool()
    d = p.to_dict()
    del d["schema_version"]
    with pytest.raises(Exception):
        Pool.from_dict(d)


def test_pool_from_dict_missing_fields_key_rejected() -> None:
    d = _make_pool().to_dict()
    del d["fields"]
    with pytest.raises(PoolError):
        Pool.from_dict(d)


def test_pool_from_dict_missing_operators_key_rejected() -> None:
    d = _make_pool().to_dict()
    del d["operators"]
    with pytest.raises(PoolError):
        Pool.from_dict(d)


# --- Pool: file IO ---

def test_pool_file_roundtrip(tmp_path: Path) -> None:
    p = _make_pool()
    f = tmp_path / "pool.json"
    p.save(f)
    assert Pool.from_file(f) == p


def test_pool_save_leaves_no_tmp(tmp_path: Path) -> None:
    p = _make_pool()
    f = tmp_path / "pool.json"
    p.save(f)
    assert list(tmp_path.glob("*.tmp")) == []


def test_pool_from_file_corrupt_json(tmp_path: Path) -> None:
    f = tmp_path / "pool.json"
    f.write_text("{ not json")
    with pytest.raises(PoolError):
        Pool.from_file(f)


def test_pool_from_file_missing(tmp_path: Path) -> None:
    with pytest.raises(PoolError):
        Pool.from_file(tmp_path / "nope.json")


# --- Pool: empty pools ---

def test_empty_pool_is_legal() -> None:
    p = Pool(fields=FieldPool([]), operators=OperatorPool([]))
    assert len(p.fields) == 0
    assert len(p.operators) == 0
    assert Pool.from_dict(p.to_dict()) == p


# --- Pool: summary ---

def test_pool_summary() -> None:
    s = _make_pool().summary()
    assert s["n_fields"] == 4
    assert s["n_operators"] == 4
    assert s["fields_MATRIX"] == 3
    assert s["fields_GROUP"] == 1