"""Tests for GP-layer serialization. Phase 3 Step 5 gate."""
from __future__ import annotations

import pytest

from wqgp.ast import ASTNode, NodeKind
from wqgp.field import FieldDef
from wqgp.gp_serialization import (
    GpSerializationError,
    ast_from_dict,
    ast_to_dict,
    individual_from_dict,
    individual_to_dict,
    population_from_dict,
    population_to_dict,
)
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.population import Individual, Population
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
        commutative=True, associative=True, symmetric=True,
    )


def _ts_mean() -> OperatorDef:
    return OperatorDef(
        "ts_mean",
        (DataType.MATRIX,),
        DataType.MATRIX,
        param_domains=(("window", ParamDomain.int_range(1, 512)),),
    )


def _fields() -> dict[str, FieldDef]:
    return {f.field_id: f for f in (_close(), _open(), _sector())}


def _ops() -> dict[str, OperatorDef]:
    return {o.name: o for o in (_add(), _ts_mean())}


def _flookup(fid: str) -> FieldDef:
    try:
        return _fields()[fid]
    except KeyError:
        raise KeyError(fid)


def _olookup(name: str) -> OperatorDef:
    try:
        return _ops()[name]
    except KeyError:
        raise KeyError(name)


# --- AST --------------------------------------------------------------------

def test_ast_literal_roundtrip() -> None:
    node = ASTNode.literal(5)
    d = ast_to_dict(node)
    assert d == {"kind": "LITERAL", "value": 5}
    back = ast_from_dict(d, _flookup, _olookup)
    assert back.fingerprint() == node.fingerprint()


def test_ast_field_ref_roundtrip() -> None:
    node = ASTNode.field_ref(_close())
    d = ast_to_dict(node)
    assert d == {"kind": "FIELD_REF", "field_id": "close"}
    back = ast_from_dict(d, _flookup, _olookup)
    assert back.fingerprint() == node.fingerprint()


def test_ast_call_roundtrip() -> None:
    node = ASTNode.call(
        _ts_mean(), (ASTNode.field_ref(_close()),), params=(5,)
    )
    back = ast_from_dict(ast_to_dict(node), _flookup, _olookup)
    assert back.fingerprint() == node.fingerprint()


def test_ast_call_with_two_children_roundtrip() -> None:
    node = ASTNode.call(
        _add(),
        (ASTNode.field_ref(_close()), ASTNode.field_ref(_open())),
    )
    back = ast_from_dict(ast_to_dict(node), _flookup, _olookup)
    assert back.fingerprint() == node.fingerprint()


def test_ast_unknown_kind_rejected() -> None:
    with pytest.raises(GpSerializationError):
        ast_from_dict({"kind": "BOGUS"}, _flookup, _olookup)


def test_ast_field_ref_unknown_field() -> None:
    with pytest.raises(GpSerializationError):
        ast_from_dict(
            {"kind": "FIELD_REF", "field_id": "nope"},
            _flookup, _olookup,
        )


def test_ast_call_unknown_operator() -> None:
    with pytest.raises(GpSerializationError):
        ast_from_dict(
            {
                "kind": "CALL",
                "operator": "bogus",
                "children": [],
                "params": [],
            },
            _flookup, _olookup,
        )


def test_ast_call_invalid_arity_rejected() -> None:
    with pytest.raises(GpSerializationError):
        ast_from_dict(
            {
                "kind": "CALL",
                "operator": "ts_mean",
                "children": [],   # ts_mean takes 1
                "params": [5],
            },
            _flookup, _olookup,
        )


def test_ast_not_a_dict_rejected() -> None:
    with pytest.raises(GpSerializationError):
        ast_from_dict([1, 2, 3], _flookup, _olookup)


# --- Individual -------------------------------------------------------------

def _ind(
    ast: ASTNode,
    fitness: float | None = None,
    generation: int = 0,
) -> Individual:
    if fitness is None:
        return Individual.from_ast(ast, generation=generation)
    return Individual.from_ast(
        ast,
        fitness=fitness,
        status=EvaluationStatus.SUCCESS,
        generation=generation,
    )


def test_individual_unevaluated_roundtrip() -> None:
    ind = _ind(ASTNode.field_ref(_close()))
    back = individual_from_dict(
        individual_to_dict(ind), _flookup, _olookup
    )
    assert back.fingerprint == ind.fingerprint
    assert back.fitness is None
    assert back.status is EvaluationStatus.PENDING


def test_individual_evaluated_roundtrip() -> None:
    ast = ASTNode.call(_ts_mean(), (ASTNode.field_ref(_close()),), params=(5,))
    ind = Individual.from_ast(
        ast,
        fitness=1.5,
        metrics=(("sharpe", 2.0), ("turnover", 0.3)),
        status=EvaluationStatus.SUCCESS,
        generation=3,
        parents=("parent_a", "parent_b"),
    )
    back = individual_from_dict(
        individual_to_dict(ind), _flookup, _olookup
    )
    assert back.fingerprint == ind.fingerprint
    assert back.fitness == 1.5
    assert set(back.metrics) == set(ind.metrics)
    assert back.generation == 3
    assert back.parents == ("parent_a", "parent_b")
    assert back.status is EvaluationStatus.SUCCESS


def test_individual_missing_fingerprint_rejected() -> None:
    d = individual_to_dict(_ind(ASTNode.field_ref(_close())))
    del d["fingerprint"]
    with pytest.raises(GpSerializationError):
        individual_from_dict(d, _flookup, _olookup)


def test_individual_bad_status_rejected() -> None:
    d = individual_to_dict(_ind(ASTNode.field_ref(_close())))
    d["status"] = "BOGUS"
    with pytest.raises(GpSerializationError):
        individual_from_dict(d, _flookup, _olookup)


def test_individual_failed_roundtrip() -> None:
    ind = Individual(
        ast=ASTNode.field_ref(_close()),
        fingerprint=ASTNode.field_ref(_close()).fingerprint(),
        status=EvaluationStatus.FAILED,
    )
    back = individual_from_dict(
        individual_to_dict(ind), _flookup, _olookup
    )
    assert back.status is EvaluationStatus.FAILED


# --- Population -------------------------------------------------------------

def test_population_roundtrip() -> None:
    a = _ind(ASTNode.field_ref(_close()), fitness=1.0)
    b = _ind(
        ASTNode.call(_ts_mean(), (ASTNode.field_ref(_open()),), params=(5,)),
        fitness=2.0,
    )
    c = _ind(ASTNode.field_ref(_sector()))   # unevaluated
    p = Population([a, b, c])
    back = population_from_dict(
        population_to_dict(p), _flookup, _olookup
    )
    assert sorted(i.fingerprint for i in back) == sorted(
        i.fingerprint for i in p
    )


def test_population_empty_roundtrip() -> None:
    p = Population.empty()
    back = population_from_dict(
        population_to_dict(p), _flookup, _olookup
    )
    assert len(back) == 0


def test_population_not_a_list_rejected() -> None:
    with pytest.raises(GpSerializationError):
        population_from_dict({"not": "a list"}, _flookup, _olookup)


def test_population_with_duplicate_fingerprint_rejected() -> None:
    ind = _ind(ASTNode.field_ref(_close()))
    payload = [individual_to_dict(ind), individual_to_dict(ind)]
    with pytest.raises(GpSerializationError):
        population_from_dict(payload, _flookup, _olookup)