"""Serialization for GP-layer objects: AST, Individual, Population.

Design decisions:
- Split from serialization.py to keep the Field/Operator layer free of
  GP concepts, and to avoid a circular import:
      pool.py -> serialization.py -> (FieldDef, OperatorDef)
      gp_serialization.py -> population.py -> pool.py
  There is no path from population.py back to this module.
- AST nodes reference fields by field_id and operators by name. The
  caller passes lookup functions, not a Pool object, so this module
  never imports pool.py. See D-065.
- Individual and Population round-trip through plain dicts and lists.
  The population payload is a list of individual dicts.
"""
from __future__ import annotations

from typing import Any, Callable

from wqgp.ast import ASTNode, NodeKind
from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.population import Individual, Population


FieldLookup = Callable[[str], FieldDef]
OperatorLookup = Callable[[str], OperatorDef]


class GpSerializationError(RuntimeError):
    """Raised when a serialized GP object cannot be parsed."""


# --- AST --------------------------------------------------------------------

def ast_to_dict(node: ASTNode) -> dict[str, Any]:
    """Render an AST node (and its subtree) as a JSON-friendly dict."""
    if node.kind is NodeKind.LITERAL:
        return {"kind": "LITERAL", "value": node.value}
    if node.kind is NodeKind.FIELD_REF:
        assert node.field_def is not None
        return {"kind": "FIELD_REF", "field_id": node.field_def.field_id}
    assert node.op_def is not None
    return {
        "kind": "CALL",
        "operator": node.op_def.name,
        "children": [ast_to_dict(c) for c in node.children],
        "params": list(node.params),
    }


def ast_from_dict(
    data: Any,
    field_lookup: FieldLookup,
    operator_lookup: OperatorLookup,
) -> ASTNode:
    """Parse an AST node from a dict produced by `ast_to_dict`."""
    if not isinstance(data, dict):
        raise GpSerializationError(
            f"AST node must be an object, got {type(data).__name__}"
        )
    kind = data.get("kind")

    if kind == "LITERAL":
        if "value" not in data:
            raise GpSerializationError("LITERAL node missing 'value'")
        try:
            return ASTNode.literal(data["value"])
        except (ValueError, TypeError) as e:
            raise GpSerializationError(f"invalid literal: {e}") from e

    if kind == "FIELD_REF":
        fid = data.get("field_id")
        if not isinstance(fid, str) or not fid:
            raise GpSerializationError(
                "FIELD_REF node missing 'field_id'"
            )
        try:
            fdef = field_lookup(fid)
        except KeyError as e:
            raise GpSerializationError(
                f"unknown field_id {fid!r}"
            ) from e
        return ASTNode.field_ref(fdef)

    if kind == "CALL":
        opname = data.get("operator")
        if not isinstance(opname, str) or not opname:
            raise GpSerializationError("CALL node missing 'operator'")
        try:
            opdef = operator_lookup(opname)
        except KeyError as e:
            raise GpSerializationError(
                f"unknown operator {opname!r}"
            ) from e
        raw_children = data.get("children")
        if not isinstance(raw_children, list):
            raise GpSerializationError(
                "CALL node 'children' must be a list"
            )
        children = tuple(
            ast_from_dict(c, field_lookup, operator_lookup)
            for c in raw_children
        )
        raw_params = data.get("params", [])
        if not isinstance(raw_params, list):
            raise GpSerializationError(
                "CALL node 'params' must be a list"
            )
        params = tuple(raw_params)
        try:
            return ASTNode.call(opdef, children, params)
        except ValueError as e:
            raise GpSerializationError(
                f"invalid CALL to {opname!r}: {e}"
            ) from e

    raise GpSerializationError(f"unknown AST node kind: {kind!r}")


# --- Individual -------------------------------------------------------------

def individual_to_dict(ind: Individual) -> dict[str, Any]:
    return {
        "ast": ast_to_dict(ind.ast),
        "fingerprint": ind.fingerprint,
        "fitness": ind.fitness,
        "metrics": [[k, v] for k, v in ind.metrics],
        "generation": ind.generation,
        "parents": list(ind.parents),
        "status": ind.status.value,
    }


def individual_from_dict(
    data: Any,
    field_lookup: FieldLookup,
    operator_lookup: OperatorLookup,
) -> Individual:
    if not isinstance(data, dict):
        raise GpSerializationError(
            f"Individual must be an object, got {type(data).__name__}"
        )
    if "ast" not in data:
        raise GpSerializationError("Individual missing 'ast'")
    if "fingerprint" not in data:
        raise GpSerializationError("Individual missing 'fingerprint'")

    ast = ast_from_dict(data["ast"], field_lookup, operator_lookup)

    raw_metrics = data.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raise GpSerializationError("Individual 'metrics' must be a list")
    try:
        metrics = tuple(
            (str(k), float(v)) for k, v in raw_metrics
        )
    except (TypeError, ValueError) as e:
        raise GpSerializationError(
            f"malformed Individual 'metrics': {e}"
        ) from e

    raw_parents = data.get("parents", [])
    if not isinstance(raw_parents, list):
        raise GpSerializationError(
            "Individual 'parents' must be a list"
        )
    parents = tuple(str(p) for p in raw_parents)

    raw_status = data.get("status", EvaluationStatus.PENDING.value)
    try:
        status = EvaluationStatus(raw_status)
    except ValueError as e:
        raise GpSerializationError(
            f"unknown EvaluationStatus: {raw_status!r}"
        ) from e

    raw_fitness = data.get("fitness")
    fitness = None if raw_fitness is None else float(raw_fitness)

    try:
        return Individual(
            ast=ast,
            fingerprint=data["fingerprint"],
            fitness=fitness,
            metrics=metrics,
            generation=int(data.get("generation", 0)),
            parents=parents,
            status=status,
        )
    except (ValueError, KeyError) as e:
        raise GpSerializationError(f"invalid Individual: {e}") from e


# --- Population -------------------------------------------------------------

def population_to_dict(pop: Population) -> list[dict[str, Any]]:
    return [individual_to_dict(i) for i in pop.all()]


def population_from_dict(
    data: Any,
    field_lookup: FieldLookup,
    operator_lookup: OperatorLookup,
) -> Population:
    if not isinstance(data, list):
        raise GpSerializationError(
            f"Population payload must be a list, got {type(data).__name__}"
        )
    individuals = [
        individual_from_dict(d, field_lookup, operator_lookup)
        for d in data
    ]
    try:
        return Population(individuals)
    except Exception as e:
        raise GpSerializationError(f"invalid Population: {e}") from e