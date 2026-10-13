"""ASTNode: the expression tree that the GP evolves over.

Design decisions vs. the reference project (VIPspaceman/wq):
- No associative flattening in Phase 1. OperatorDef is fixed-arity, so
  flattening would produce arity-mismatched CALLs. See DECISIONS.md D-010.
- SCALAR literals broadcast to numeric positions only, never to GROUP or
  UNIVERSE. See DECISIONS.md D-011.
- Integer-valued float literals are normalized to int in the fingerprint.
  See DECISIONS.md D-012.
- Tree traversal and replacement utilities live here so both mutation
  and crossover can share them. See DECISIONS.md D-051.
- Expression rendering (`render`) also lives here. See DECISIONS.md D-066.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from wqgp.field import FieldDef
from wqgp.operator import OperatorDef
from wqgp.types import DataType, ScalarValue


class NodeKind(str, Enum):
    LITERAL = "LITERAL"
    FIELD_REF = "FIELD_REF"
    CALL = "CALL"


# Types a SCALAR literal may broadcast into.
_NUMERIC_TYPES = frozenset(
    {DataType.MATRIX, DataType.VECTOR, DataType.SCALAR}
)


def _compatible(actual: DataType, expected: frozenset[DataType]) -> bool:
    """True iff an argument of type `actual` is acceptable where `expected`
    is the operator's declared type set."""
    if actual in expected:
        return True
    # SCALAR literals broadcast into numeric positions only.
    if actual is DataType.SCALAR and (expected & _NUMERIC_TYPES):
        return True
    return False


def _norm_scalar(value: ScalarValue) -> str:
    """Canonical text for a scalar, collapsing 1.0 -> 1."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value)


def _canonical(node: "ASTNode") -> str:
    """Canonical string form used for fingerprinting.

    Determinism rules:
      - FIELD_REF: by field_id only.
      - LITERAL:    normalized scalar.
      - CALL:       sorted children iff symmetric, or commutative with
                    arity == 2. Params are NEVER reordered.
    """
    if node.kind is NodeKind.LITERAL:
        return f"L({_norm_scalar(node.value)})"  # type: ignore[arg-type]
    if node.kind is NodeKind.FIELD_REF:
        assert node.field_def is not None
        return f"F({node.field_def.field_id})"
    # CALL
    assert node.op_def is not None
    op = node.op_def
    child_strs = [_canonical(c) for c in node.children]
    if op.symmetric or (op.commutative and op.arity == 2):
        child_strs.sort()
    param_strs = [
        f"{name}={_norm_scalar(v)}"
        for (name, _), v in zip(op.param_domains, node.params)
    ]
    return f"C({op.name}|{','.join(child_strs)}|{','.join(param_strs)})"


@dataclass(frozen=True, slots=True)
class ASTNode:
    """Immutable expression-tree node.

    Exactly one of (value, field_def, op_def) is populated, selected by
    `kind`. Use the classmethods `literal`, `field_ref`, `call` instead of
    the raw constructor.
    """
    kind: NodeKind
    value: ScalarValue | None = None
    field_def: FieldDef | None = None
    op_def: OperatorDef | None = None
    children: tuple["ASTNode", ...] = ()
    params: tuple[ScalarValue, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is NodeKind.LITERAL:
            self._validate_literal()
        elif self.kind is NodeKind.FIELD_REF:
            self._validate_field_ref()
        elif self.kind is NodeKind.CALL:
            self._validate_call()
        else:
            raise ValueError(f"unknown node kind: {self.kind!r}")

    # --- validators ---

    def _validate_literal(self) -> None:
        if self.value is None:
            raise ValueError("LITERAL node requires `value`")
        if isinstance(self.value, bool):
            raise ValueError("LITERAL value must not be bool")
        if not isinstance(self.value, (int, float, str)):
            raise ValueError(
                f"LITERAL value must be int/float/str, "
                f"got {type(self.value).__name__}"
            )
        if self.field_def is not None or self.op_def is not None:
            raise ValueError("LITERAL node must not set field_def/op_def")
        if self.children or self.params:
            raise ValueError("LITERAL node must not have children/params")

    def _validate_field_ref(self) -> None:
        if self.field_def is None:
            raise ValueError("FIELD_REF node requires `field_def`")
        if self.value is not None or self.op_def is not None:
            raise ValueError("FIELD_REF node must not set value/op_def")
        if self.children or self.params:
            raise ValueError("FIELD_REF node must not have children/params")

    def _validate_call(self) -> None:
        if self.op_def is None:
            raise ValueError("CALL node requires `op_def`")
        if self.value is not None or self.field_def is not None:
            raise ValueError("CALL node must not set value/field_def")

        op = self.op_def
        if len(self.children) != op.arity:
            raise ValueError(
                f"CALL to {op.name!r}: expected {op.arity} data args, "
                f"got {len(self.children)}"
            )
        if len(self.params) != len(op.param_domains):
            raise ValueError(
                f"CALL to {op.name!r}: expected "
                f"{len(op.param_domains)} params, got {len(self.params)}"
            )

        for pos, child in enumerate(self.children):
            actual = child.data_type
            expected = op.input_types[pos]
            if not _compatible(actual, expected):
                expected_str = ", ".join(sorted(t.value for t in expected))
                raise ValueError(
                    f"CALL to {op.name!r} position {pos}: expected "
                    f"{{{expected_str}}}, got {actual.value}"
                )

        for (name, dom), val in zip(op.param_domains, self.params):
            if not dom.contains(val):
                raise ValueError(
                    f"CALL to {op.name!r} param {name!r}: value {val!r} "
                    f"not in domain"
                )

    # --- type ---

    @property
    def data_type(self) -> DataType:
        if self.kind is NodeKind.LITERAL:
            return DataType.SCALAR
        if self.kind is NodeKind.FIELD_REF:
            assert self.field_def is not None
            return self.field_def.data_type
        assert self.op_def is not None
        return self.op_def.output_type

    # --- constructors ---

    @classmethod
    def literal(cls, value: ScalarValue) -> "ASTNode":
        return cls(kind=NodeKind.LITERAL, value=value)

    @classmethod
    def field_ref(cls, field_def: FieldDef) -> "ASTNode":
        return cls(kind=NodeKind.FIELD_REF, field_def=field_def)

    @classmethod
    def call(
        cls,
        op_def: OperatorDef,
        children: tuple["ASTNode", ...],
        params: tuple[ScalarValue, ...] = (),
    ) -> "ASTNode":
        return cls(
            kind=NodeKind.CALL,
            op_def=op_def,
            children=children,
            params=params,
        )

    # --- fingerprint ---

    def canonical(self) -> str:
        return _canonical(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


# --- tree utilities (shared by mutation, crossover, serialization) ----------

# A path from the root: () is the root, (0,) is root's first child, etc.
Path = tuple[int, ...]


def iter_nodes(
    ast: ASTNode, prefix: Path = ()
) -> Iterator[tuple[Path, ASTNode]]:
    """Yield (path, node) for every node in the tree, pre-order."""
    yield prefix, ast
    for i, child in enumerate(ast.children):
        yield from iter_nodes(child, prefix + (i,))


def iter_calls(ast: ASTNode) -> Iterator[tuple[Path, ASTNode]]:
    """Yield (path, node) for every CALL node, pre-order."""
    for path, node in iter_nodes(ast):
        if node.kind is NodeKind.CALL:
            yield path, node


def iter_field_refs(ast: ASTNode) -> Iterator[tuple[Path, ASTNode]]:
    """Yield (path, node) for every FIELD_REF node, pre-order."""
    for path, node in iter_nodes(ast):
        if node.kind is NodeKind.FIELD_REF:
            yield path, node


def replace_subtree(
    ast: ASTNode, path: Path, new_subtree: ASTNode
) -> ASTNode:
    """Return a copy of `ast` with the subtree at `path` replaced.

    Path () replaces the whole tree. Any other path must descend through
    CALL nodes and stay within the arity of each operator; violations
    raise ValueError.
    """
    if not path:
        return new_subtree
    if ast.kind is not NodeKind.CALL:
        raise ValueError("cannot descend into a non-CALL node")
    index = path[0]
    if index < 0 or index >= len(ast.children):
        raise ValueError(f"child index {index} out of range")
    new_children = list(ast.children)
    new_children[index] = replace_subtree(
        ast.children[index], path[1:], new_subtree
    )
    assert ast.op_def is not None
    return ASTNode.call(ast.op_def, tuple(new_children), ast.params)


def replace_param(
    ast: ASTNode,
    path: Path,
    param_index: int,
    new_value: ScalarValue,
) -> ASTNode:
    """Return a copy of `ast` with one parameter of the CALL at `path`
    replaced. Only the CALL at `path` itself is modified."""
    if not path:
        if ast.kind is not NodeKind.CALL:
            raise ValueError("cannot set a parameter on a non-CALL node")
        assert ast.op_def is not None
        if not 0 <= param_index < len(ast.params):
            raise ValueError(
                f"param index {param_index} out of range "
                f"(have {len(ast.params)})"
            )
        new_params = list(ast.params)
        new_params[param_index] = new_value
        return ASTNode.call(ast.op_def, ast.children, tuple(new_params))
    if ast.kind is not NodeKind.CALL:
        raise ValueError("cannot descend into a non-CALL node")
    index = path[0]
    if index < 0 or index >= len(ast.children):
        raise ValueError(f"child index {index} out of range")
    new_children = list(ast.children)
    new_children[index] = replace_param(
        ast.children[index], path[1:], param_index, new_value
    )
    assert ast.op_def is not None
    return ASTNode.call(ast.op_def, tuple(new_children), ast.params)


def tree_depth(ast: ASTNode) -> int:
    """Number of nodes on the longest root-to-leaf path. Leaf = 1."""
    if not ast.children:
        return 1
    return 1 + max(tree_depth(c) for c in ast.children)


def tree_size(ast: ASTNode) -> int:
    """Total node count (root + every descendant)."""
    return 1 + sum(tree_size(c) for c in ast.children)


# --- expression rendering ---------------------------------------------------

def _render_scalar(value: ScalarValue) -> str:
    """Render a scalar for BRAIN expression syntax.

    Integers and integer-valued floats render without a decimal point.
    Strings render verbatim (BRAIN does not quote grouping identifiers
    like `sector`)."""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def render(node: ASTNode) -> str:
    """Render an AST as a BRAIN-compatible expression string.

    Parameters are appended after the data arguments, matching the
    platform's call syntax: ts_mean(close, 5).
    """
    if node.kind is NodeKind.LITERAL:
        return _render_scalar(node.value)  # type: ignore[arg-type]
    if node.kind is NodeKind.FIELD_REF:
        assert node.field_def is not None
        return node.field_def.field_id
    assert node.op_def is not None
    args = [render(c) for c in node.children]
    args.extend(_render_scalar(p) for p in node.params)
    return f"{node.op_def.name}({', '.join(args)})"