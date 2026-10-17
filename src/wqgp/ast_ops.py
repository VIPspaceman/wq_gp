"""AST simplification and field-counting utilities.

Design decisions:
- Simplification rules are strictly semantic-equivalence rules that
  hold under IEEE 754 for the numeric operators we simplify. Rules
  whose behavior under NaN, infinity, or signed zero is not certain
  are NOT applied. See D-164.
- Simplification is bottom-up: children first, then the current node,
  then re-simplify the result to catch cascades. This lets
  `max(max(x, x), x)` collapse to `x` in one pass.
- Semantic equality for the `max(x, x)` and `min(x, x)` rules uses
  `ASTNode.fingerprint()`, not structural equality, so that
  `max(add(a, b), add(b, a))` is recognized as `max(x, x)` under
  commutative normalization.
- `count_distinct_fields` counts fields in the simplified tree, not
  the raw tree. See D-164.
"""
from __future__ import annotations

from wqgp.ast import ASTNode, NodeKind


def _is_zero_literal(node: ASTNode) -> bool:
    return (
        node.kind is NodeKind.LITERAL
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 0.0
    )


def _is_one_literal(node: ASTNode) -> bool:
    return (
        node.kind is NodeKind.LITERAL
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 1.0
    )


def _semantic_equal(a: ASTNode, b: ASTNode) -> bool:
    if a is b:
        return True
    if a == b:
        return True
    return a.fingerprint() == b.fingerprint()


def _try_simplify_call(node: ASTNode) -> ASTNode | None:
    """Return a simplified replacement for `node`, or None."""
    assert node.op_def is not None
    name = node.op_def.name
    args = node.children

    if name in ("max", "min") and len(args) == 2:
        a, b = args
        if _semantic_equal(a, b):
            return a
        return None

    if name == "add" and len(args) == 2:
        a, b = args
        if _is_zero_literal(a):
            return b
        if _is_zero_literal(b):
            return a
        return None

    if name == "subtract" and len(args) == 2:
        a, b = args
        if _is_zero_literal(b):
            return a
        return None

    if name == "multiply" and len(args) == 2:
        a, b = args
        if _is_one_literal(a):
            return b
        if _is_one_literal(b):
            return a
        return None

    if name == "divide" and len(args) == 2:
        a, b = args
        if _is_one_literal(b):
            return a
        return None

    return None


def simplify_ast(ast: ASTNode) -> ASTNode:
    """Return a semantically equivalent AST with redundant nodes removed.

    Applies the safe rules from D-164, recursively, until no rule
    fires. The returned node is re-validated by `ASTNode.call`, so a
    simplification that would produce a type-invalid tree raises a
    `ValueError` instead of silently corrupting the AST.
    """
    if ast.kind is not NodeKind.CALL:
        return ast

    new_children = tuple(simplify_ast(c) for c in ast.children)
    assert ast.op_def is not None
    current = ASTNode.call(ast.op_def, new_children, ast.params)

    simplified = _try_simplify_call(current)
    if simplified is None:
        return current

    # The simplification may expose new opportunities one level up
    # or inside the returned subtree.
    return simplify_ast(simplified)


def collect_fields(ast: ASTNode) -> frozenset[str]:
    """Distinct field IDs appearing anywhere in the tree.

    Counts fields in the tree as given. Callers who want the count
    for a raw (unsimplified) AST should call `simplify_ast` first.
    """
    if ast.kind is NodeKind.FIELD_REF:
        assert ast.field_def is not None
        return frozenset({ast.field_def.field_id})
    if ast.kind is NodeKind.LITERAL:
        return frozenset()
    result: set[str] = set()
    for c in ast.children:
        result |= collect_fields(c)
    return frozenset(result)


def count_distinct_fields(ast: ASTNode) -> int:
    """Number of distinct field IDs in the tree.

    Convention: callers should pass a *simplified* AST. See D-164.
    """
    return len(collect_fields(ast))