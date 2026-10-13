"""Type-aware mutation operators for the GP loop.

Design decisions:
- One mutation per call, no recursion. See D-045.
- Node location is by integer path, not parent pointers. See D-046.
- A mutation must change the fingerprint or it is retried. See D-047.
- Operator swap requires matching parameter names. See D-048.
- Subtree swap uses a small-budget builder. See D-049.
- AST traversal and replacement utilities now live in ast.py. See D-051.

This module does NOT evaluate, does NOT talk to BRAIN, and does NOT
carry state between calls. `Mutator` is a thin dispatcher that tries
strategies in random order until one produces a distinct fingerprint.
"""
from __future__ import annotations

import random
from typing import Callable

from wqgp.ast import (
    ASTNode,
    Path,
    iter_calls,
    iter_field_refs,
    iter_nodes,
    replace_param,
    replace_subtree,
)
from wqgp.operator import OperatorDef
from wqgp.pool import Pool
from wqgp.population import GenerationError, RandomAstBuilder
from wqgp.types import DataType, ParamDomain, ParamKind


# --- errors -----------------------------------------------------------------

class MutationError(RuntimeError):
    """Raised when no strategy produced a distinct-fingerprint AST."""


# --- input-type compatibility (mirrors ASTNode._compatible) -----------------

_NUMERIC_TYPES = frozenset(
    {DataType.MATRIX, DataType.VECTOR, DataType.SCALAR}
)


def _inputs_compatible(
    op: OperatorDef, children: tuple[ASTNode, ...]
) -> bool:
    if len(op.input_types) != len(children):
        return False
    for expected, child in zip(op.input_types, children):
        actual = child.data_type
        if actual in expected:
            continue
        if actual is DataType.SCALAR and (expected & _NUMERIC_TYPES):
            continue
        return False
    return True


# --- param sampling (shared with population.py semantics) ------------------

def _sample_param(
    dom: ParamDomain, rng: random.Random
) -> int | float | str:
    if dom.kind is ParamKind.INT_RANGE:
        return rng.randint(
            int(dom.min_value), int(dom.max_value)  # type: ignore[arg-type]
        )
    if dom.kind is ParamKind.FLOAT_RANGE:
        return rng.uniform(
            float(dom.min_value), float(dom.max_value)  # type: ignore[arg-type]
        )
    if dom.kind is ParamKind.CHOICE:
        return rng.choice(list(dom.choices))
    raise MutationError(f"unknown param kind: {dom.kind!r}")


# --- strategies -------------------------------------------------------------

Strategy = Callable[[ASTNode, Pool, random.Random], ASTNode | None]


def mutate_window(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
    """Resample one integer or float parameter of a CALL node."""
    candidates: list[tuple[Path, int, ParamDomain]] = []
    for path, node in iter_calls(ast):
        assert node.op_def is not None
        for idx, (_, dom) in enumerate(node.op_def.param_domains):
            if dom.kind in (ParamKind.INT_RANGE, ParamKind.FLOAT_RANGE):
                candidates.append((path, idx, dom))
    if not candidates:
        return None

    path, idx, dom = rng.choice(candidates)
    # Walk to the node to read the current value.
    node = ast
    for step in path:
        node = node.children[step]
    old_value = node.params[idx]
    new_value = _sample_param(dom, rng)
    if new_value == old_value:
        return None
    try:
        return replace_param(ast, path, idx, new_value)
    except ValueError:
        return None


def mutate_field(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
    """Swap one FIELD_REF for another field of the same DataType."""
    refs = list(iter_field_refs(ast))
    if not refs:
        return None

    path, node = rng.choice(refs)
    assert node.field_def is not None
    current = node.field_def
    candidates = [
        f for f in pool.fields.by_type(current.data_type)
        if f.field_id != current.field_id
    ]
    if not candidates:
        return None

    new_field = rng.choice(candidates)
    try:
        return replace_subtree(ast, path, ASTNode.field_ref(new_field))
    except ValueError:
        return None


def mutate_operator(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
    """Swap the operator of a CALL node for a signature-compatible one.

    Compatible means: same arity, same output DataType, same parameter
    names (see D-048), and input types accept the existing children."""
    calls = list(iter_calls(ast))
    if not calls:
        return None
    rng.shuffle(calls)

    for path, node in calls:
        assert node.op_def is not None
        old_op = node.op_def
        same_arity = pool.operators.by_arity(old_op.arity)
        compatible = [
            op for op in same_arity
            if op.name != old_op.name
            and op.output_type is old_op.output_type
            and op.param_names == old_op.param_names
            and _inputs_compatible(op, node.children)
        ]
        if not compatible:
            continue

        new_op = rng.choice(compatible)
        try:
            new_call = ASTNode.call(new_op, node.children, node.params)
        except ValueError:
            continue
        try:
            return replace_subtree(ast, path, new_call)
        except ValueError:
            continue
    return None


def mutate_subtree(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
    """Replace a random subtree with a freshly built one of the same type."""
    nodes = list(iter_nodes(ast))
    if not nodes:
        return None
    rng.shuffle(nodes)

    builder = RandomAstBuilder(pool, max_depth=2, max_nodes=5)
    for path, node in nodes:
        target_type = node.data_type
        try:
            new_subtree = builder.build(target_type, rng)
        except GenerationError:
            continue
        try:
            return replace_subtree(ast, path, new_subtree)
        except ValueError:
            continue
    return None


DEFAULT_STRATEGIES: tuple[Strategy, ...] = (
    mutate_window,
    mutate_field,
    mutate_operator,
    mutate_subtree,
)


# --- dispatcher -------------------------------------------------------------

class Mutator:
    """Runs mutation strategies in random order until one yields a
    distinct fingerprint.

    Not thread-safe: hold one Mutator per thread, or serialize calls.
    """

    __slots__ = ("_pool", "_strategies", "_max_attempts")

    def __init__(
        self,
        pool: Pool,
        *,
        strategies: tuple[Strategy, ...] = DEFAULT_STRATEGIES,
        max_attempts: int = 20,
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._pool = pool
        self._strategies = strategies
        self._max_attempts = max_attempts

    @property
    def strategies(self) -> tuple[Strategy, ...]:
        return self._strategies

    def mutate(self, ast: ASTNode, rng: random.Random) -> ASTNode:
        original_fp = ast.fingerprint()
        for _ in range(self._max_attempts):
            strategy = rng.choice(self._strategies)
            candidate = strategy(ast, self._pool, rng)
            if candidate is None:
                continue
            if candidate.fingerprint() == original_fp:
                continue  # D-047: must produce a distinct fingerprint
            return candidate
        raise MutationError(
            f"no strategy produced a distinct fingerprint in "
            f"{self._max_attempts} attempts"
        )