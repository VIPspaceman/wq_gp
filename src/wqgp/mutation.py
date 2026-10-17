"""Type-aware mutation operators for the GP loop.

Design decisions:
- One mutation per call, no recursion. See D-045.
- Node location is by integer path, not parent pointers. See D-046.
- A mutation must change the fingerprint or it is retried. See D-047.
- Operator swap requires matching parameter names. See D-048.
- Subtree swap uses a small-budget builder. See D-049.
- AST traversal utilities live in ast.py. See D-051.
- `mutate_field_weighted` mirrors `mutate_field` but samples with
  inverse-frequency weighting when the caller supplies a field_usage
  map. The Mutator picks the weighted variant automatically when a
  map is present. See D-165.
"""
from __future__ import annotations

import random
from typing import Callable, Mapping

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


class MutationError(RuntimeError):
    """Raised when no strategy produced a distinct-fingerprint AST."""


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


# Same clamp values as population.py; see D-165.
_FIELD_WEIGHT_ALPHA = 0.5
_FIELD_WEIGHT_MIN = 0.25
_FIELD_WEIGHT_MAX = 4.0


Strategy = Callable[[ASTNode, Pool, random.Random], ASTNode | None]


def mutate_window(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
    candidates: list[tuple[Path, int, ParamDomain]] = []
    for path, node in iter_calls(ast):
        assert node.op_def is not None
        for idx, (_, dom) in enumerate(node.op_def.param_domains):
            if dom.kind in (ParamKind.INT_RANGE, ParamKind.FLOAT_RANGE):
                candidates.append((path, idx, dom))
    if not candidates:
        return None

    path, idx, dom = rng.choice(candidates)
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
    """Uniform field swap: same-type field, random choice."""
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


def mutate_field_weighted(
    ast: ASTNode,
    pool: Pool,
    rng: random.Random,
    field_usage: Mapping[str, int],
) -> ASTNode | None:
    """Field swap using inverse-frequency weighting.

    Same as `mutate_field` but the candidate fields are sampled with
    weight `clamp(1/(1+count)^0.5, 0.25, 4.0)` where `count` is the
    number of individuals in the current population that already use
    that field. See D-165.
    """
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

    if not field_usage:
        new_field = rng.choice(candidates)
    else:
        weights: list[float] = []
        for f in candidates:
            count = field_usage.get(f.field_id, 0)
            w = 1.0 / ((1.0 + count) ** _FIELD_WEIGHT_ALPHA)
            if w < _FIELD_WEIGHT_MIN:
                w = _FIELD_WEIGHT_MIN
            elif w > _FIELD_WEIGHT_MAX:
                w = _FIELD_WEIGHT_MAX
            weights.append(w)
        new_field = rng.choices(candidates, weights=weights, k=1)[0]

    try:
        return replace_subtree(ast, path, ASTNode.field_ref(new_field))
    except ValueError:
        return None


def mutate_operator(
    ast: ASTNode, pool: Pool, rng: random.Random
) -> ASTNode | None:
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


class Mutator:
    """Runs mutation strategies in random order until one yields a
    distinct fingerprint.

    Not thread-safe: hold one Mutator per thread, or serialize calls.

    When `field_usage` is supplied, the `mutate_field` strategy is
    replaced by `mutate_field_weighted` at dispatch time. The list
    `self.strategies` continues to report the caller's original
    strategies; the swap is internal. See D-165.
    """

    __slots__ = (
        "_pool",
        "_strategies",
        "_max_attempts",
        "_field_usage",
    )

    def __init__(
        self,
        pool: Pool,
        *,
        strategies: tuple[Strategy, ...] = DEFAULT_STRATEGIES,
        max_attempts: int = 20,
        field_usage: Mapping[str, int] | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._pool = pool
        self._strategies = strategies
        self._max_attempts = max_attempts
        self._field_usage = field_usage or {}

    @property
    def strategies(self) -> tuple[Strategy, ...]:
        return self._strategies

    def _dispatch(
        self, strategy: Strategy, ast: ASTNode, rng: random.Random
    ) -> ASTNode | None:
        """Call `strategy`, substituting the weighted variant for
        `mutate_field` when field usage is available."""
        if strategy is mutate_field and self._field_usage:
            return mutate_field_weighted(
                ast, self._pool, rng, self._field_usage,
            )
        return strategy(ast, self._pool, rng)

    def mutate(self, ast: ASTNode, rng: random.Random) -> ASTNode:
        original_fp = ast.fingerprint()
        for _ in range(self._max_attempts):
            strategy = rng.choice(self._strategies)
            candidate = self._dispatch(strategy, ast, rng)
            if candidate is None:
                continue
            if candidate.fingerprint() == original_fp:
                continue
            return candidate
        raise MutationError(
            f"no strategy produced a distinct fingerprint in "
            f"{self._max_attempts} attempts"
        )