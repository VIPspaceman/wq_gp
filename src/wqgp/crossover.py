"""Subtree crossover for the GP loop.

Design decisions:
- AST utilities are shared from ast.py. See D-051.
- Only same-DataType subtrees are swapped. See D-052.
- One child produced per call, not two. See D-053.
- Depth and size limits enforced after crossover. See D-054.
- The root node is never swapped. See D-055.

This module does NOT evaluate, does NOT talk to BRAIN, and does NOT
carry state between calls. `Crossover` is a thin dispatcher that tries
random compatible swaps until one produces a child whose fingerprint
differs from both parents.
"""
from __future__ import annotations

import random

from wqgp.ast import ASTNode, Path, iter_nodes, replace_subtree, tree_depth, tree_size
from wqgp.pool import Pool  # imported for signature symmetry with Mutator


# --- errors -----------------------------------------------------------------

class CrossoverError(RuntimeError):
    """Raised when no crossover produced a distinct offspring."""


# --- core operation ---------------------------------------------------------

def subtree_crossover(
    a: ASTNode,
    b: ASTNode,
    rng: random.Random,
    *,
    max_depth: int = 6,
    max_nodes: int = 40,
) -> ASTNode | None:
    """Return a child of A where one non-root subtree of A has been
    replaced by a compatible non-root subtree of B, or None if no such
    swap is possible within the given limits.

    Compatibility means same `data_type` (D-052). Root swaps are
    forbidden (D-055). The result is checked against `max_depth` and
    `max_nodes` (D-054) before being returned; larger children are
    skipped and a smaller swap is tried.
    """
    if max_depth < 1 or max_nodes < 1:
        raise ValueError("max_depth and max_nodes must be >= 1")

    # Collect non-root nodes from each parent, grouped by DataType.
    a_nodes = [(p, n) for p, n in iter_nodes(a) if p]
    b_nodes = [(p, n) for p, n in iter_nodes(b) if p]
    if not a_nodes or not b_nodes:
        return None

    # Candidate swaps: (path_in_a, path_in_b, subtree_b).
    candidates: list[tuple[Path, Path, ASTNode]] = []
    for pa, na in a_nodes:
        for pb, nb in b_nodes:
            if na.data_type is nb.data_type:
                candidates.append((pa, pb, nb))
    if not candidates:
        return None

    rng.shuffle(candidates)
    for pa, _pb, subtree_b in candidates:
        try:
            child = replace_subtree(a, pa, subtree_b)
        except ValueError:
            continue
        if tree_depth(child) > max_depth:
            continue
        if tree_size(child) > max_nodes:
            continue
        return child
    return None


# --- dispatcher -------------------------------------------------------------

class Crossover:
    """Applies subtree crossover between two parents until the child
    differs from both by fingerprint.

    Not thread-safe: hold one Crossover per thread, or serialize calls.
    """

    __slots__ = ("_pool", "_max_attempts", "_max_depth", "_max_nodes")

    def __init__(
        self,
        pool: Pool,
        *,
        max_attempts: int = 20,
        max_depth: int = 6,
        max_nodes: int = 40,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        self._pool = pool
        self._max_attempts = max_attempts
        self._max_depth = max_depth
        self._max_nodes = max_nodes

    def crossover(
        self, a: ASTNode, b: ASTNode, rng: random.Random
    ) -> ASTNode:
        fp_a = a.fingerprint()
        fp_b = b.fingerprint()
        if fp_a == fp_b:
            raise CrossoverError("cannot crossover two identical ASTs")

        for _ in range(self._max_attempts):
            child = subtree_crossover(
                a, b, rng,
                max_depth=self._max_depth,
                max_nodes=self._max_nodes,
            )
            if child is None:
                # No compatible swap; retry with the same rng is useless.
                break
            fp_child = child.fingerprint()
            if fp_child == fp_a or fp_child == fp_b:
                continue  # D-047 analogue: must be distinct from both
            return child

        raise CrossoverError(
            f"no crossover produced a distinct offspring in "
            f"{self._max_attempts} attempts"
        )