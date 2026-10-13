"""Individuals and populations for the GP loop.

Design decisions:
- Individual equality is by fingerprint only. See D-039.
- Population is a plain class, not a frozen dataclass, so it can cache
  its fingerprint index. See D-040.
- RandomAstBuilder is stateful and single-use per build. See D-041.
- AST generation is bottom-up with three hard limits. See D-042.
- Generation failure raises; it does not return None. See D-043.
- Only SCALAR leaves can be literals. See D-044.

This module does NOT evaluate individuals and does NOT talk to BRAIN.
Fitness on an Individual is whatever the caller set; it may be None.
"""
from __future__ import annotations

import random
from types import MappingProxyType
from typing import Mapping

from wqgp.ast import ASTNode
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.pool import Pool
from wqgp.types import DataType, ParamDomain, ParamKind


# --- errors -----------------------------------------------------------------

class PopulationError(RuntimeError):
    """Base class for population-level failures."""


class DuplicateIndividualError(PopulationError):
    """Raised when two individuals share a fingerprint in one population."""


class GenerationError(PopulationError):
    """Raised when RandomAstBuilder cannot construct a valid AST."""


# --- Individual -------------------------------------------------------------

class Individual:
    """One candidate alpha expression plus its evaluation state.

    Equality and hashing are by fingerprint only: two Individuals with the
    same expression are the same Individual, regardless of fitness. This
    makes `set(individuals)` and `dict[fingerprint, Individual]` do the
    right thing during dedup.

    Instances are effectively immutable: attribute reassignment raises.
    Use `with_evaluation()` to obtain a new Individual carrying results.
    """

    __slots__ = (
        "ast",
        "fingerprint",
        "fitness",
        "metrics",
        "generation",
        "parents",
        "status",
    )

    def __init__(
        self,
        *,
        ast: ASTNode,
        fingerprint: str,
        fitness: float | None = None,
        metrics: tuple[tuple[str, float], ...] = (),
        generation: int = 0,
        parents: tuple[str, ...] = (),
        status: EvaluationStatus = EvaluationStatus.PENDING,
    ) -> None:
        if not fingerprint:
            raise ValueError("fingerprint must be non-empty")
        if generation < 0:
            raise ValueError("generation must be >= 0")
        for k, v in metrics:
            if not isinstance(k, str) or not k:
                raise ValueError("metric name must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"metric {k!r} must be int/float, got {type(v).__name__}"
                )
        if status is EvaluationStatus.SUCCESS and fitness is None:
            raise ValueError("SUCCESS individual requires fitness")

        self.ast = ast
        self.fingerprint = fingerprint
        self.fitness = fitness
        self.metrics = metrics
        self.generation = generation
        self.parents = parents
        self.status = status

    @classmethod
    def from_ast(
        cls,
        ast: ASTNode,
        *,
        fitness: float | None = None,
        metrics: tuple[tuple[str, float], ...] = (),
        generation: int = 0,
        parents: tuple[str, ...] = (),
        status: EvaluationStatus = EvaluationStatus.PENDING,
    ) -> "Individual":
        """Construct an Individual, computing the fingerprint from the AST."""
        return cls(
            ast=ast,
            fingerprint=ast.fingerprint(),
            fitness=fitness,
            metrics=metrics,
            generation=generation,
            parents=parents,
            status=status,
        )

    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def with_evaluation(
        self,
        *,
        fitness: float,
        metrics: tuple[tuple[str, float], ...] = (),
    ) -> "Individual":
        """Return a new Individual carrying evaluation results."""
        return Individual(
            ast=self.ast,
            fingerprint=self.fingerprint,
            fitness=fitness,
            metrics=metrics,
            generation=self.generation,
            parents=self.parents,
            status=EvaluationStatus.SUCCESS,
        )

    # --- immutability ---

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError(
                f"Individual is immutable; cannot reassign {name!r}"
            )
        object.__setattr__(self, name, value)

    # --- identity: by fingerprint only (D-039) ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        fit = "None" if self.fitness is None else f"{self.fitness:.4g}"
        return (
            f"Individual(fp={self.fingerprint[:8]}, "
            f"gen={self.generation}, fitness={fit})"
        )


# --- Population -------------------------------------------------------------

class Population:
    """A collection of Individuals with unique fingerprints.

    - Duplicates by fingerprint are rejected at construction.
    - Iteration order is deterministic (sorted by fingerprint).
    - `__eq__` is order-independent: same fingerprint set == equal.
    - Instances are logically immutable; internal indexes are MappingProxy.
    """

    __slots__ = ("_individuals", "_by_fp")

    def __init__(
        self,
        individuals: list[Individual] | tuple[Individual, ...],
    ) -> None:
        by_fp: dict[str, Individual] = {}
        for ind in individuals:
            if ind.fingerprint in by_fp:
                raise DuplicateIndividualError(
                    f"duplicate fingerprint in population: "
                    f"{ind.fingerprint[:8]}"
                )
            by_fp[ind.fingerprint] = ind

        self._by_fp: Mapping[str, Individual] = MappingProxyType(by_fp)
        self._individuals: tuple[Individual, ...] = tuple(
            by_fp[fp] for fp in sorted(by_fp)
        )

    # --- construction helpers ---

    @classmethod
    def empty(cls) -> "Population":
        return cls(())

    @classmethod
    def concat(cls, *populations: "Population") -> "Population":
        """Union of populations. Duplicates across inputs are deduplicated;
        the first occurrence (in argument order) wins."""
        seen: dict[str, Individual] = {}
        for pop in populations:
            for ind in pop:
                seen.setdefault(ind.fingerprint, ind)
        return cls(tuple(seen.values()))

    # --- access ---

    def by_fingerprint(self, fp: str) -> Individual | None:
        return self._by_fp.get(fp)

    def contains(self, fp: str) -> bool:
        return fp in self._by_fp

    def all(self) -> tuple[Individual, ...]:
        return self._individuals

    def evaluated(self) -> tuple[Individual, ...]:
        return tuple(i for i in self._individuals if i.is_evaluated())

    def with_fitness_at_least(
        self, threshold: float
    ) -> tuple[Individual, ...]:
        return tuple(
            i for i in self._individuals
            if i.fitness is not None and i.fitness >= threshold
        )

    # --- dunders ---

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self):
        return iter(self._individuals)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._by_fp
        if isinstance(item, Individual):
            return item.fingerprint in self._by_fp
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Population):
            return NotImplemented
        return set(self._by_fp) == set(other._by_fp)

    def __hash__(self) -> int:
        return hash(frozenset(self._by_fp))

    def __repr__(self) -> str:
        return f"Population(n={len(self._individuals)})"


# --- Random AST builder -----------------------------------------------------

def _sample_param(
    domain: ParamDomain, rng: random.Random
) -> int | float | str:
    """Draw a value from a ParamDomain."""
    if domain.kind is ParamKind.INT_RANGE:
        lo = int(domain.min_value)  # type: ignore[arg-type]
        hi = int(domain.max_value)  # type: ignore[arg-type]
        return rng.randint(lo, hi)
    if domain.kind is ParamKind.FLOAT_RANGE:
        lo = float(domain.min_value)  # type: ignore[arg-type]
        hi = float(domain.max_value)  # type: ignore[arg-type]
        return rng.uniform(lo, hi)
    if domain.kind is ParamKind.CHOICE:
        return rng.choice(list(domain.choices))
    raise GenerationError(f"unknown param kind: {domain.kind!r}")


# Literal values used when a SCALAR leaf is a constant rather than a field.
_LITERAL_POOL: tuple[int | float, ...] = (1, 5, 10, 0.5, 1.0, -1)


class RandomAstBuilder:
    """Builds random, type-valid ASTs from a Pool.

    Stateful and single-use per build: `build()` resets internal counters
    and stores the RNG on the instance. Do not call concurrently.
    """

    __slots__ = (
        "_pool",
        "_max_depth",
        "_max_nodes",
        "_max_operator_attempts",
        "_nodes_left",
        "_rng",
    )

    def __init__(
        self,
        pool: Pool,
        *,
        max_depth: int = 4,
        max_nodes: int = 25,
        max_operator_attempts: int = 15,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if max_operator_attempts < 1:
            raise ValueError("max_operator_attempts must be >= 1")
        self._pool = pool
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._max_operator_attempts = max_operator_attempts
        self._nodes_left = 0
        # Placeholder; overwritten by build(). Using a real Random rather
        # than None keeps internal methods free of None checks.
        self._rng: random.Random = random.Random()

    # --- public ---

    def build(self, target_type: DataType, rng: random.Random) -> ASTNode:
        """Return a random AST of `target_type`, or raise GenerationError."""
        self._rng = rng
        for _ in range(self._max_operator_attempts):
            self._nodes_left = self._max_nodes
            node = self._build(target_type, self._max_depth)
            if node is not None:
                return node
        raise GenerationError(
            f"failed to build AST of type {target_type.value} "
            f"within {self._max_operator_attempts} attempts"
        )

    # --- recursion ---

    def _build(self, target_type: DataType, depth: int) -> ASTNode | None:
        if self._nodes_left < 1:
            return None
        if depth <= 0:
            return self._build_leaf(target_type)
        call = self._build_call(target_type, depth)
        if call is not None:
            return call
        return self._build_leaf(target_type)

    def _build_leaf(self, target_type: DataType) -> ASTNode | None:
        # Only SCALAR may be a literal. See D-044.
        if target_type is DataType.SCALAR:
            scalar_fields = self._pool.fields.by_type(DataType.SCALAR)
            if scalar_fields and self._rng.random() < 0.5:
                self._nodes_left -= 1
                return ASTNode.field_ref(self._rng.choice(scalar_fields))
            return self._make_literal()

        fields = self._pool.fields.by_type(target_type)
        if not fields:
            return None
        self._nodes_left -= 1
        return ASTNode.field_ref(self._rng.choice(fields))

    def _make_literal(self) -> ASTNode:
        self._nodes_left -= 1
        return ASTNode.literal(self._rng.choice(_LITERAL_POOL))

    def _build_call(
        self, target_type: DataType, depth: int
    ) -> ASTNode | None:
        candidates = self._pool.operators.by_output_type(target_type)
        if not candidates:
            return None
        ops = list(candidates)
        self._rng.shuffle(ops)
        limit = min(len(ops), self._max_operator_attempts)
        for _ in range(limit):
            op = ops.pop()
            node = self._try_operator(op, depth)
            if node is not None:
                return node
        return None

    def _try_operator(
        self, op: OperatorDef, depth: int
    ) -> ASTNode | None:
        if self._nodes_left < 1:
            return None
        self._nodes_left -= 1  # reserve a slot for the call itself

        children: list[ASTNode] = []
        for position_types in op.input_types:
            child_type = self._pick_position_type(position_types)
            if child_type is None:
                self._nodes_left += 1  # refund
                return None
            child = self._build(child_type, depth - 1)
            if child is None:
                self._nodes_left += 1
                return None
            children.append(child)

        params = tuple(
            _sample_param(dom, self._rng) for _, dom in op.param_domains
        )
        try:
            return ASTNode.call(op, tuple(children), params)
        except ValueError:
            # Type or param validation failed: refund and give up on this
            # operator. The caller will try another candidate.
            self._nodes_left += 1
            return None

    def _pick_position_type(
        self, expected: frozenset[DataType]
    ) -> DataType | None:
        """Pick a concrete DataType from a union position, preferring types
        the pool can actually populate. SCALAR is always available because
        a literal can always be constructed."""
        if not expected:
            return None
        available: list[DataType] = []
        for dt in expected:
            if dt is DataType.SCALAR:
                available.append(dt)
            elif self._pool.fields.by_type(dt):
                available.append(dt)
        if not available:
            return None
        return self._rng.choice(available)