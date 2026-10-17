"""Individuals and populations for the GP loop.

Design decisions:
- Individual equality is by fingerprint only. See D-039.
- Population is a plain class, not a frozen dataclass, so it can cache
  its fingerprint index. See D-040.
- RandomAstBuilder is stateful and single-use per build. See D-041.
- AST generation is bottom-up with three hard limits. See D-042.
- Generation failure raises; it does not return None. See D-043.
- Only SCALAR leaves can be literals. See D-044.
- Individual.with_status propagates terminal states. See D-125.
- RandomAstBuilder takes an optional field_usage map for weighted
  field sampling. See D-165.
"""
from __future__ import annotations

import math
import random
from types import MappingProxyType
from typing import Mapping

from wqgp.ast import ASTNode
from wqgp.operator import OperatorDef
from wqgp.persistence import EvaluationStatus
from wqgp.pool import Pool
from wqgp.types import DataType, ParamDomain, ParamKind


class PopulationError(RuntimeError):
    """Base class for population-level failures."""


class DuplicateIndividualError(PopulationError):
    """Raised when two individuals share a fingerprint in one population."""


class GenerationError(PopulationError):
    """Raised when RandomAstBuilder cannot construct a valid AST."""


def _is_finite_number(v: object) -> bool:
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return True


class Individual:
    """One candidate alpha expression plus its evaluation state."""

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

        if fitness is not None and not _is_finite_number(fitness):
            raise ValueError(
                f"fitness must be a finite number, got {fitness!r}"
            )

        for k, v in metrics:
            if not isinstance(k, str) or not k:
                raise ValueError("metric name must be a non-empty string")
            if not _is_finite_number(v):
                raise ValueError(
                    f"metric {k!r} must be a finite number, got {v!r}"
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
        return Individual(
            ast=self.ast,
            fingerprint=self.fingerprint,
            fitness=fitness,
            metrics=metrics,
            generation=self.generation,
            parents=self.parents,
            status=EvaluationStatus.SUCCESS,
        )

    def with_status(self, status: EvaluationStatus) -> "Individual":
        return Individual(
            ast=self.ast,
            fingerprint=self.fingerprint,
            fitness=self.fitness,
            metrics=self.metrics,
            generation=self.generation,
            parents=self.parents,
            status=status,
        )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError(
                f"Individual is immutable; cannot reassign {name!r}"
            )
        object.__setattr__(self, name, value)

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


class Population:
    """A collection of Individuals with unique fingerprints."""

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

    @classmethod
    def empty(cls) -> "Population":
        return cls(())

    @classmethod
    def concat(cls, *populations: "Population") -> "Population":
        seen: dict[str, Individual] = {}
        for pop in populations:
            for ind in pop:
                seen.setdefault(ind.fingerprint, ind)
        return cls(tuple(seen.values()))

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


def _sample_param(
    domain: ParamDomain, rng: random.Random
) -> int | float | str:
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


_LITERAL_POOL: tuple[int | float, ...] = (1, 5, 10, 0.5, 1.0, -1)


# Field sampling weight bounds for D-165. Clamp keeps any field
# selectable even if it is currently overrepresented (prevents "hard
# ban"). The exponent 0.5 is a mild square-root penalty.
_FIELD_WEIGHT_ALPHA = 0.5
_FIELD_WEIGHT_MIN = 0.25
_FIELD_WEIGHT_MAX = 4.0


class RandomAstBuilder:
    """Builds random, type-valid ASTs from a Pool.

    Stateful and single-use per build. Not thread-safe.

    The optional `field_usage` map changes field sampling from uniform
    to inverse-frequency weighted. The key is a field_id and the value
    is the number of individuals in the *current population* that use
    that field. See D-165.
    """

    __slots__ = (
        "_pool",
        "_max_depth",
        "_max_nodes",
        "_max_operator_attempts",
        "_nodes_left",
        "_rng",
        "_field_usage",
    )

    def __init__(
        self,
        pool: Pool,
        *,
        max_depth: int = 4,
        max_nodes: int = 25,
        max_operator_attempts: int = 15,
        field_usage: Mapping[str, int] | None = None,
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
        self._rng: random.Random = random.Random()
        self._field_usage: Mapping[str, int] = field_usage or {}

    def build(self, target_type: DataType, rng: random.Random) -> ASTNode:
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

    def _pick_field(self, fields: tuple) -> object:
        """Sample one field from `fields` with inverse-frequency weight.

        When `_field_usage` is empty (initial generation, or the flag
        is off), falls back to uniform choice. See D-165.
        """
        if not self._field_usage:
            return self._rng.choice(fields)

        weights: list[float] = []
        for f in fields:
            count = self._field_usage.get(f.field_id, 0)
            w = 1.0 / ((1.0 + count) ** _FIELD_WEIGHT_ALPHA)
            if w < _FIELD_WEIGHT_MIN:
                w = _FIELD_WEIGHT_MIN
            elif w > _FIELD_WEIGHT_MAX:
                w = _FIELD_WEIGHT_MAX
            weights.append(w)
        return self._rng.choices(fields, weights=weights, k=1)[0]

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
        if target_type is DataType.SCALAR:
            scalar_fields = self._pool.fields.by_type(DataType.SCALAR)
            if scalar_fields and self._rng.random() < 0.5:
                self._nodes_left -= 1
                return ASTNode.field_ref(self._pick_field(scalar_fields))
            return self._make_literal()

        fields = self._pool.fields.by_type(target_type)
        if not fields:
            return None
        self._nodes_left -= 1
        return ASTNode.field_ref(self._pick_field(fields))

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
        self._nodes_left -= 1

        children: list[ASTNode] = []
        for position_types in op.input_types:
            child_type = self._pick_position_type(position_types)
            if child_type is None:
                self._nodes_left += 1
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
            self._nodes_left += 1
            return None

    def _pick_position_type(
        self, expected: frozenset[DataType]
    ) -> DataType | None:
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