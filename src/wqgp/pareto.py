"""Multi-objective selection via Pareto non-dominated sorting.

Design decisions:
- Self-contained: this module does not import selection.py, so
  selection.py can dispatch to it without a cycle. See D-126.
- Objectives are explicit (name + direction). Maximize vs minimize is
  part of the objective definition, not a convention. See D-127.
- Individuals missing a metric are ranked worst on that objective.
  See D-128.
- Crowding distance uses the standard NSGA-II formulation. See D-129.
- Ranking is precomputed once per generation via `prepare()` and
  reused by every tournament draw in that generation. See D-130.

This module does NOT evaluate individuals. It reads the metrics they
already carry.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from wqgp.population import Individual, Population


# --- objective --------------------------------------------------------------

class ObjectiveDirection(str, Enum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


@dataclass(frozen=True, slots=True)
class Objective:
    """One optimization target.

    `metric` is the name as it appears in Individual.metrics. When the
    individual does not carry this metric, extraction falls back to
    `Individual.fitness` for "sharpe" (the default fitness function's
    source) and to a worst-case sentinel otherwise. See D-128.
    """
    metric: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE

    def __post_init__(self) -> None:
        if not self.metric or not self.metric.strip():
            raise ValueError("objective metric must be non-empty")
        if not isinstance(self.direction, ObjectiveDirection):
            raise ValueError(
                f"direction must be ObjectiveDirection, "
                f"got {type(self.direction).__name__}"
            )

    def extract(self, ind: Individual) -> float:
        """Return the value of this objective for `ind`."""
        for k, v in ind.metrics:
            if k == self.metric:
                return float(v)
        # Fallback for the default fitness source.
        if self.metric == "sharpe" and ind.fitness is not None:
            return float(ind.fitness)
        # Worst-case sentinel: push missing-metric individuals to the
        # back of their objective without a special case in the sort.
        if self.direction is ObjectiveDirection.MAXIMIZE:
            return -math.inf
        return math.inf


DEFAULT_OBJECTIVES: tuple[Objective, ...] = (
    Objective("sharpe", ObjectiveDirection.MAXIMIZE),
    Objective("turnover", ObjectiveDirection.MINIMIZE),
)


# --- config -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParetoConfig:
    """Frozen configuration for Pareto selection."""
    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ValueError("at least one objective is required")
        seen = set()
        for obj in self.objectives:
            if obj.metric in seen:
                raise ValueError(
                    f"duplicate objective metric: {obj.metric!r}"
                )
            seen.add(obj.metric)

    @classmethod
    def default(cls) -> "ParetoConfig":
        return cls(DEFAULT_OBJECTIVES)


# --- ranking result ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParetoRanking:
    """Frozen result of `ParetoSelector.prepare`.

    Keys are fingerprints. Every individual passed to `prepare` has an
    entry in both maps.
    """
    front_index: Mapping[str, int]
    crowding: Mapping[str, float]

    def front_of(self, fingerprint: str) -> int:
        return self.front_index.get(fingerprint, 1 << 30)

    def crowding_of(self, fingerprint: str) -> float:
        return self.crowding.get(fingerprint, 0.0)


# --- selector ---------------------------------------------------------------

class ParetoSelector:
    """NSGA-II style non-dominated sorting + crowding distance.

    Lifecycle:
        selector = ParetoSelector(ParetoConfig.default())
        ranking = selector.prepare(population)
        elites = selector.select_elites(population, ranking, n)
        parent = selector.tournament_select(
            population, ranking, k, rng,
        )

    `prepare` is called once per generation; every tournament draw in
    that generation reuses the same ranking. See D-130.
    """

    __slots__ = ("_config", "_objectives")

    def __init__(self, config: ParetoConfig) -> None:
        self._config = config
        self._objectives = config.objectives

    @property
    def config(self) -> ParetoConfig:
        return self._config

    # --- public ---------------------------------------------------------

    def prepare(self, population: Population) -> ParetoRanking:
        """Compute (front_index, crowding_distance) for every member."""
        individuals = list(population)
        if not individuals:
            return ParetoRanking(
                front_index=MappingProxyType({}),
                crowding=MappingProxyType({}),
            )

        fronts = self._non_dominated_sort(individuals)

        front_index: dict[str, int] = {}
        crowding: dict[str, float] = {}
        for fi, front in enumerate(fronts):
            distances = self._crowding_distance(front)
            for ind, d in zip(front, distances):
                front_index[ind.fingerprint] = fi
                crowding[ind.fingerprint] = d

        return ParetoRanking(
            front_index=MappingProxyType(front_index),
            crowding=MappingProxyType(crowding),
        )

    def select_elites(
        self,
        population: Population,
        ranking: ParetoRanking,
        n: int,
    ) -> tuple[Individual, ...]:
        """Return the top-n individuals by (front, crowding) order."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return ()

        # Group by front index, preserving front order.
        by_front: dict[int, list[Individual]] = {}
        for ind in population:
            fi = ranking.front_of(ind.fingerprint)
            by_front.setdefault(fi, []).append(ind)

        result: list[Individual] = []
        for fi in sorted(by_front):
            # Sort this front by crowding distance (desc), then
            # fingerprint (asc) for determinism.
            front_sorted = sorted(
                by_front[fi],
                key=lambda i: (
                    -ranking.crowding_of(i.fingerprint),
                    i.fingerprint,
                ),
            )
            take = min(n - len(result), len(front_sorted))
            result.extend(front_sorted[:take])
            if len(result) >= n:
                break

        return tuple(result)

    def tournament_select(
        self,
        population: Population,
        ranking: ParetoRanking,
        k: int,
        rng: random.Random,
    ) -> Individual:
        """Sample min(k, len) individuals, return the one that ranks best
        by (front_index asc, crowding desc, fingerprint asc)."""
        members = list(population)
        if not members:
            raise ValueError("cannot select from an empty population")
        if k < 1:
            raise ValueError("tournament size must be >= 1")
        sample_size = min(k, len(members))
        contestants = rng.sample(members, sample_size)
        return min(
            contestants,
            key=lambda i: (
                ranking.front_of(i.fingerprint),
                -ranking.crowding_of(i.fingerprint),
                i.fingerprint,
            ),
        )

    # --- internals ------------------------------------------------------

    def _non_dominated_sort(
        self, individuals: list[Individual]
    ) -> list[list[Individual]]:
        """Fast non-dominated sort. O(n^2 * m) where m is #objectives.

        Returns a list of fronts, fronts[0] being the non-dominated
        set, fronts[1] the set dominated only by fronts[0], etc.
        """
        n = len(individuals)
        dominated_by: list[list[int]] = [[] for _ in range(n)]
        domination_count = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(individuals[i], individuals[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(individuals[j], individuals[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        fronts: list[list[Individual]] = []
        current_indices = [
            i for i in range(n) if domination_count[i] == 0
        ]

        while current_indices:
            fronts.append([individuals[i] for i in current_indices])
            next_indices: list[int] = []
            for i in current_indices:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_indices.append(j)
            current_indices = next_indices

        return fronts

    def _dominates(self, a: Individual, b: Individual) -> bool:
        """True iff a dominates b: at least as good on every objective
        and strictly better on at least one."""
        strictly_better = False
        for obj in self._objectives:
            av = obj.extract(a)
            bv = obj.extract(b)
            if obj.direction is ObjectiveDirection.MAXIMIZE:
                if av < bv:
                    return False
                if av > bv:
                    strictly_better = True
            else:
                if av > bv:
                    return False
                if av < bv:
                    strictly_better = True
        return strictly_better

    def _crowding_distance(
        self, front: list[Individual]
    ) -> list[float]:
        """NSGA-II crowding distance within one front.

        Boundary individuals (best or worst on any objective) get
        infinite distance so they survive truncation. Interior
        individuals accumulate normalized neighbor gaps. See D-129.
        """
        n = len(front)
        if n == 0:
            return []
        if n <= 2:
            return [math.inf] * n

        distances = [0.0] * n

        for obj in self._objectives:
            values = [obj.extract(ind) for ind in front]
            # Sort ascending by value. Direction is irrelevant for
            # crowding; we only care how far apart neighbors are.
            order = sorted(range(n), key=lambda i: values[i])
            v_min = values[order[0]]
            v_max = values[order[-1]]
            rng = v_max - v_min

            distances[order[0]] = math.inf
            distances[order[-1]] = math.inf
            if rng <= 0:
                continue

            for k in range(1, n - 1):
                gap = values[order[k + 1]] - values[order[k - 1]]
                distances[order[k]] += gap / rng

        return distances