"""Selection operators for the GP loop.

Design decisions:
- Tournament uses raw fitness, sampled without replacement. See
  D-059, D-060.
- Elites carry over unchanged. See D-061.
- Crossover requires two parents with distinct fingerprints. See
  D-063.
- next_generation is best-effort on size. See D-062.
- mode="pareto" dispatches to Pareto selection. See D-126.
- mode="single" with diversity_enabled=True reserves one elite slot
  for a multi-field individual (A'). See D-166.
- The old fitness-sharing mechanism (D-149 / D-151) is removed: it
  was measured to make concentration worse, not better. See D-164
  reference and D-166.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from wqgp.ast_ops import count_distinct_fields, simplify_ast
from wqgp.crossover import Crossover, CrossoverError
from wqgp.mutation import MutationError, Mutator
from wqgp.population import Individual, Population


class SelectionError(RuntimeError):
    """Raised when selection cannot produce a new population."""


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Frozen parameters for one next_generation call."""
    population_size: int
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.7
    max_breeding_attempts: int = 200
    mode: str = "single"
    pareto_objectives: tuple[tuple[str, str], ...] = ()
    diversity_enabled: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.elite_count < 0:
            raise ValueError("elite_count must be >= 0")
        if self.elite_count > self.population_size:
            raise ValueError(
                f"elite_count ({self.elite_count}) must be <= "
                f"population_size ({self.population_size})"
            )
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(
                f"crossover_rate must be in [0.0, 1.0], "
                f"got {self.crossover_rate!r}"
            )
        if self.max_breeding_attempts < 1:
            raise ValueError("max_breeding_attempts must be >= 1")
        if self.mode not in ("single", "pareto"):
            raise ValueError(
                f"mode must be 'single' or 'pareto', got {self.mode!r}"
            )
        for entry in self.pareto_objectives:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
            ):
                raise ValueError(
                    f"pareto_objectives entries must be "
                    f"(metric, direction) string pairs, got {entry!r}"
                )


def _fitness_value(ind: Individual) -> float:
    return ind.fitness if ind.fitness is not None else float("-inf")


def _pick_best(candidates: list[Individual]) -> Individual:
    return max(
        candidates,
        key=lambda i: (_fitness_value(i), i.fingerprint),
    )


def tournament_select(
    population: Population,
    k: int,
    rng: random.Random,
) -> Individual:
    """Sample min(k, len) individuals without replacement; return the
    highest raw-fitness one.
    """
    if len(population) == 0:
        raise SelectionError("cannot select from an empty population")
    if k < 1:
        raise ValueError("tournament size must be >= 1")
    members = list(population)
    sample_size = min(k, len(members))
    contestants = rng.sample(members, sample_size)
    return _pick_best(contestants)


def select_elites(
    population: Population,
    n: int,
    *,
    reserve_multi_field: bool = False,
) -> tuple[Individual, ...]:
    """Top-n by raw fitness, ties broken by fingerprint.

    When `reserve_multi_field` is True and n >= 2 and at least one
    multi-field individual is available, one slot is reserved for the
    best such individual. The other slots follow raw fitness.

    "Multi-field" means `count_distinct_fields(simplify_ast(ast)) >= 2`,
    so a single-field expression that repeats the field in a
    semantically redundant wrapper does not qualify. See D-164 / D-166.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return ()

    evaluated = [i for i in population if i.fitness is not None]
    evaluated.sort(
        key=lambda i: (-i.fitness, i.fingerprint)  # type: ignore[operator]
    )

    if (
        not reserve_multi_field
        or n == 1
        or len(evaluated) < 2
    ):
        return tuple(evaluated[:n])

    raw_best = evaluated[0]
    remaining = evaluated[1:]
    multi_candidates = [
        i for i in remaining
        if count_distinct_fields(simplify_ast(i.ast)) >= 2
    ]
    result: list[Individual] = [raw_best]
    if multi_candidates:
        # remaining is already sorted by fitness; take the first
        # multi-field individual.
        multi_pick = multi_candidates[0]
        result.append(multi_pick)
        remaining = [i for i in remaining if i is not multi_pick]
    result.extend(remaining[: n - len(result)])
    return tuple(result)


TournamentFn = Callable[[], Individual]


def _breed_one(
    parents: Population,
    mutator: Mutator,
    crossover: Crossover,
    config: SelectionConfig,
    rng: random.Random,
    tournament_fn: TournamentFn,
) -> Individual | None:
    parent_a = tournament_fn()

    use_crossover = (
        rng.random() < config.crossover_rate and len(parents) >= 2
    )
    parent_b: Individual | None = None
    if use_crossover:
        for _ in range(5):
            candidate = tournament_fn()
            if candidate.fingerprint != parent_a.fingerprint:
                parent_b = candidate
                break
        if parent_b is None:
            use_crossover = False

    if use_crossover:
        assert parent_b is not None
        try:
            child_ast = crossover.crossover(
                parent_a.ast, parent_b.ast, rng
            )
        except CrossoverError:
            return None
        return Individual.from_ast(
            child_ast,
            generation=parent_a.generation + 1,
            parents=(parent_a.fingerprint, parent_b.fingerprint),
        )

    try:
        child_ast = mutator.mutate(parent_a.ast, rng)
    except MutationError:
        return None
    return Individual.from_ast(
        child_ast,
        generation=parent_a.generation + 1,
        parents=(parent_a.fingerprint,),
    )


def _resolve_pareto_config(config: SelectionConfig):
    from wqgp.pareto import (
        DEFAULT_OBJECTIVES,
        Objective,
        ObjectiveDirection,
        ParetoConfig,
    )

    if not config.pareto_objectives:
        return ParetoConfig(DEFAULT_OBJECTIVES)

    objectives: list[Objective] = []
    for metric, direction_str in config.pareto_objectives:
        try:
            direction = ObjectiveDirection(direction_str)
        except ValueError as e:
            raise SelectionError(
                f"invalid pareto direction {direction_str!r}; "
                f"expected MAXIMIZE or MINIMIZE"
            ) from e
        objectives.append(Objective(metric, direction))
    return ParetoConfig(tuple(objectives))


def next_generation(
    parents: Population,
    mutator: Mutator,
    crossover: Crossover,
    config: SelectionConfig,
    rng: random.Random,
) -> Population:
    """Build a new Population from `parents`."""
    if len(parents) == 0:
        raise SelectionError("cannot evolve an empty population")

    if config.mode == "single":
        elites = select_elites(
            parents,
            config.elite_count,
            reserve_multi_field=config.diversity_enabled,
        )

        def tournament_fn() -> Individual:
            return tournament_select(
                parents, config.tournament_size, rng,
            )

    elif config.mode == "pareto":
        from wqgp.pareto import ParetoSelector

        selector = ParetoSelector(_resolve_pareto_config(config))
        ranking = selector.prepare(parents)
        elites = selector.select_elites(
            parents, ranking, config.elite_count,
        )

        def tournament_fn() -> Individual:
            return selector.tournament_select(
                parents, ranking, config.tournament_size, rng,
            )

    else:
        raise SelectionError(f"unknown selection mode: {config.mode!r}")

    result: list[Individual] = list(elites)
    seen: set[str] = {i.fingerprint for i in elites}

    attempts = 0
    while (
        len(result) < config.population_size
        and attempts < config.max_breeding_attempts
    ):
        attempts += 1
        child = _breed_one(
            parents, mutator, crossover, config, rng, tournament_fn
        )
        if child is None:
            continue
        if child.fingerprint in seen:
            continue
        seen.add(child.fingerprint)
        result.append(child)

    if not result:
        raise SelectionError(
            f"no individuals produced after {attempts} attempts"
        )

    return Population(result)