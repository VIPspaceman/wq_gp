"""Selection operators for the GP loop.

Design decisions:
- Selection is single-objective (fitness only). See D-059.
- Tournament sampling is without replacement. See D-060.
- Elites are carried over unchanged; they are not mutated. See D-061.
- next_generation is best-effort on size, hard-fail only if empty. See D-062.
- Crossover requires two parents with distinct fingerprints. See D-063.

This module does NOT evaluate, does NOT talk to BRAIN, and does NOT
mutate its inputs. It reads a Population and returns a new Population.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from wqgp.crossover import Crossover, CrossoverError
from wqgp.mutation import MutationError, Mutator
from wqgp.population import Individual, Population


class SelectionError(RuntimeError):
    """Raised when selection cannot produce a new population."""


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Frozen parameters for one next_generation call.

    Invariants:
      - population_size >= 1
      - 0 <= elite_count <= population_size
      - tournament_size >= 1
      - 0.0 <= crossover_rate <= 1.0
      - max_breeding_attempts >= 1
    """
    population_size: int
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.7
    max_breeding_attempts: int = 200

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


# --- helpers ----------------------------------------------------------------

def _fitness_value(ind: Individual) -> float:
    """Fitness for ranking. Unevaluated individuals rank lowest."""
    return ind.fitness if ind.fitness is not None else float("-inf")


def _pick_best(candidates: list[Individual]) -> Individual:
    """Return the highest-fitness candidate; ties broken by fingerprint."""
    return max(
        candidates,
        key=lambda i: (_fitness_value(i), i.fingerprint),
    )


# --- public operators -------------------------------------------------------

def tournament_select(
    population: Population,
    k: int,
    rng: random.Random,
) -> Individual:
    """Sample min(k, len(population)) individuals WITHOUT replacement and
    return the one with the highest fitness.

    Raises SelectionError if the population is empty.
    Raises ValueError if k < 1.
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
) -> tuple[Individual, ...]:
    """Return the top-n evaluated individuals by fitness, ties broken by
    fingerprint. If fewer than n are evaluated, return what is available.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return ()
    evaluated = [i for i in population if i.fitness is not None]
    evaluated.sort(key=lambda i: (-i.fitness, i.fingerprint))  # type: ignore[operator]
    return tuple(evaluated[:n])


# --- breeding ---------------------------------------------------------------

def _breed_one(
    parents: Population,
    mutator: Mutator,
    crossover: Crossover,
    config: SelectionConfig,
    rng: random.Random,
) -> Individual | None:
    """Produce one offspring via crossover or mutation.

    Returns None if the chosen path failed (e.g. no compatible swap,
    no mutation available). The caller retries.
    """
    parent_a = tournament_select(parents, config.tournament_size, rng)

    use_crossover = (
        rng.random() < config.crossover_rate and len(parents) >= 2
    )
    parent_b: Individual | None = None
    if use_crossover:
        for _ in range(5):
            candidate = tournament_select(
                parents, config.tournament_size, rng
            )
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


def next_generation(
    parents: Population,
    mutator: Mutator,
    crossover: Crossover,
    config: SelectionConfig,
    rng: random.Random,
) -> Population:
    """Build a new Population from `parents`.

    Flow:
      1. Copy `config.elite_count` top individuals unchanged.
      2. Fill the rest with offspring from mutation or crossover.
      3. Deduplicate by fingerprint (first occurrence wins).
      4. Stop at `config.population_size` or when the breeding-attempt
         budget is exhausted.

    Raises SelectionError if `parents` is empty, or if no individual at
    all could be produced (not even an elite).
    """
    if len(parents) == 0:
        raise SelectionError("cannot evolve an empty population")

    elites = select_elites(parents, config.elite_count)
    result: list[Individual] = list(elites)
    seen: set[str] = {i.fingerprint for i in elites}

    attempts = 0
    while (
        len(result) < config.population_size
        and attempts < config.max_breeding_attempts
    ):
        attempts += 1
        child = _breed_one(parents, mutator, crossover, config, rng)
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