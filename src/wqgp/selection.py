"""Selection operators for the GP loop.

Design decisions:
- Selection defaults to single-objective (fitness only). See D-059.
- Tournament sampling is without replacement. See D-060.
- Crossover requires two parents with distinct fingerprints. See D-063.
- next_generation is best-effort on size. See D-062.
- mode="pareto" dispatches to Pareto selection. See D-126.
- mode="single" (default) uses fitness sharing when
  diversity_enabled=True. See D-149.
- Similarity is a weighted blend of field-Jaccard and operator-Jaccard.
  See D-150.
- Fitness sharing uses continuous similarity kernel + rank utility +
  elite protection. See D-151.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from wqgp.ast import iter_field_refs, iter_nodes
from wqgp.crossover import Crossover, CrossoverError
from wqgp.mutation import MutationError, Mutator
from wqgp.population import Individual, Population


# --- errors -----------------------------------------------------------------

class SelectionError(RuntimeError):
    """Raised when selection cannot produce a new population."""


# --- fitness sharing constants ----------------------------------------------

# Weight of the field-set Jaccard in the similarity blend. See D-150.
_FIELD_WEIGHT = 0.7

# Weight of the operator-set Jaccard in the similarity blend.
_OPERATOR_WEIGHT = 0.3

# Similarity values at or below this contribute nothing to crowding.
_SIMILARITY_FLOOR = 0.4

# elite protection reserves one slot for the raw best individual.
# See D-151.


def _extract_field_set(ind: Individual) -> frozenset[str]:
    """Field ids appearing anywhere in the individual's AST."""
    return frozenset(
        node.field_def.field_id
        for _, node in iter_field_refs(ind.ast)
        if node.field_def is not None
    )


def _extract_operator_set(ind: Individual) -> frozenset[str]:
    """Operator names appearing anywhere in the individual's AST."""
    return frozenset(
        node.op_def.name
        for _, node in iter_nodes(ind.ast)
        if node.op_def is not None
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity. Empty unions map to 1.0 (two empties are
    identical)."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _similarity(
    fields_a: frozenset[str],
    ops_a: frozenset[str],
    fields_b: frozenset[str],
    ops_b: frozenset[str],
) -> float:
    """Weighted blend of field-Jaccard and operator-Jaccard."""
    field_j = _jaccard(fields_a, fields_b)
    op_j = _jaccard(ops_a, ops_b)
    return _FIELD_WEIGHT * field_j + _OPERATOR_WEIGHT * op_j


def _share_contribution(sim: float) -> float:
    """Continuous kernel: 0 below floor, linear to 1 at sim=1.

    Chosen over a hard threshold to avoid the cliff effect where a
    similarity of 0.49 contributes nothing and 0.51 contributes 0.5.
    See D-151.
    """
    if sim <= _SIMILARITY_FLOOR:
        return 0.0
    return (sim - _SIMILARITY_FLOOR) / (1.0 - _SIMILARITY_FLOOR)


def compute_adjusted_fitness(
    pop: Population,
) -> dict[str, float]:
    """Return {fingerprint: adjusted_fitness} for every individual.

    Algorithm (D-151):
      1. Cache field and operator sets per individual.
      2. O(n^2) similarity matrix.
      3. crowding(i) = 1 + sum(share(sim(i, j)) for j != i)
      4. utility(i) = ((N - r_i + 1) / N) ** 2, where r_i is the rank
         by raw fitness descending (rank 1 = best).
      5. adjusted(i) = utility(i) / crowding(i)
    """
    members = list(pop)
    n = len(members)
    if n == 0:
        return {}

    fields = [_extract_field_set(m) for m in members]
    ops = [_extract_operator_set(m) for m in members]

    crowd = [1.0] * n
    for i in range(n):
        acc = 0.0
        fi, oi = fields[i], ops[i]
        for j in range(n):
            if i == j:
                continue
            sim = _similarity(fi, oi, fields[j], ops[j])
            acc += _share_contribution(sim)
        crowd[i] = 1.0 + acc

    # Rank by raw fitness descending, ties broken by fingerprint for
    # determinism.
    def raw_key(idx: int) -> tuple[float, str]:
        f = members[idx].fitness
        fv = f if f is not None else float("-inf")
        return (-fv, members[idx].fingerprint)

    order = sorted(range(n), key=raw_key)

    adjusted: dict[str, float] = {}
    for rank, idx in enumerate(order, start=1):
        utility = ((n - rank + 1) / n) ** 2
        adjusted[members[idx].fingerprint] = utility / crowd[idx]
    return adjusted


# --- config -----------------------------------------------------------------

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


# --- helpers ----------------------------------------------------------------

def _fitness_value(ind: Individual) -> float:
    return ind.fitness if ind.fitness is not None else float("-inf")


def _pick_best(candidates: list[Individual]) -> Individual:
    """Return the highest-fitness candidate; ties by fingerprint."""
    return max(
        candidates,
        key=lambda i: (_fitness_value(i), i.fingerprint),
    )


# --- public operators (single-objective) ------------------------------------

def tournament_select(
    population: Population,
    k: int,
    rng: random.Random,
) -> Individual:
    """Sample min(k, len) individuals WITHOUT replacement and return
    the one with the highest fitness. Ties broken by fingerprint.
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
    """Top-n evaluated individuals by raw fitness. Ties by fingerprint.
    Returns fewer than n if fewer are evaluated.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return ()
    evaluated = [i for i in population if i.fitness is not None]
    evaluated.sort(key=lambda i: (-i.fitness, i.fingerprint))  # type: ignore[operator]
    return tuple(evaluated[:n])


def _select_elites_with_diversity(
    population: Population,
    n: int,
    adjusted: dict[str, float],
) -> tuple[Individual, ...]:
    """Elite selection with diversity.

    - One slot is reserved for the raw-fitness best evaluated
      individual (the protection rule; see D-151).
    - Remaining slots are filled by adjusted fitness descending.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return ()

    evaluated = [i for i in population if i.fitness is not None]
    if not evaluated:
        return ()

    # Slot 1: the raw best.
    raw_best = max(
        evaluated,
        key=lambda i: (i.fitness, i.fingerprint),  # type: ignore[arg-type,return-value]
    )
    if n == 1:
        return (raw_best,)

    remaining = [i for i in evaluated if i is not raw_best]
    remaining.sort(
        key=lambda i: (
            -adjusted.get(i.fingerprint, 0.0),
            i.fingerprint,
        )
    )
    return (raw_best, *remaining[: n - 1])


def _tournament_select_with_diversity(
    population: Population,
    k: int,
    rng: random.Random,
    adjusted: dict[str, float],
) -> Individual:
    """Tournament selection using adjusted fitness instead of raw."""
    if len(population) == 0:
        raise SelectionError("cannot select from an empty population")
    if k < 1:
        raise ValueError("tournament size must be >= 1")
    members = list(population)
    sample_size = min(k, len(members))
    contestants = rng.sample(members, sample_size)
    return max(
        contestants,
        key=lambda i: (
            adjusted.get(i.fingerprint, 0.0),
            i.fingerprint,
        ),
    )


# --- breeding ---------------------------------------------------------------

TournamentFn = Callable[[], Individual]


def _breed_one(
    parents: Population,
    mutator: Mutator,
    crossover: Crossover,
    config: SelectionConfig,
    rng: random.Random,
    tournament_fn: TournamentFn,
) -> Individual | None:
    """Produce one offspring via crossover or mutation.

    `tournament_fn` is supplied by the caller so the same breeding
    loop serves single-objective, diversity-aware, and Pareto paths.
    """
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

    if config.mode == "single" and config.diversity_enabled:
        adjusted = compute_adjusted_fitness(parents)
        elites = _select_elites_with_diversity(
            parents, config.elite_count, adjusted,
        )

        def tournament_fn() -> Individual:
            return _tournament_select_with_diversity(
                parents, config.tournament_size, rng, adjusted,
            )

    elif config.mode == "single":
        elites = select_elites(parents, config.elite_count)

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