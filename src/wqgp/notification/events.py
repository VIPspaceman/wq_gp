"""Event payloads passed to Notifier callbacks.

Design decisions:
- Frozen dataclasses; notifications are observations, not mutations.
  See D-096.
- Only the fields a notifier actually needs. The notifier is not a
  debugging dump; the checkpoint and evaluation store are for that.

No `format()` method on the events: rendering text is the notifier's
job, not the event's. A Telegram message and an email digest can share
the same event with different formatting.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunStartEvent:
    """Emitted once, before the first evaluation of the run."""
    run_id: str
    generation: int
    population_size: int
    resuming: bool


@dataclass(frozen=True, slots=True)
class GenerationEndEvent:
    """Emitted at the end of each generation, after evaluation and
    before `next_generation`."""
    run_id: str
    generation: int
    evaluated_count: int
    best_fitness: float | None
    total_simulations_used: int
    stopped_reason: str | None


@dataclass(frozen=True, slots=True)
class RunEndEvent:
    """Emitted once, after the run loop exits."""
    run_id: str
    generation: int
    stopped_reason: str
    total_simulations_used: int
    budget_exhausted: bool
    population_size: int