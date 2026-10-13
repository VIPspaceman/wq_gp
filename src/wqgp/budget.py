"""Layered budget tracking for a GP run.

Design decisions vs. the reference project (VIPspaceman/wq):
- Budget is layered: total / per-generation / wall-clock. See D-013.
- Consumption is all-or-nothing: a failed check leaves state untouched.
- Exhaustion raises `BudgetExhausted`; it is NOT a soft signal. See D-013.
- `BudgetSnapshot` is the frozen, JSON-friendly view used by checkpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    """Raised when a budget consumption would exceed a hard limit."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Immutable budget limits for one run.

    - total_simulations: hard cap across the whole run (required).
    - per_generation_simulations: soft cap reset by `start_generation()`.
      None means unlimited per generation.
    - wall_time_seconds: hard cap on elapsed time. None means unlimited.
    """
    total_simulations: int
    per_generation_simulations: int | None = None
    wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.total_simulations < 0:
            raise ValueError("total_simulations must be >= 0")
        if (
            self.per_generation_simulations is not None
            and self.per_generation_simulations < 0
        ):
            raise ValueError("per_generation_simulations must be >= 0")
        if self.wall_time_seconds is not None and self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be >= 0")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Frozen, JSON-friendly view of a tracker's state."""
    total_simulations_limit: int
    per_generation_limit: int | None
    wall_time_limit_seconds: float | None
    total_simulations_used: int
    generation_simulations_used: int
    elapsed_seconds: float
    frozen: bool


class BudgetTracker:
    """Mutable budget state machine.

    Invariants:
      - total_simulations_used <= budget.total_simulations
      - generation_simulations_used <= per_generation_simulations (if set)
      - once `freeze()` is called, every consume raises.
    """

    __slots__ = (
        "_budget",
        "_total_used",
        "_generation_used",
        "_started_at",
        "_frozen",
    )

    def __init__(
        self,
        budget: Budget,
        *,
        elapsed_offset_seconds: float = 0.0,
        total_simulations_used: int = 0,
        generation_simulations_used: int = 0,
    ) -> None:
        if elapsed_offset_seconds < 0:
            raise ValueError("elapsed_offset_seconds must be >= 0")
        if not 0 <= total_simulations_used <= budget.total_simulations:
            raise ValueError(
                f"total_simulations_used={total_simulations_used} "
                f"outside [0, {budget.total_simulations}]"
            )
        if budget.per_generation_simulations is not None:
            if not 0 <= generation_simulations_used <= budget.per_generation_simulations:
                raise ValueError(
                    f"generation_simulations_used={generation_simulations_used} "
                    f"outside [0, {budget.per_generation_simulations}]"
                )
        elif generation_simulations_used < 0:
            raise ValueError("generation_simulations_used must be >= 0")

        self._budget = budget
        self._total_used = total_simulations_used
        self._generation_used = generation_simulations_used
        # monotonic is immune to wall-clock changes, but resets across
        # process restarts -- hence the offset parameter.
        self._started_at = time.monotonic() - elapsed_offset_seconds
        self._frozen = False

    # --- properties ---

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def total_simulations_used(self) -> int:
        return self._total_used

    @property
    def generation_simulations_used(self) -> int:
        return self._generation_used

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def remaining_total_simulations(self) -> int:
        return self._budget.total_simulations - self._total_used

    def remaining_generation_simulations(self) -> int | None:
        if self._budget.per_generation_simulations is None:
            return None
        return (
            self._budget.per_generation_simulations - self._generation_used
        )

    def remaining_wall_time_seconds(self) -> float | None:
        if self._budget.wall_time_seconds is None:
            return None
        return max(0.0, self._budget.wall_time_seconds - self.elapsed_seconds())

    # --- mutation ---

    def start_generation(self) -> None:
        """Reset the per-generation counter. Called at each generation boundary."""
        if self._frozen:
            raise BudgetExhausted("tracker is frozen")
        self._generation_used = 0

    def freeze(self) -> None:
        """Irreversibly stop all consumption. Used at run end."""
        self._frozen = True

    def consume_simulation(self, n: int = 1) -> None:
        """Atomically consume `n` simulations, or raise.

        All checks happen before any state mutation: a failed call leaves
        the tracker unchanged.
        """
        if self._frozen:
            raise BudgetExhausted("tracker is frozen")
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return

        # wall clock first: if time is up, nothing else matters
        if self._budget.wall_time_seconds is not None:
            if self.elapsed_seconds() >= self._budget.wall_time_seconds:
                raise BudgetExhausted(
                    f"wall time exhausted: "
                    f"{self.elapsed_seconds():.1f}s >= "
                    f"{self._budget.wall_time_seconds:.1f}s"
                )

        if self._total_used + n > self._budget.total_simulations:
            raise BudgetExhausted(
                f"total simulations: used {self._total_used} + {n} > "
                f"limit {self._budget.total_simulations}"
            )
        per_gen = self._budget.per_generation_simulations
        if per_gen is not None and self._generation_used + n > per_gen:
            raise BudgetExhausted(
                f"per-generation simulations: used {self._generation_used} "
                f"+ {n} > limit {per_gen}"
            )

        self._total_used += n
        self._generation_used += n

    # --- views ---

    def is_exhausted(self) -> bool:
        if self._total_used >= self._budget.total_simulations:
            return True
        if self._budget.wall_time_seconds is not None:
            if self.elapsed_seconds() >= self._budget.wall_time_seconds:
                return True
        return False

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total_simulations_limit=self._budget.total_simulations,
            per_generation_limit=self._budget.per_generation_simulations,
            wall_time_limit_seconds=self._budget.wall_time_seconds,
            total_simulations_used=self._total_used,
            generation_simulations_used=self._generation_used,
            elapsed_seconds=self.elapsed_seconds(),
            frozen=self._frozen,
        )