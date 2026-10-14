"""BRAIN simulation settings.

Design decisions:
- Moved out of brain/client.py so that EvaluationConfig can depend on
  it without a circular import when BrainClient starts accepting an
  EvaluationConfig in Phase 4 Step 8 Round 2. See D-138.
- Defaults (decay=4, neutralization="INDUSTRY") match values observed
  to be accepted by BRAIN's /simulations endpoint. See D-113.
- Validation is type-only: the allowed values for region / universe /
  neutralization change over time. See D-074.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BrainSettings:
    """Frozen simulation settings."""
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 4
    neutralization: str = "INDUSTRY"
    truncation: float = 0.08
    pasteurization: str = "ON"
    nan_handling: str = "OFF"
    unit_handling: str = "VERIFY"
    language: str = "FASTEXPR"
    visualization: bool = False
    test_period: str = "P0Y0M"

    def __post_init__(self) -> None:
        if not isinstance(self.delay, int) or isinstance(self.delay, bool):
            raise ValueError("delay must be an int")
        if self.delay < 0:
            raise ValueError("delay must be >= 0")
        if not isinstance(self.decay, int) or isinstance(self.decay, bool):
            raise ValueError("decay must be an int")
        if self.decay < 0:
            raise ValueError("decay must be >= 0")
        if not isinstance(self.truncation, (int, float)) or isinstance(
            self.truncation, bool
        ):
            raise ValueError("truncation must be a number")
        if not 0.0 <= float(self.truncation) <= 1.0:
            raise ValueError("truncation must be in [0.0, 1.0]")
        if not isinstance(self.visualization, bool):
            raise ValueError("visualization must be a bool")

    def to_api_dict(self) -> dict[str, Any]:
        """Render to the JSON body the BRAIN /simulations endpoint expects."""
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": float(self.truncation),
            "pasteurization": self.pasteurization,
            "nanHandling": self.nan_handling,
            "unitHandling": self.unit_handling,
            "language": self.language,
            "visualization": self.visualization,
            "testPeriod": self.test_period,
        }