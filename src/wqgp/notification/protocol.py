"""Notifier protocol.

Design decisions:
- Notifier is a typing.Protocol, structural typing. A caller can pass
  any object with the three methods, including a lambda-in-a-class
  for tests. See D-093.
- Every method returns None. Notifications are fire-and-forget; a
  return value would invite the coordinator to depend on the result,
  which would couple the GP loop to the notification transport.
- Every method must swallow its own exceptions. See D-095. The
  coordinator does not wrap notifier calls in try/except; the contract
  is that a notifier never raises.
"""
from __future__ import annotations

from typing import Protocol

from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)


class Notifier(Protocol):
    """Callback interface for run lifecycle events."""

    def on_run_start(self, event: RunStartEvent) -> None: ...

    def on_generation_end(self, event: GenerationEndEvent) -> None: ...

    def on_run_end(self, event: RunEndEvent) -> None: ...