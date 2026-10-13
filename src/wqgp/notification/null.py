"""A Notifier that does nothing.

This is the default. Every Coordinator without an explicit notifier
gets one of these, so the coordinator code path is uniform: there is
no `if self._notifier is not None` anywhere.
"""
from __future__ import annotations

from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)


class NullNotifier:
    """Discards every event."""

    __slots__ = ()

    def on_run_start(self, event: RunStartEvent) -> None:
        pass

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        pass

    def on_run_end(self, event: RunEndEvent) -> None:
        pass