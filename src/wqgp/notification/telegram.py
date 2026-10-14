"""Telegram Notifier.

Design decisions:
- Sends over HttpTransport, not requests. See D-071, D-094.
- Credentials come from environment variables only. See D-089, D-097.
- Every method swallows its own exceptions. See D-095.
- Messages are plain text; no Markdown or HTML parse mode.
- Logs go to the "wqgp.notification" channel. Send attempts are at
  DEBUG; failures are at WARNING. See D-134.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from wqgp.brain.transport import HttpTransport, TransportError
from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)


log = logging.getLogger("wqgp.notification")


_API_BASE = "https://api.telegram.org"


class TelegramConfigError(RuntimeError):
    """Raised on missing or invalid Telegram configuration."""


@dataclass(frozen=True, slots=True)
class TelegramCredentials:
    """Bot token and chat id."""
    bot_token: str
    chat_id: str

    def __post_init__(self) -> None:
        if not self.bot_token or not self.bot_token.strip():
            raise TelegramConfigError("bot_token must be non-empty")
        if not self.chat_id or not self.chat_id.strip():
            raise TelegramConfigError("chat_id must be non-empty")

    @classmethod
    def from_env(cls) -> "TelegramCredentials":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat:
            raise TelegramConfigError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set"
            )
        return cls(bot_token=token, chat_id=chat)


class TelegramNotifier:
    """Sends run lifecycle events to a Telegram chat."""

    __slots__ = ("_credentials", "_transport", "_http_timeout")

    def __init__(
        self,
        credentials: TelegramCredentials,
        transport: HttpTransport,
        *,
        http_timeout: float = 10.0,
    ) -> None:
        if http_timeout <= 0:
            raise ValueError("http_timeout must be > 0")
        self._credentials = credentials
        self._transport = transport
        self._http_timeout = http_timeout

    def close(self) -> None:
        self._transport.close()

    # --- Notifier Protocol --------------------------------------------

    def on_run_start(self, event: RunStartEvent) -> None:
        verb = "resuming" if event.resuming else "starting"
        self._send(
            f"[wqgp] {verb} run {event.run_id} "
            f"at generation {event.generation} "
            f"with population {event.population_size}"
        )

    def on_generation_end(self, event: GenerationEndEvent) -> None:
        best = (
            "n/a" if event.best_fitness is None
            else f"{event.best_fitness:.4f}"
        )
        suffix = (
            "" if event.stopped_reason is None
            else f" ({event.stopped_reason})"
        )
        self._send(
            f"[wqgp] {event.run_id} gen {event.generation}: "
            f"evaluated={event.evaluated_count} "
            f"best={best} "
            f"sims={event.total_simulations_used}{suffix}"
        )

    def on_run_end(self, event: RunEndEvent) -> None:
        self._send(
            f"[wqgp] {event.run_id} finished: "
            f"reason={event.stopped_reason} "
            f"generation={event.generation} "
            f"sims={event.total_simulations_used} "
            f"population={event.population_size} "
            f"budget_exhausted={event.budget_exhausted}"
        )

    # --- internals -----------------------------------------------------

    def _send(self, text: str) -> None:
        """POST a message. Swallows every exception (D-095)."""
        url = (
            f"{_API_BASE}/bot{self._credentials.bot_token}/sendMessage"
        )
        body = {
            "chat_id": self._credentials.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        log.debug("[notif] sending: %s", text[:80])
        try:
            resp = self._transport.request(
                "POST",
                url,
                json_body=body,
                headers={"Content-Type": "application/json"},
                timeout=self._http_timeout,
            )
        except TransportError as e:
            log.warning("[notif] transport error: %s", e)
            return
        except Exception as e:  # noqa: BLE001 -- D-095
            log.warning("[notif] unexpected error: %s", e)
            return

        if resp.status_code >= 400:
            log.warning(
                "[notif] send failed with HTTP %d", resp.status_code
            )