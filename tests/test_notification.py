"""Tests for notification adapters. Phase 4 Step 3 gate."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from tests._fakes import ScriptedTransport
from wqgp.brain.transport import TransportError, make_response
from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)
from wqgp.notification.null import NullNotifier
from wqgp.notification.telegram import (
    TelegramConfigError,
    TelegramCredentials,
    TelegramNotifier,
)


# --- fixtures ---------------------------------------------------------------

def _creds() -> TelegramCredentials:
    return TelegramCredentials(
        bot_token="123456:ABCDEF", chat_id="987654",
    )


def _sent_message(t: ScriptedTransport) -> dict[str, Any]:
    """Return the JSON body of the last POST that was attempted."""
    for method, url, meta in reversed(t.calls):
        if method == "POST" and "/sendMessage" in url:
            return meta["json_body"]
    raise AssertionError("no sendMessage call recorded")


def _ok() -> Any:
    return make_response(200, json_body={"ok": True})


# --- TelegramCredentials ---------------------------------------------------

def test_credentials_reject_empty_token() -> None:
    with pytest.raises(TelegramConfigError):
        TelegramCredentials(bot_token="", chat_id="1")


def test_credentials_reject_empty_chat_id() -> None:
    with pytest.raises(TelegramConfigError):
        TelegramCredentials(bot_token="t", chat_id="")


def test_credentials_reject_whitespace_token() -> None:
    with pytest.raises(TelegramConfigError):
        TelegramCredentials(bot_token="   ", chat_id="1")


def test_credentials_from_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(TelegramConfigError):
        TelegramCredentials.from_env()


def test_credentials_from_env_ok(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    c = TelegramCredentials.from_env()
    assert c.bot_token == "tok"
    assert c.chat_id == "42"


# --- NullNotifier -----------------------------------------------------------

def test_null_notifier_accepts_every_event() -> None:
    n = NullNotifier()
    n.on_run_start(RunStartEvent(
        run_id="r", generation=0, population_size=1, resuming=False,
    ))
    n.on_generation_end(GenerationEndEvent(
        run_id="r", generation=0, evaluated_count=1,
        best_fitness=1.0, total_simulations_used=1,
        stopped_reason=None,
    ))
    n.on_run_end(RunEndEvent(
        run_id="r", generation=0, stopped_reason="max_generations",
        total_simulations_used=1, budget_exhausted=False,
        population_size=1,
    ))


# --- TelegramNotifier: happy paths ------------------------------------------

def test_on_run_start_sends_starting() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_run_start(RunStartEvent(
        run_id="alpha", generation=0, population_size=10, resuming=False,
    ))
    body = _sent_message(t)
    assert body["chat_id"] == "987654"
    assert "starting" in body["text"]
    assert "alpha" in body["text"]
    assert body["disable_web_page_preview"] is True


def test_on_run_start_sends_resuming() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_run_start(RunStartEvent(
        run_id="alpha", generation=3, population_size=5, resuming=True,
    ))
    body = _sent_message(t)
    assert "resuming" in body["text"]
    assert "3" in body["text"]


def test_on_generation_end_includes_best_fitness() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_generation_end(GenerationEndEvent(
        run_id="alpha", generation=2, evaluated_count=15,
        best_fitness=2.5, total_simulations_used=42,
        stopped_reason=None,
    ))
    body = _sent_message(t)
    assert "gen 2" in body["text"]
    assert "2.5" in body["text"]
    assert "42" in body["text"]


def test_on_generation_end_without_fitness() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_generation_end(GenerationEndEvent(
        run_id="alpha", generation=0, evaluated_count=0,
        best_fitness=None, total_simulations_used=0,
        stopped_reason=None,
    ))
    body = _sent_message(t)
    assert "n/a" in body["text"]


def test_on_generation_end_with_stopped_reason() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_generation_end(GenerationEndEvent(
        run_id="alpha", generation=4, evaluated_count=5,
        best_fitness=1.0, total_simulations_used=100,
        stopped_reason="budget_exhausted",
    ))
    body = _sent_message(t)
    assert "budget_exhausted" in body["text"]


def test_on_run_end_message_shape() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_run_end(RunEndEvent(
        run_id="alpha", generation=9,
        stopped_reason="max_generations",
        total_simulations_used=500,
        budget_exhausted=False,
        population_size=20,
    ))
    body = _sent_message(t)
    assert "alpha" in body["text"]
    assert "max_generations" in body["text"]
    assert "500" in body["text"]


# --- TelegramNotifier: URL construction -------------------------------------

def test_send_uses_bot_token_in_url() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", _ok())
    n = TelegramNotifier(_creds(), t)
    n.on_run_start(RunStartEvent(
        run_id="r", generation=0, population_size=1, resuming=False,
    ))
    urls = [url for _, url, _ in t.calls]
    assert any(
        "123456:ABCDEF" in u and "/sendMessage" in u for u in urls
    )


# --- TelegramNotifier: error handling (D-095) -------------------------------

def test_transport_error_is_swallowed(caplog) -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", TransportError("dns"))
    n = TelegramNotifier(_creds(), t)
    # Must NOT raise.
    with caplog.at_level(logging.WARNING, logger="wqgp.notification.telegram"):
        n.on_run_start(RunStartEvent(
            run_id="r", generation=0, population_size=1, resuming=False,
        ))
    assert any("transport error" in r.message for r in caplog.records)


def test_http_4xx_is_swallowed(caplog) -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", make_response(400, body=b"bad"))
    n = TelegramNotifier(_creds(), t)
    with caplog.at_level(logging.WARNING, logger="wqgp.notification.telegram"):
        n.on_run_start(RunStartEvent(
            run_id="r", generation=0, population_size=1, resuming=False,
        ))
    assert any("HTTP 400" in r.message for r in caplog.records)


def test_http_5xx_is_swallowed(caplog) -> None:
    t = ScriptedTransport()
    t.expect("POST", "/sendMessage", make_response(500))
    n = TelegramNotifier(_creds(), t)
    with caplog.at_level(logging.WARNING, logger="wqgp.notification.telegram"):
        n.on_run_end(RunEndEvent(
            run_id="r", generation=0, stopped_reason="x",
            total_simulations_used=0, budget_exhausted=False,
            population_size=0,
        ))
    # Does not raise; a warning was recorded.
    assert caplog.records


def test_unexpected_error_is_swallowed(caplog) -> None:
    class BoomTransport:
        def request(self, *a, **k):
            raise RuntimeError("surprise")
        def close(self):
            pass

    n = TelegramNotifier(_creds(), BoomTransport())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="wqgp.notification.telegram"):
        n.on_run_start(RunStartEvent(
            run_id="r", generation=0, population_size=1, resuming=False,
        ))
    assert any("unexpected" in r.message for r in caplog.records)


# --- construction validation -----------------------------------------------

def test_notifier_rejects_bad_http_timeout() -> None:
    with pytest.raises(ValueError):
        TelegramNotifier(_creds(), ScriptedTransport(), http_timeout=0)
    with pytest.raises(ValueError):
        TelegramNotifier(_creds(), ScriptedTransport(), http_timeout=-1)


def test_close_is_idempotent() -> None:
    n = TelegramNotifier(_creds(), ScriptedTransport())
    n.close()
    n.close()  # must not raise