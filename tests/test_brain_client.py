"""Tests for the BRAIN client. Phase 3 Step 6 gate.

Every test uses a ScriptedTransport; no real network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pytest

from wqgp.brain.client import (
    BrainAuthError,
    BrainClient,
    BrainClientError,
    BrainCredentials,
    BrainProtocolError,
    BrainSettings,
    SimulationResult,
    build_default_client,
    default_fitness,
)
from wqgp.brain.transport import HttpResponse, TransportError, make_response
from wqgp.execution import Task, TaskStatus


# --- ScriptedTransport ------------------------------------------------------

@dataclass
class ScriptedTransport:
    """Test double for HttpTransport.

    Register responses in FIFO order per (method, url-path-prefix). Every
    request pops the next queued response. Records every call.

    Prefixes are URL PATHS ("/authentication"), not full URLs. See D-079.
    """
    calls: list[tuple[str, str, dict[str, Any]]] = field(
        default_factory=list
    )
    _queues: dict[tuple[str, str], list[Any]] = field(default_factory=dict)

    def expect(self, method: str, url_prefix: str, *responses: Any) -> None:
        key = (method.upper(), url_prefix)
        self._queues.setdefault(key, []).extend(responses)

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        self.calls.append(
            (method.upper(), url, {
                "json_body": json_body,
                "headers": headers,
                "auth": auth,
            })
        )
        # Match against the URL's PATH, not the full URL. Registered
        # prefixes are paths ("/authentication"), while the client sends
        # absolute URLs ("https://api.../authentication"). See D-079.
        path = urlparse(url).path
        for (m, prefix), q in self._queues.items():
            if m == method.upper() and path.startswith(prefix) and q:
                item = q.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        raise AssertionError(
            f"no scripted response for {method} {url}; "
            f"queues: {list(self._queues)}"
        )

    def close(self) -> None:
        pass


# --- fixtures ---------------------------------------------------------------

def _creds() -> BrainCredentials:
    return BrainCredentials(username="u", password="p")


def _settings() -> BrainSettings:
    return BrainSettings()


def _task(expr: str = "ts_mean(close, 5)") -> Task:
    return Task(fingerprint="fp1", expression=expr, generation=0)


def _login_ok() -> HttpResponse:
    return make_response(201, json_body={"user": "u"})


def _submit_ok(sim_url: str = "/simulations/123") -> HttpResponse:
    return make_response(201, headers={"Location": sim_url})


def _sim_done(sharpe: float = 1.5, **extra: Any) -> HttpResponse:
    metrics = {"sharpe": sharpe, "turnover": 0.3, "returns": 0.05}
    metrics.update(extra)
    return make_response(
        200,
        json_body={
            "id": "alpha_1",
            "status": "COMPLETE",
            "is": metrics,
        },
    )


def _client(
    transport: ScriptedTransport,
    *,
    max_polls: int = 5,
    sleep_fn=None,
) -> BrainClient:
    return BrainClient(
        credentials=_creds(),
        settings=_settings(),
        transport=transport,
        sleep_fn=sleep_fn or (lambda _s: None),
        max_polls=max_polls,
        poll_timeout_seconds=100.0,
        base_poll_interval=0.001,
    )


# --- BrainCredentials -------------------------------------------------------

def test_credentials_reject_empty_username() -> None:
    with pytest.raises(ValueError):
        BrainCredentials(username="", password="p")


def test_credentials_reject_whitespace_username() -> None:
    with pytest.raises(ValueError):
        BrainCredentials(username="   ", password="p")


def test_credentials_reject_empty_password() -> None:
    with pytest.raises(ValueError):
        BrainCredentials(username="u", password="")


def test_credentials_from_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("WQ_BRAIN_USERNAME", raising=False)
    monkeypatch.delenv("WQ_BRAIN_PASSWORD", raising=False)
    with pytest.raises(BrainAuthError):
        BrainCredentials.from_env()


def test_credentials_from_env_ok(monkeypatch) -> None:
    monkeypatch.setenv("WQ_BRAIN_USERNAME", "u")
    monkeypatch.setenv("WQ_BRAIN_PASSWORD", "p")
    creds = BrainCredentials.from_env()
    assert creds.username == "u"
    assert creds.password == "p"


# --- BrainSettings ----------------------------------------------------------

def test_settings_rejects_negative_delay() -> None:
    with pytest.raises(ValueError):
        BrainSettings(delay=-1)


def test_settings_rejects_negative_decay() -> None:
    with pytest.raises(ValueError):
        BrainSettings(decay=-1)


def test_settings_rejects_out_of_range_truncation() -> None:
    with pytest.raises(ValueError):
        BrainSettings(truncation=-0.1)
    with pytest.raises(ValueError):
        BrainSettings(truncation=1.1)


def test_settings_rejects_bool_delay() -> None:
    with pytest.raises(ValueError):
        BrainSettings(delay=True)  # type: ignore[arg-type]


def test_settings_to_api_dict_keys() -> None:
    d = BrainSettings().to_api_dict()
    for k in (
        "instrumentType", "region", "universe", "delay", "decay",
        "neutralization", "truncation", "pasteurization",
        "nanHandling", "unitHandling", "language", "visualization",
        "testPeriod",
    ):
        assert k in d


# --- default_fitness --------------------------------------------------------

def test_default_fitness_returns_sharpe() -> None:
    assert default_fitness({"sharpe": 2.5}) == 2.5


def test_default_fitness_none_when_missing() -> None:
    assert default_fitness({}) is None


# --- happy path -------------------------------------------------------------

def test_call_submits_and_returns_success() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect("GET", "/simulations/123", _sim_done(sharpe=2.0))

    result = _client(t)(_task())

    assert result.status is TaskStatus.SUCCESS
    assert result.fitness == 2.0
    assert ("sharpe", 2.0) in result.metrics


def test_call_reuses_authentication() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations",
             _submit_ok("/simulations/1"),
             _submit_ok("/simulations/2"))
    t.expect("GET", "/simulations/1", _sim_done(1.0))
    t.expect("GET", "/simulations/2", _sim_done(2.0))

    c = _client(t)
    c(_task("a"))
    c(_task("b"))

    login_calls = [c for c in t.calls if c[1].endswith("/authentication")]
    assert len(login_calls) == 1


def test_call_sends_correct_body() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect("GET", "/simulations/123", _sim_done())

    _client(t)(_task("ts_mean(close, 5)"))

    submit = [c for c in t.calls if c[1].endswith("/simulations")]
    assert len(submit) == 1
    body = submit[0][2]["json_body"]
    assert body["type"] == "REGULAR"
    assert body["regular"] == "ts_mean(close, 5)"
    assert "settings" in body


# --- polling ---------------------------------------------------------------

def test_call_waits_when_poll_returns_202() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(202, headers={"Retry-After": "0.001"}),
        make_response(202, headers={"Retry-After": "0.001"}),
        _sim_done(3.0),
    )

    result = _client(t)(_task())
    assert result.status is TaskStatus.SUCCESS
    assert result.fitness == 3.0


def test_call_honors_retry_after() -> None:
    sleeps: list[float] = []
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(202, headers={"Retry-After": "0.5"}),
        _sim_done(),
    )

    c = BrainClient(
        credentials=_creds(),
        settings=_settings(),
        transport=t,
        sleep_fn=sleeps.append,
        max_polls=5,
        poll_timeout_seconds=100.0,
    )
    c(_task())
    assert sleeps == [0.5]


def test_call_exceeds_max_polls() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        *[make_response(202) for _ in range(3)],
    )
    result = _client(t, max_polls=3)(_task())
    assert result.status is TaskStatus.FAILED
    assert "max_polls" in (result.error or "") or "attempts" in (
        result.error or ""
    )


# --- transient errors -------------------------------------------------------

def test_call_retries_transient_transport_error() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        TransportError("connection reset"),
        _sim_done(4.0),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.SUCCESS
    assert result.fitness == 4.0


def test_call_retries_5xx() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(503, body=b"service unavailable"),
        _sim_done(5.0),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.SUCCESS
    assert result.fitness == 5.0


# --- server-side failures --------------------------------------------------

def test_call_handles_error_status_in_body() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(
            200,
            json_body={"status": "ERROR", "message": "bad expression"},
        ),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.FAILED
    assert "bad expression" in (result.error or "")


def test_call_handles_submit_4xx_as_failure() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect(
        "POST", "/simulations",
        make_response(400, body=b"invalid settings"),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.FAILED
    assert "400" in (result.error or "")


def test_call_handles_missing_location_header() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", make_response(201))
    result = _client(t)(_task())
    assert result.status is TaskStatus.FAILED
    assert "Location" in (result.error or "")


def test_call_handles_missing_is_block() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(200, json_body={"status": "COMPLETE"}),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.FAILED


def test_call_fails_when_fitness_fn_returns_none() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect(
        "GET", "/simulations/123",
        make_response(200, json_body={"id": "a", "is": {"turnover": 0.3}}),
    )
    result = _client(t)(_task())
    assert result.status is TaskStatus.FAILED
    assert "fitness" in (result.error or "").lower()


# --- auth ------------------------------------------------------------------

def test_login_failure_raises_auth_error() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", make_response(401))
    with pytest.raises(BrainAuthError):
        _client(t)(_task())


def test_login_transport_error_raises_auth_error() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", TransportError("dns"))
    with pytest.raises(BrainAuthError):
        _client(t)(_task())


def test_poll_401_invalidates_auth() -> None:
    t = ScriptedTransport()
    t.expect("POST", "/authentication", _login_ok())
    t.expect("POST", "/simulations", _submit_ok())
    t.expect("GET", "/simulations/123", make_response(401))
    with pytest.raises(BrainAuthError):
        _client(t)(_task())


# --- construction ----------------------------------------------------------

def test_client_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        BrainClient(
            credentials=_creds(),
            settings=_settings(),
            transport=ScriptedTransport(),
            max_polls=0,
        )
    with pytest.raises(ValueError):
        BrainClient(
            credentials=_creds(),
            settings=_settings(),
            transport=ScriptedTransport(),
            poll_timeout_seconds=0,
        )
    with pytest.raises(ValueError):
        BrainClient(
            credentials=_creds(),
            settings=_settings(),
            transport=ScriptedTransport(),
            base_poll_interval=0,
        )
    with pytest.raises(ValueError):
        BrainClient(
            credentials=_creds(),
            settings=_settings(),
            transport=ScriptedTransport(),
            http_timeout=0,
        )


def test_build_default_client_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("WQ_BRAIN_USERNAME", "u")
    monkeypatch.setenv("WQ_BRAIN_PASSWORD", "p")
    c = build_default_client()
    assert isinstance(c, BrainClient)


def test_build_default_client_missing_env(monkeypatch) -> None:
    monkeypatch.delenv("WQ_BRAIN_USERNAME", raising=False)
    monkeypatch.delenv("WQ_BRAIN_PASSWORD", raising=False)
    with pytest.raises(BrainAuthError):
        build_default_client()


# --- SimulationResult ------------------------------------------------------

def test_simulation_result_metrics_dict() -> None:
    r = SimulationResult(
        alpha_id="a",
        metrics=(("sharpe", 1.0), ("turnover", 0.2)),
    )
    assert r.metrics_dict() == {"sharpe": 1.0, "turnover": 0.2}