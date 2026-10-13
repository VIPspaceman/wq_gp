"""BRAIN API client that implements the Evaluator protocol.

Design decisions:
- BrainClient depends on HttpTransport, not requests. See D-071.
- Cookies live in the transport's session. See D-072.
- Credentials are supplied at construction; from_env() reads the
  standard variables. See D-073.
- BrainSettings is a frozen dataclass; validation is type-only. See
  D-074.
- Fitness is derived from in-sample metrics via a pluggable callable;
  the default returns the sharpe. See D-075.
- Poll intervals come from the server's Retry-After header; transient
  errors get exponential backoff. See D-076.
- Each task is bounded by max_polls AND poll_timeout_seconds; sleep_fn
  is injectable for tests. See D-077.
- Server rejections produce TaskResult(status=FAILED). Auth failures
  raise BrainAuthError, which __call__ propagates unchanged. See D-078
  and D-080.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wqgp.brain.transport import (
    HttpResponse,
    HttpTransport,
    RequestsTransport,
    TransportError,
)
from wqgp.execution import Task, TaskResult, TaskStatus


# --- errors -----------------------------------------------------------------

class BrainClientError(RuntimeError):
    """Base class for BRAIN client failures."""


class BrainAuthError(BrainClientError):
    """Raised on authentication failure. Propagates out of __call__.

    Note: this is a subclass of BrainClientError, but __call__ re-raises
    it before the generic BrainClientError handler. See D-080.
    """


class BrainProtocolError(BrainClientError):
    """Raised when a server response does not fit the expected shape."""


# --- config -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BrainCredentials:
    """BRAIN platform credentials."""
    username: str
    password: str

    def __post_init__(self) -> None:
        if not self.username or not self.username.strip():
            raise ValueError("username must be non-empty")
        if not self.password:
            raise ValueError("password must be non-empty")

    @classmethod
    def from_env(cls) -> "BrainCredentials":
        """Read WQ_BRAIN_USERNAME / WQ_BRAIN_PASSWORD from the environment."""
        user = os.environ.get("WQ_BRAIN_USERNAME")
        pwd = os.environ.get("WQ_BRAIN_PASSWORD")
        if not user or not pwd:
            raise BrainAuthError(
                "WQ_BRAIN_USERNAME and WQ_BRAIN_PASSWORD must be set"
            )
        return cls(username=user, password=pwd)


@dataclass(frozen=True, slots=True)
class BrainSettings:
    """Frozen simulation settings. Type-only validation (see D-074)."""
    instrument_type: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "SUBINDUSTRY"
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


# --- fitness -----------------------------------------------------------------

MetricsView = dict[str, float]
FitnessFunction = Callable[[MetricsView], float | None]


def default_fitness(metrics: MetricsView) -> float | None:
    """Default: return `sharpe` from the in-sample metrics, or None."""
    sharpe = metrics.get("sharpe")
    if sharpe is None:
        return None
    return float(sharpe)


# --- simulation result -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Parsed simulation outcome."""
    alpha_id: str | None
    metrics: tuple[tuple[str, float], ...]
    error_message: str | None = None

    def metrics_dict(self) -> MetricsView:
        return dict(self.metrics)


# --- client ------------------------------------------------------------------

_BASE_URL = "https://api.worldquantbrain.com"
_LOGIN_URL = f"{_BASE_URL}/authentication"
_SIMULATIONS_URL = f"{_BASE_URL}/simulations"


class BrainClient:
    """Implements the Evaluator protocol against the BRAIN API.

    Thread-safety: `__call__` may be invoked from multiple worker threads
    concurrently. Login state is guarded by an internal lock; the rest of
    the request flow is stateless apart from that flag.
    """

    __slots__ = (
        "_credentials",
        "_settings",
        "_transport",
        "_fitness_fn",
        "_sleep_fn",
        "_max_polls",
        "_poll_timeout_seconds",
        "_base_poll_interval",
        "_http_timeout",
        "_authenticated",
        "_lock",
    )

    def __init__(
        self,
        credentials: BrainCredentials,
        settings: BrainSettings,
        transport: HttpTransport,
        *,
        fitness_fn: FitnessFunction = default_fitness,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_polls: int = 120,
        poll_timeout_seconds: float = 600.0,
        base_poll_interval: float = 1.0,
        http_timeout: float = 30.0,
    ) -> None:
        if max_polls < 1:
            raise ValueError("max_polls must be >= 1")
        if poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be > 0")
        if base_poll_interval <= 0:
            raise ValueError("base_poll_interval must be > 0")
        if http_timeout <= 0:
            raise ValueError("http_timeout must be > 0")

        self._credentials = credentials
        self._settings = settings
        self._transport = transport
        self._fitness_fn = fitness_fn
        self._sleep_fn = sleep_fn
        self._max_polls = max_polls
        self._poll_timeout_seconds = poll_timeout_seconds
        self._base_poll_interval = base_poll_interval
        self._http_timeout = http_timeout
        self._authenticated = False
        self._lock = threading.Lock()

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    # --- Evaluator Protocol -------------------------------------------

    def __call__(self, task: Task) -> TaskResult:
        # Login failures raise: the Executor's D-034 handler records
        # this task as FAILED, and a subsequent task will retry login.
        self._ensure_authenticated()

        try:
            sim = self._submit_and_poll(task.expression)
        except BrainAuthError:
            # Auth failure during submit/poll is a distinct channel:
            # propagate it unchanged so it is not confused with a
            # per-expression failure. See D-078 and D-080.
            raise
        except BrainClientError as e:
            return TaskResult(
                fingerprint=task.fingerprint,
                status=TaskStatus.FAILED,
                error=str(e),
            )

        if sim.error_message is not None:
            return TaskResult(
                fingerprint=task.fingerprint,
                status=TaskStatus.FAILED,
                error=sim.error_message,
            )

        fitness = self._fitness_fn(sim.metrics_dict())
        if fitness is None:
            return TaskResult(
                fingerprint=task.fingerprint,
                status=TaskStatus.FAILED,
                error="no fitness derivable from metrics",
            )

        return TaskResult(
            fingerprint=task.fingerprint,
            status=TaskStatus.SUCCESS,
            fitness=float(fitness),
            metrics=sim.metrics,
        )

    # --- auth ----------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        with self._lock:
            if self._authenticated:
                return
            self._login_locked()
            self._authenticated = True

    def _login_locked(self) -> None:
        try:
            resp = self._transport.request(
                "POST",
                _LOGIN_URL,
                auth=(self._credentials.username, self._credentials.password),
                timeout=self._http_timeout,
            )
        except TransportError as e:
            raise BrainAuthError(f"login transport error: {e}") from e

        if resp.status_code not in (200, 201):
            raise BrainAuthError(
                f"login failed with HTTP {resp.status_code}"
            )

    def _invalidate_auth(self) -> None:
        with self._lock:
            self._authenticated = False

    # --- submit + poll -------------------------------------------------

    def _submit_and_poll(self, expression: str) -> SimulationResult:
        sim_url = self._submit(expression)
        return self._poll(sim_url)

    def _submit(self, expression: str) -> str:
        body = {
            "type": "REGULAR",
            "settings": self._settings.to_api_dict(),
            "regular": expression,
        }
        try:
            resp = self._transport.request(
                "POST",
                _SIMULATIONS_URL,
                json_body=body,
                headers={"Content-Type": "application/json"},
                timeout=self._http_timeout,
            )
        except TransportError as e:
            raise BrainClientError(f"submit transport error: {e}") from e

        if resp.status_code == 401:
            self._invalidate_auth()
            raise BrainAuthError("submit rejected: not authenticated")
        if resp.status_code not in (200, 201):
            raise BrainClientError(
                f"submit failed with HTTP {resp.status_code}: "
                f"{_safe_text(resp)}"
            )

        location = resp.header("Location")
        if not location:
            raise BrainProtocolError(
                "submit response missing Location header"
            )
        return location

    def _poll(self, sim_url: str) -> SimulationResult:
        deadline = time.monotonic() + self._poll_timeout_seconds
        consecutive_errors = 0

        for _ in range(self._max_polls):
            if time.monotonic() > deadline:
                raise BrainClientError(
                    f"poll exceeded {self._poll_timeout_seconds}s"
                )

            try:
                resp = self._transport.request(
                    "GET", sim_url, timeout=self._http_timeout
                )
                consecutive_errors = 0
            except TransportError:
                consecutive_errors += 1
                self._sleep_fn(
                    self._backoff(consecutive_errors)
                )
                continue

            if resp.status_code == 401:
                self._invalidate_auth()
                raise BrainAuthError("poll rejected: not authenticated")

            if resp.status_code == 200:
                return self._parse_result(resp)

            if resp.status_code in (201, 202):
                self._sleep_fn(self._next_poll_interval(resp))
                continue

            if 500 <= resp.status_code < 600:
                consecutive_errors += 1
                self._sleep_fn(self._backoff(consecutive_errors))
                continue

            raise BrainClientError(
                f"poll failed with HTTP {resp.status_code}: "
                f"{_safe_text(resp)}"
            )

        raise BrainClientError(f"poll exceeded {self._max_polls} attempts")

    def _next_poll_interval(self, resp: HttpResponse) -> float:
        """Honor Retry-After; fall back to base_poll_interval."""
        raw = resp.header("Retry-After")
        if raw is None:
            return self._base_poll_interval
        try:
            return max(0.0, float(raw))
        except ValueError:
            return self._base_poll_interval

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff, capped so tests can't hang."""
        return min(self._base_poll_interval * (2 ** (attempt - 1)), 30.0)

    def _parse_result(self, resp: HttpResponse) -> SimulationResult:
        try:
            data = resp.json()
        except ValueError as e:
            raise BrainProtocolError(
                f"poll returned non-JSON body: {e}"
            ) from e

        if not isinstance(data, dict):
            raise BrainProtocolError(
                f"poll returned {type(data).__name__}, expected object"
            )

        # Explicit server-side error status.
        status = data.get("status")
        if isinstance(status, str) and status.upper() in {
            "ERROR", "FAIL", "FAILED", "CANCELLED",
        }:
            msg = (
                data.get("message")
                or data.get("error")
                or f"simulation status={status}"
            )
            return SimulationResult(
                alpha_id=None,
                metrics=(),
                error_message=str(msg),
            )

        metrics_raw = data.get("is")
        if not isinstance(metrics_raw, dict):
            raise BrainProtocolError(
                "poll response missing 'is' block"
            )

        metrics: list[tuple[str, float]] = []
        for k, v in metrics_raw.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                metrics.append((str(k), float(v)))

        alpha_id = data.get("id")
        if alpha_id is not None and not isinstance(alpha_id, str):
            alpha_id = None

        return SimulationResult(
            alpha_id=alpha_id,
            metrics=tuple(metrics),
        )


def _safe_text(resp: HttpResponse) -> str:
    """First 200 chars of the body, decoded leniently. Never raises."""
    try:
        return resp.body.decode("utf-8", errors="replace")[:200]
    except Exception:  # noqa: BLE001 -- truly defensive
        return "<undecodable>"


# --- convenience factory -----------------------------------------------------

def build_default_client(
    *,
    settings: BrainSettings | None = None,
    fitness_fn: FitnessFunction = default_fitness,
) -> BrainClient:
    """Construct a BrainClient wired to real HTTP and environment creds."""
    return BrainClient(
        credentials=BrainCredentials.from_env(),
        settings=settings or BrainSettings(),
        transport=RequestsTransport(),
        fitness_fn=fitness_fn,
    )