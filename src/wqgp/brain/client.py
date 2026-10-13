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
  raise BrainAuthError, which __call__ propagates unchanged. See
  D-078 and D-080.
- Poll responses have two shapes: {"progress": float} while running,
  {"status": "COMPLETE", "alpha": "<id>"} when done. Metrics are NOT
  in the poll response; a second GET /alphas/{id} is required. See
  D-110.
- WARNING with a non-empty message is a terminal failure, not a
  continuing state. See D-111.
- A semaphore bounds simultaneously active simulations. See D-112.
- BrainSettings defaults match BRAIN's observed-accepted values.
  See D-113.
"""
from __future__ import annotations

import os
import logging
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
log = logging.getLogger(__name__)


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


class BrainConcurrencyLimitError(BrainClientError):
    """Raised on 429 + CONCURRENT_SIMULATION_LIMIT_EXCEEDED.

    Distinct from a generic rate limit because BRAIN's message is
    specific: the account has too many active simulations. The
    semaphore (D-112) should prevent this in normal operation; seeing
    it means the configured max_concurrent is too high for the account.
    """


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
    """Frozen simulation settings.

    Defaults (decay=4, neutralization="INDUSTRY") match values observed
    to be accepted by BRAIN's /simulations endpoint for a TUTORIAL
    account. See D-113.
    """
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

    Thread-safety:
    - Login state is guarded by a lock.
    - Active simulations are bounded by a semaphore (D-112).
    Both are reentrant-safe; multiple worker threads may call the
    client concurrently.
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
        "_max_concurrent",
        "_concurrency_semaphore",
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
        base_poll_interval: float = 3.0,
        http_timeout: float = 30.0,
        max_concurrent: int = 2,
    ) -> None:
        if max_polls < 1:
            raise ValueError("max_polls must be >= 1")
        if poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be > 0")
        if base_poll_interval <= 0:
            raise ValueError("base_poll_interval must be > 0")
        if http_timeout <= 0:
            raise ValueError("http_timeout must be > 0")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        self._credentials = credentials
        self._settings = settings
        self._transport = transport
        self._fitness_fn = fitness_fn
        self._sleep_fn = sleep_fn
        self._max_polls = max_polls
        self._poll_timeout_seconds = poll_timeout_seconds
        self._base_poll_interval = base_poll_interval
        self._http_timeout = http_timeout
        self._max_concurrent = max_concurrent
        self._concurrency_semaphore = threading.BoundedSemaphore(max_concurrent)
        self._authenticated = False
        self._lock = threading.Lock()

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    # --- Evaluator Protocol -------------------------------------------

    def __call__(self, task: Task) -> TaskResult:
        self._ensure_authenticated()

        # Bound simultaneously active simulations. A slot is held
        # across the whole submit+poll cycle, not just the POST. See
        # D-112.
        self._concurrency_semaphore.acquire()
        try:
            try:
                sim = self._submit_and_poll(task.expression)
            except BrainAuthError:
                # Auth failure during submit/poll is a distinct channel:
                # propagate unchanged. See D-078 and D-080.
                raise
            except BrainClientError as e:
                return TaskResult(
                    fingerprint=task.fingerprint,
                    status=TaskStatus.FAILED,
                    error=str(e),
                )
        finally:
            self._concurrency_semaphore.release()

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

        if resp.status_code == 429:
            detail = _extract_error_detail(resp)
            if "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in detail:
                raise BrainConcurrencyLimitError(
                    f"concurrent simulation limit exceeded: "
                    f"{detail or 'no detail'}"
                )
            raise BrainClientError(
                f"submit rate limited (429): {detail or _safe_text(resp)}"
            )

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
        log.debug("[submit] %s -> %s", expression[:80], location)
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
                self._sleep_fn(self._backoff(consecutive_errors))
                continue

            if resp.status_code == 401:
                self._invalidate_auth()
                raise BrainAuthError("poll rejected: not authenticated")

            # 202 = accepted, not done. No body to parse; just wait.
            # BRAIN observed in practice returns 200 + progress instead
            # of 202, but the reference project still accepts 202 and
            # the HTTP semantics ("accepted") fit our loop. See D-114.
            if resp.status_code == 202:
                self._sleep_fn(self._next_poll_interval(resp))
                continue

            if resp.status_code != 200:
                if 500 <= resp.status_code < 600:
                    consecutive_errors += 1
                    self._sleep_fn(self._backoff(consecutive_errors))
                    continue
                raise BrainClientError(
                    f"poll failed with HTTP {resp.status_code}: "
                    f"{_safe_text(resp)}"
                )

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

            status = data.get("status")

            # ---- ERROR: terminal failure ----
            if status == "ERROR":
                msg = (
                    data.get("message")
                    or data.get("error")
                    or "simulation ERROR"
                )
                return SimulationResult(
                    alpha_id=None,
                    metrics=(),
                    error_message=str(msg),
                )

            # ---- WARNING: terminal if it carries a message (D-111) ----
            if status == "WARNING":
                msg = data.get("message", "")
                if msg:
                    return SimulationResult(
                        alpha_id=None,
                        metrics=(),
                        error_message=str(msg),
                    )
                # No message: treat as still-running and keep polling.
                self._sleep_fn(self._next_poll_interval(resp))
                continue

            # ---- COMPLETE: fetch metrics in a second request (D-110) ----
            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                if not isinstance(alpha_id, str) or not alpha_id:
                    raise BrainProtocolError(
                        "COMPLETE response without 'alpha' id"
                    )
                log.debug("[poll] complete alpha=%s", alpha_id)
                return self._fetch_alpha_metrics(alpha_id)

            # ---- No status: {"progress": float}, keep waiting ----
            self._sleep_fn(self._next_poll_interval(resp))

        raise BrainClientError(f"poll exceeded {self._max_polls} attempts")

    def _fetch_alpha_metrics(self, alpha_id: str) -> SimulationResult:
        """GET /alphas/{id} for the in-sample metrics block (D-110)."""
        url = f"{_BASE_URL}/alphas/{alpha_id}"
        try:
            resp = self._transport.request(
                "GET", url, timeout=self._http_timeout
            )
        except TransportError as e:
            raise BrainClientError(
                f"alpha fetch transport error: {e}"
            ) from e

        if resp.status_code == 401:
            self._invalidate_auth()
            raise BrainAuthError("alpha fetch rejected: not authenticated")
        if resp.status_code != 200:
            raise BrainClientError(
                f"alpha fetch failed with HTTP {resp.status_code}: "
                f"{_safe_text(resp)}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise BrainProtocolError(
                f"alpha response non-JSON: {e}"
            ) from e

        if not isinstance(data, dict):
            raise BrainProtocolError("alpha response is not an object")

        metrics_raw = data.get("is")
        if not isinstance(metrics_raw, dict):
            raise BrainProtocolError(
                "alpha response missing 'is' block"
            )

        metrics: list[tuple[str, float]] = []
        for k, v in metrics_raw.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                metrics.append((str(k), float(v)))

        log.debug("[alpha] %s metrics=%d", alpha_id, len(metrics))
        return SimulationResult(
            alpha_id=alpha_id,
            metrics=tuple(metrics),
        )

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


def _extract_error_detail(resp: HttpResponse) -> str:
    """Read BRAIN's {'detail': '...'} error body, or fall back to text."""
    try:
        data = resp.json()
    except ValueError:
        return _safe_text(resp)
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
    return _safe_text(resp)


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
    max_concurrent: int = 2,
) -> BrainClient:
    """Construct a BrainClient wired to real HTTP and environment creds."""
    return BrainClient(
        credentials=BrainCredentials.from_env(),
        settings=settings or BrainSettings(),
        transport=RequestsTransport(),
        fitness_fn=fitness_fn,
        max_concurrent=max_concurrent,
    )