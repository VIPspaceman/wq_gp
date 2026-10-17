"""BRAIN API client that implements the Evaluator protocol.

Design decisions:
- BrainClient depends on HttpTransport, not requests. See D-071.
- Cookies live in the transport's session. See D-072.
- Credentials: env vars, with optional config-file fallback. See D-073,
  D-089, D-135.
- BrainSettings validation is type-only. See D-074.
- Fitness is a pluggable callable taking (metrics, checks). See D-075
  and D-156.
- Poll intervals honor Retry-After; transient errors back off. See
  D-076.
- Each task is bounded by max_polls AND poll_timeout_seconds. See
  D-077.
- Server rejections -> TaskResult(FAILED). Auth failures raise. See
  D-078, D-080.
- Poll responses are two-step; metrics need a second GET. See D-110.
- WARNING with a message is terminal. See D-111.
- A semaphore bounds simultaneous simulations. See D-112.
- Defaults match BRAIN-accepted values. See D-113.
- All logs go to "wqgp.brain". Poll logs carry the sim id. See D-134.
- When an AlphaRegistry is supplied, every successful simulation's
  raw alpha detail is upserted into it. See D-155.
- The default fitness is sharpe-first with soft-tier constraints.
  See D-163.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wqgp.alpha_registry import AlphaCheck, AlphaRegistry, extract_checks
from wqgp.brain.settings import BrainSettings
from wqgp.brain.transport import (
    HttpResponse,
    HttpTransport,
    RequestsTransport,
    TransportError,
)
from wqgp.execution import Task, TaskResult, TaskStatus


log = logging.getLogger("wqgp.brain")


# --- errors -----------------------------------------------------------------

class BrainClientError(RuntimeError):
    """Base class for BRAIN client failures."""


class BrainAuthError(BrainClientError):
    """Raised on authentication failure. Propagates out of __call__."""


class BrainProtocolError(BrainClientError):
    """Raised when a server response does not fit the expected shape."""


class BrainConcurrencyLimitError(BrainClientError):
    """Raised on 429 + CONCURRENT_SIMULATION_LIMIT_EXCEEDED."""


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
        user = os.environ.get("WQ_BRAIN_USERNAME")
        pwd = os.environ.get("WQ_BRAIN_PASSWORD")
        if not user or not pwd:
            raise BrainAuthError(
                "WQ_BRAIN_USERNAME and WQ_BRAIN_PASSWORD must be set"
            )
        return cls(username=user, password=pwd)


# --- fitness -----------------------------------------------------------------

MetricsView = dict[str, float]
FitnessFunction = Callable[
    [MetricsView, "tuple[AlphaCheck, ...] | None"],
    "float | None",
]


# v3 fitness parameters. See D-163.
_FAIL_BASE_PENALTY = 0.35
_VIOLATION_PENALTY = 0.5
_VIOLATION_CAP = 2.0
_PASS_BONUS = 4.0
_PENDING_MULTIPLIER = 0.5


def _check_violation(check: AlphaCheck) -> float:
    """How badly a FAILed check missed its threshold.

    0.0 = at threshold, 1.0 = missed by a whole limit's worth.
    Returns 1.0 for boolean checks without value/limit.
    """
    value = check.value
    limit = check.limit
    if value is None or limit is None:
        return 1.0
    if limit == 0:
        return 1.0
    name = check.name
    if name.startswith("LOW_"):
        gap = limit - value
    elif name.startswith("HIGH_"):
        gap = value - limit
    else:
        return 1.0
    if gap <= 0:
        return 0.0
    return gap / abs(limit)


def sharpe_only_fitness(
    metrics: MetricsView,
    checks: tuple[AlphaCheck, ...] | None = None,
) -> float | None:
    """Legacy fitness: return sharpe, ignore checks.

    Kept for `--fitness-strategy sharpe` and for A/B comparison.
    """
    sharpe = metrics.get("sharpe")
    if sharpe is None:
        return None
    if isinstance(sharpe, bool):
        return None
    try:
        value = float(sharpe)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def feasibility_fitness(
    metrics: MetricsView,
    checks: tuple[AlphaCheck, ...] | None = None,
) -> float | None:
    """Sharpe-first fitness with soft-tier constraint penalties.

    Replaces the three-tier formula from D-157 with a formula that
    keeps sharpe as the dominant axis while still preferring
    submittable alphas. See D-163.

    Behavior:
      - Sharpe is the base score, not compressed by tanh.
      - No FAIL, no PENDING: score = sharpe + 4.0.
      - No FAIL, at least one PENDING/WARNING:
        score = sharpe + 2.0.
      - At least one FAIL:
        score = sharpe - 0.35 * n_fails
                     - 0.5 * sum(min(violation, 2.0))

    Effect on the A/B failure modes seen in D-162:
      - A submittable alpha with sharpe=0.5 lands at 4.5.
      - A non-submittable alpha with sharpe=1.35 and one FAIL lands
        at ~0.85. The submittable one wins, but the non-submittable
        one is not annihilated; it still competes on the basis of
        its sharpe within the FAIL tier.
      - Inside the PASS tier, sharpe differences are preserved in
        full (a pass at 1.30 vs 0.50 differ by 0.80, not 0.12).

    `checks` may be None for records written before D-156. In that
    case the function falls back to raw sharpe.
    """
    sharpe = metrics.get("sharpe")
    if sharpe is None:
        return None
    if isinstance(sharpe, bool):
        return None
    try:
        sharpe = float(sharpe)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(sharpe):
        return None

    if not checks:
        return sharpe

    # SELF_CORRELATION is PENDING until submission; ignore at GP time.
    relevant = tuple(c for c in checks if c.name != "SELF_CORRELATION")
    if not relevant:
        return sharpe

    failures = tuple(c for c in relevant if c.failed)
    if failures:
        score = sharpe
        for check in failures:
            score -= _FAIL_BASE_PENALTY
            gap = _check_violation(check)
            score -= _VIOLATION_PENALTY * min(gap, _VIOLATION_CAP)
        return score

    has_pending = any(
        c.result in ("PENDING", "WARNING") for c in relevant
    )
    if has_pending:
        return sharpe + _PASS_BONUS * _PENDING_MULTIPLIER
    return sharpe + _PASS_BONUS


# The default is the v3 formula (formerly called "feasibility"; the
# name is kept for backward compatibility with --fitness-strategy).
default_fitness = feasibility_fitness


# --- simulation result -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Parsed simulation outcome."""
    alpha_id: str | None
    metrics: tuple[tuple[str, float], ...]
    error_message: str | None = None
    raw_alpha: dict | None = None

    def metrics_dict(self) -> MetricsView:
        return dict(self.metrics)


# --- helpers ---------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _sim_id_from_url(sim_url: str) -> str:
    seg = sim_url.rstrip("/").rsplit("/", 1)[-1]
    if not seg or len(seg) > 32:
        return "????"
    return seg


def _format_result_line(
    *,
    score: float | None,
    metrics: MetricsView,
    expression: str,
    error: str | None,
) -> str:
    expr = _truncate(expression, 100)
    if error is not None:
        return f"[result] FAILED err={_truncate(error, 80)} expr={expr}"

    parts: list[str] = []
    if score is not None:
        parts.append(f"score={score:.4f}")
    for key in ("sharpe", "fitness", "turnover", "drawdown", "returns"):
        v = metrics.get(key)
        if v is not None:
            parts.append(f"{key}={v:.4f}")
    return f"[result] {' '.join(parts)} expr={expr}"


# --- client ------------------------------------------------------------------

_BASE_URL = "https://api.worldquantbrain.com"
_LOGIN_URL = f"{_BASE_URL}/authentication"
_SIMULATIONS_URL = f"{_BASE_URL}/simulations"


class BrainClient:
    """Implements the Evaluator protocol against the BRAIN API."""

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
        "_alpha_registry",
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
        fitness_fn: FitnessFunction = feasibility_fitness,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_polls: int = 120,
        poll_timeout_seconds: float = 600.0,
        base_poll_interval: float = 3.0,
        http_timeout: float = 30.0,
        max_concurrent: int = 2,
        alpha_registry: AlphaRegistry | None = None,
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
        self._alpha_registry = alpha_registry
        self._concurrency_semaphore = threading.BoundedSemaphore(max_concurrent)
        self._authenticated = False
        self._lock = threading.Lock()

    # --- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    # --- Evaluator Protocol -------------------------------------------

    def __call__(self, task: Task) -> TaskResult:
        self._ensure_authenticated()

        self._concurrency_semaphore.acquire()
        try:
            try:
                sim = self._submit_and_poll(task.expression)
            except BrainAuthError:
                raise
            except BrainClientError as e:
                log.info(
                    "%s",
                    _format_result_line(
                        score=None,
                        metrics={},
                        expression=task.expression,
                        error=str(e),
                    ),
                )
                return TaskResult(
                    fingerprint=task.fingerprint,
                    status=TaskStatus.FAILED,
                    error=str(e),
                )
        finally:
            self._concurrency_semaphore.release()

        if (
            sim.raw_alpha is not None
            and self._alpha_registry is not None
        ):
            try:
                self._alpha_registry.upsert_raw(sim.raw_alpha)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "alpha registry upsert failed: %s", e,
                )

        metrics_dict = sim.metrics_dict()

        if sim.error_message is not None:
            log.info(
                "%s",
                _format_result_line(
                    score=None,
                    metrics=metrics_dict,
                    expression=task.expression,
                    error=sim.error_message,
                ),
            )
            return TaskResult(
                fingerprint=task.fingerprint,
                status=TaskStatus.FAILED,
                error=sim.error_message,
            )

        checks: tuple[AlphaCheck, ...] | None = None
        if sim.raw_alpha is not None:
            checks = extract_checks(sim.raw_alpha)

        fitness = self._fitness_fn(metrics_dict, checks)
        if fitness is None:
            log.info(
                "%s",
                _format_result_line(
                    score=None,
                    metrics=metrics_dict,
                    expression=task.expression,
                    error="no fitness derivable",
                ),
            )
            return TaskResult(
                fingerprint=task.fingerprint,
                status=TaskStatus.FAILED,
                error="no fitness derivable from metrics",
            )

        log.info(
            "%s",
            _format_result_line(
                score=float(fitness),
                metrics=metrics_dict,
                expression=task.expression,
                error=None,
            ),
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
        log.info(
            "login: attempting (user=%s)", self._credentials.username
        )
        started = time.monotonic()
        try:
            resp = self._transport.request(
                "POST",
                _LOGIN_URL,
                auth=(self._credentials.username, self._credentials.password),
                timeout=self._http_timeout,
            )
        except TransportError as e:
            log.error("login: transport error: %s", e)
            raise BrainAuthError(f"login transport error: {e}") from e

        elapsed = time.monotonic() - started
        if resp.status_code not in (200, 201):
            log.error(
                "login: failed HTTP %d in %.2fs",
                resp.status_code, elapsed,
            )
            raise BrainAuthError(
                f"login failed with HTTP {resp.status_code}"
            )
        log.info("login: OK in %.2fs", elapsed)

    def _invalidate_auth(self) -> None:
        with self._lock:
            self._authenticated = False

    # --- submit + poll -------------------------------------------------

    def _submit_and_poll(self, expression: str) -> SimulationResult:
        sim_url = self._submit(expression)
        return self._poll(sim_url)

    def _submit(self, expression: str) -> str:
        log.debug("[submit] %s", _truncate(expression, 80))

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
            log.warning("[submit] transport error: %s", e)
            raise BrainClientError(f"submit transport error: {e}") from e

        if resp.status_code == 401:
            self._invalidate_auth()
            log.warning("[submit] 401, invalidating auth")
            raise BrainAuthError("submit rejected: not authenticated")

        if resp.status_code == 429:
            detail = _extract_error_detail(resp)
            if "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in detail:
                log.warning("[submit] concurrent limit: %s", detail)
                raise BrainConcurrencyLimitError(
                    f"concurrent simulation limit exceeded: "
                    f"{detail or 'no detail'}"
                )
            log.warning("[submit] rate limited: %s", detail)
            raise BrainClientError(
                f"submit rate limited (429): {detail or _safe_text(resp)}"
            )

        if resp.status_code not in (200, 201):
            log.warning(
                "[submit] HTTP %d: %s",
                resp.status_code, _safe_text(resp),
            )
            raise BrainClientError(
                f"submit failed with HTTP {resp.status_code}: "
                f"{_safe_text(resp)}"
            )

        location = resp.header("Location")
        if not location:
            log.warning("[submit] 201 without Location header")
            raise BrainProtocolError(
                "submit response missing Location header"
            )
        log.debug(
            "[submit] OK sim=%s", _sim_id_from_url(location)
        )
        return location

    def _poll(self, sim_url: str) -> SimulationResult:
        sim_id = _sim_id_from_url(sim_url)
        deadline = time.monotonic() + self._poll_timeout_seconds
        consecutive_errors = 0
        last_progress: float | None = None

        for _ in range(self._max_polls):
            if time.monotonic() > deadline:
                log.warning(
                    "[poll sim=%s] exceeded %.0fs timeout",
                    sim_id, self._poll_timeout_seconds,
                )
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
                wait = self._backoff(consecutive_errors)
                log.debug(
                    "[poll sim=%s] transport error, backoff %.1fs",
                    sim_id, wait,
                )
                self._sleep_fn(wait)
                continue

            if resp.status_code == 401:
                self._invalidate_auth()
                log.warning("[poll sim=%s] 401, invalidating auth", sim_id)
                raise BrainAuthError("poll rejected: not authenticated")

            if resp.status_code == 202:
                self._sleep_fn(self._next_poll_interval(resp))
                continue

            if resp.status_code != 200:
                if 500 <= resp.status_code < 600:
                    consecutive_errors += 1
                    wait = self._backoff(consecutive_errors)
                    log.debug(
                        "[poll sim=%s] HTTP %d, backoff %.1fs",
                        sim_id, resp.status_code, wait,
                    )
                    self._sleep_fn(wait)
                    continue
                log.warning(
                    "[poll sim=%s] HTTP %d: %s",
                    sim_id, resp.status_code, _safe_text(resp),
                )
                raise BrainClientError(
                    f"poll failed with HTTP {resp.status_code}: "
                    f"{_safe_text(resp)}"
                )

            try:
                data = resp.json()
            except ValueError as e:
                log.warning("[poll sim=%s] non-JSON body: %s", sim_id, e)
                raise BrainProtocolError(
                    f"poll returned non-JSON body: {e}"
                ) from e

            if not isinstance(data, dict):
                raise BrainProtocolError(
                    f"poll returned {type(data).__name__}, expected object"
                )

            status = data.get("status")

            if status == "ERROR":
                msg = (
                    data.get("message")
                    or data.get("error")
                    or "simulation ERROR"
                )
                log.warning(
                    "[poll sim=%s] ERROR: %s",
                    sim_id, _truncate(str(msg), 100),
                )
                return SimulationResult(
                    alpha_id=None,
                    metrics=(),
                    error_message=str(msg),
                )

            if status == "WARNING":
                msg = data.get("message", "")
                if msg:
                    log.warning(
                        "[poll sim=%s] WARNING terminal: %s",
                        sim_id, _truncate(str(msg), 100),
                    )
                    return SimulationResult(
                        alpha_id=None,
                        metrics=(),
                        error_message=str(msg),
                    )
                self._sleep_fn(self._next_poll_interval(resp))
                continue

            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                if not isinstance(alpha_id, str) or not alpha_id:
                    raise BrainProtocolError(
                        "COMPLETE response without 'alpha' id"
                    )
                log.debug(
                    "[poll sim=%s] complete alpha=%s", sim_id, alpha_id,
                )
                return self._fetch_alpha_metrics(alpha_id)

            progress = data.get("progress")
            if progress is not None:
                try:
                    p = float(progress)
                except (TypeError, ValueError):
                    p = None
                if p is not None and (
                    last_progress is None or abs(p - last_progress) > 0.001
                ):
                    log.info(
                        "[poll sim=%s] progress=%.0f%%",
                        sim_id, p * 100.0,
                    )
                    last_progress = p

            self._sleep_fn(self._next_poll_interval(resp))

        log.warning(
            "[poll sim=%s] exceeded %d attempts",
            sim_id, self._max_polls,
        )
        raise BrainClientError(
            f"poll exceeded {self._max_polls} attempts"
        )

    def _fetch_alpha_metrics(self, alpha_id: str) -> SimulationResult:
        log.debug("[alpha] fetching %s", alpha_id)
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

        log.debug("[alpha] %s got %d metrics", alpha_id, len(metrics))
        return SimulationResult(
            alpha_id=alpha_id,
            metrics=tuple(metrics),
            raw_alpha=data,
        )

    def _next_poll_interval(self, resp: HttpResponse) -> float:
        raw = resp.header("Retry-After")
        if raw is None:
            return self._base_poll_interval
        try:
            return max(0.0, float(raw))
        except ValueError:
            return self._base_poll_interval

    def _backoff(self, attempt: int) -> float:
        return min(self._base_poll_interval * (2 ** (attempt - 1)), 30.0)


def _extract_error_detail(resp: HttpResponse) -> str:
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
    try:
        return resp.body.decode("utf-8", errors="replace")[:200]
    except Exception:  # noqa: BLE001
        return "<undecodable>"


# --- convenience factory -----------------------------------------------------

def build_default_client(
    *,
    settings: BrainSettings | None = None,
    fitness_fn: FitnessFunction = feasibility_fitness,
    max_concurrent: int = 2,
    alpha_registry: AlphaRegistry | None = None,
) -> BrainClient:
    """Construct a BrainClient wired to real HTTP and environment creds."""
    return BrainClient(
        credentials=BrainCredentials.from_env(),
        settings=settings or BrainSettings(),
        transport=RequestsTransport(),
        fitness_fn=fitness_fn,
        max_concurrent=max_concurrent,
        alpha_registry=alpha_registry,
    )