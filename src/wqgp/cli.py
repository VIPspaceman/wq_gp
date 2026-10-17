"""Command-line interface for wqgp.

Design decisions:
- Subparsers via argparse. See D-087.
- Config is JSON, flag > config > default. See D-088.
- Credentials: env vars, with optional config-file fallback. See
  D-089 and D-135.
- SIGINT propagates; exit code 130. See D-090.
- Structured logging to stderr; --log-level controls verbosity. See
  D-091.
- --dry-run uses a deterministic evaluator. See D-092.
- --telegram opts into Telegram notifications. See D-097.
- `signatures` audits a signature table; never writes it. See D-102.
- `build-pool` accepts simulation settings flags. See D-107.
- Logs go to the "wqgp.cli" channel. See D-134.
- `--settings` and `--run-config` split the two loader roles.
  See D-136.
- `cache` subcommand inspects and manages the cache. See D-138.
- `run` builds an EvaluationConfig and an EvaluationCache. See
  D-146, D-147, D-149.
- `--fitness-strategy` selects the fitness function. See D-158.
- `--seed-count` and `--no-seed-injection` control seed injection.
  Default is `seed_count=3` when neither is given. See D-170.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from wqgp import __version__
from wqgp.alpha_registry import AlphaRegistry
from wqgp.brain.client import (
    BrainAuthError,
    BrainCredentials,
    BrainSettings,
    build_default_client,
    feasibility_fitness,
    sharpe_only_fitness,
)
from wqgp.brain.pool_builder import (
    OperatorSignatureTable,
    PoolBuilder,
    PoolBuilderError,
    PoolBuilderProtocolError,
)
from wqgp.brain.transport import HttpTransport, RequestsTransport
from wqgp.budget import Budget
from wqgp.cache import EvaluationCache
from wqgp.checkpoint import CheckpointStore, CorruptCheckpointError
from wqgp.coordinator import Coordinator, CoordinatorError, RunConfig
from wqgp.evaluation_config import EvaluationConfig
from wqgp.execution import Evaluator, Task, TaskResult, TaskStatus
from wqgp.logging_setup import LoggingSettings, setup_logging
from wqgp.notification.null import NullNotifier
from wqgp.notification.protocol import Notifier
from wqgp.notification.telegram import (
    TelegramConfigError,
    TelegramCredentials,
    TelegramNotifier,
)
from wqgp.pool import Pool, PoolError
from wqgp.selection import SelectionConfig
from wqgp.serialization import SerializationError
from wqgp.settings import (
    Settings,
    is_insecure_credentials_file,
)
from wqgp.signatures_audit import (
    SignatureAudit,
    audit_signatures,
    format_skeletons,
)


log = logging.getLogger("wqgp.cli")


# --- exit codes -------------------------------------------------------------

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


class CliError(RuntimeError):
    """Raised for expected user-facing errors."""


_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _resolve_settings_path(args: argparse.Namespace) -> Path | None:
    settings_path = getattr(args, "settings", None)
    if settings_path:
        return Path(settings_path)
    config_path = getattr(args, "config", None)
    if config_path:
        return Path(config_path)
    return None


def _load_settings(args: argparse.Namespace) -> Settings:
    path = _resolve_settings_path(args)
    try:
        settings = Settings.load(path)
    except (FileNotFoundError, ValueError) as e:
        raise CliError(str(e)) from e

    if path is not None and settings.credentials.is_complete():
        if is_insecure_credentials_file(path):
            log.warning(
                "credentials are stored in %s, which is readable by "
                "other users on this machine (mode=%o). Run "
                "'chmod 600 %s' to restrict access.",
                path, path.stat().st_mode & 0o777, path,
            )

    log_level = getattr(args, "log_level", None)
    if log_level is not None:
        settings = Settings(
            logging=LoggingSettings(
                level=log_level,
                style=settings.logging.style,
                channels=settings.logging.channels,
                file=settings.logging.file,
            ),
            defaults=settings.defaults,
            credentials=settings.credentials,
            cache=settings.cache,
            raw=settings.raw,
        )

    log_file = getattr(args, "log_file", None)
    if log_file is not None:
        settings = Settings(
            logging=LoggingSettings(
                level=settings.logging.level,
                style=settings.logging.style,
                channels=settings.logging.channels,
                file=str(log_file),
            ),
            defaults=settings.defaults,
            credentials=settings.credentials,
            cache=settings.cache,
            raw=settings.raw,
        )

    return settings


def _resolve_credentials(settings: Settings) -> BrainCredentials:
    try:
        return BrainCredentials.from_env()
    except BrainAuthError:
        pass

    if settings.credentials.is_complete():
        log.warning(
            "using credentials from the config file; environment "
            "variables WQ_BRAIN_USERNAME / WQ_BRAIN_PASSWORD would "
            "take precedence if set. Keep the file chmod 600 and "
            "out of git."
        )
        return BrainCredentials(
            username=settings.credentials.username,  # type: ignore[arg-type]
            password=settings.credentials.password,  # type: ignore[arg-type]
        )

    raise CliError(
        "no BRAIN credentials: set WQ_BRAIN_USERNAME / "
        "WQ_BRAIN_PASSWORD, or add a 'credentials' block to the "
        "settings file (see wqgp.schema.json)."
    )


def _make_transport(args: argparse.Namespace) -> HttpTransport:
    return RequestsTransport()


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise CliError(f"cannot read config {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise CliError(f"config {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise CliError(f"config root must be an object: {path}")
    return data


def _resolve_run_config_path(args: argparse.Namespace) -> Path | None:
    run_config = getattr(args, "run_config", None)
    if run_config:
        return Path(run_config)
    config_path = getattr(args, "config", None)
    if config_path:
        return Path(config_path)
    return None


def _pick(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    arg_name: str,
    cfg_key: str,
    default: Any,
) -> Any:
    v = getattr(args, arg_name, None)
    if v is not None:
        return v
    return cfg.get(cfg_key, default)


def _require(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    arg_name: str,
    cfg_key: str,
    flag_name: str,
) -> Any:
    v = _pick(args, cfg, arg_name, cfg_key, None)
    if v is None:
        raise CliError(
            f"{flag_name} or config key {cfg_key!r} is required"
        )
    return v


def _build_evaluation_config(
    args: argparse.Namespace,
) -> EvaluationConfig:
    defaults = BrainSettings()
    settings_defaults = args._settings.defaults
    return EvaluationConfig(BrainSettings(
        instrument_type=getattr(args, "instrument_type", None)
        or settings_defaults.instrument_type
        or defaults.instrument_type,
        region=getattr(args, "region", None)
        or settings_defaults.region
        or defaults.region,
        universe=getattr(args, "universe", None)
        or settings_defaults.universe
        or defaults.universe,
        delay=(
            getattr(args, "delay", None)
            if getattr(args, "delay", None) is not None
            else (
                settings_defaults.delay
                if settings_defaults.delay is not None
                else defaults.delay
            )
        ),
    ))


def _add_settings_flags(p: argparse.ArgumentParser) -> None:
    defaults = BrainSettings()
    p.add_argument(
        "--instrument-type",
        default=None,
        help=f"instrument type (default: {defaults.instrument_type})",
    )
    p.add_argument(
        "--region",
        default=None,
        help=f"region (default: {defaults.region})",
    )
    p.add_argument(
        "--universe",
        default=None,
        help=f"universe (default: {defaults.universe})",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=None,
        help=f"delay (default: {defaults.delay})",
    )


def _add_path_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--settings",
        default=None,
        help=(
            "path to the settings file (logging / defaults / "
            "credentials / cache). Overrides --config for those blocks."
        ),
    )
    p.add_argument(
        "--run-config",
        default=None,
        help=(
            "path to the run config file (run_id / pool_path / "
            "budget / selection). Overrides --config for those keys."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "combined config file (settings + run config). "
            "Backward-compatible alias for --settings + --run-config."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default=None,
        choices=_LOG_LEVELS,
        help="override the config file's logging.level",
    )
    common.add_argument(
        "--log-file",
        default=None,
        help="override the config file's logging.file",
    )

    parser = argparse.ArgumentParser(
        prog="wqgp",
        description="WorldQuant BRAIN alpha mining via genetic programming.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"wqgp {__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- build-pool ---
    p_bp = sub.add_parser(
        "build-pool",
        parents=[common],
        help="fetch fields + operator names from BRAIN and write a pool",
    )
    p_bp.add_argument("--signatures", required=True)
    p_bp.add_argument("--output", default="pool.json")
    p_bp.add_argument("--page-limit", type=int, default=50)
    p_bp.add_argument("--max-pages", type=int, default=200)
    p_bp.add_argument("--max-retries-on-429", type=int, default=3)
    p_bp.add_argument("--settings", default=None)
    p_bp.add_argument("--config", default=None)
    _add_settings_flags(p_bp)
    p_bp.set_defaults(handler=_cmd_build_pool)

    # --- run ---
    p_run = sub.add_parser(
        "run",
        parents=[common],
        help="run (or resume) a GP experiment",
    )
    _add_path_flags(p_run)
    p_run.add_argument("--pool", default=None)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--total-simulations", type=int, default=None)
    p_run.add_argument("--per-generation-simulations", type=int, default=None)
    p_run.add_argument("--wall-time-seconds", type=float, default=None)
    p_run.add_argument("--population-size", type=int, default=None)
    p_run.add_argument("--elite-count", type=int, default=None)
    p_run.add_argument("--tournament-size", type=int, default=None)
    p_run.add_argument("--crossover-rate", type=float, default=None)
    p_run.add_argument("--max-breeding-attempts", type=int, default=None)
    p_run.add_argument(
        "--mode", choices=("single", "pareto"), default=None,
    )
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--max-generations", type=int, default=None)
    p_run.add_argument("--checkpoint-root", default=None)
    p_run.add_argument("--evaluation-db", default=None)
    p_run.add_argument("--n-workers", type=int, default=None)
    p_run.add_argument("--max-concurrent", type=int, default=None)
    p_run.add_argument("--initial-population-size", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--no-diversity", action="store_true",
        help="disable diversity mechanism (A + A'). See D-149.",
    )
    p_run.add_argument(
        "--fitness-strategy",
        choices=("feasibility", "sharpe"),
        default=None,
        help="fitness function. Default 'feasibility'. See D-158.",
    )
    p_run.add_argument(
        "--no-cache", action="store_true",
        help="bypass the cross-run cache. See D-147.",
    )
    p_run.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help=(
            "number of seed ASTs to inject into gen 0. Default 3. "
            "Pass 0 to disable seed injection. See D-170."
        ),
    )
    p_run.add_argument(
        "--no-seed-injection", action="store_true",
        help=(
            "shortcut for --seed-count 0. Disables seed injection. "
            "See D-170."
        ),
    )
    p_run.add_argument(
        "--telegram", action="store_true",
        help="send lifecycle events to Telegram",
    )
    p_run.set_defaults(handler=_cmd_run)

    # --- signatures ---
    p_sig = sub.add_parser(
        "signatures",
        parents=[common],
        help="audit a signature table against BRAIN's operator list",
    )
    p_sig.add_argument("--signatures", required=True)
    p_sig.add_argument("--max-suggestions", type=int, default=20)
    p_sig.add_argument("--strict", action="store_true")
    p_sig.add_argument("--settings", default=None)
    p_sig.add_argument("--config", default=None)
    p_sig.set_defaults(handler=_cmd_signatures)

    # --- list-runs ---
    p_lr = sub.add_parser(
        "list-runs",
        parents=[common],
        help="list runs in the checkpoint root",
    )
    p_lr.add_argument(
        "--checkpoint-root", default="./checkpoints",
    )
    p_lr.set_defaults(handler=_cmd_list_runs)

    # --- cache ---
    p_cache = sub.add_parser(
        "cache",
        parents=[common],
        help="inspect and manage the cross-run evaluation cache",
    )
    cache_sub = p_cache.add_subparsers(
        dest="cache_command", metavar="CACHE_COMMAND"
    )

    p_cs = cache_sub.add_parser("stats")
    p_cs.add_argument("--cache-path", default=None)
    p_cs.add_argument("--settings", default=None)
    p_cs.add_argument("--config", default=None)
    p_cs.set_defaults(handler=_cmd_cache_stats)

    p_cc = cache_sub.add_parser("clear")
    p_cc.add_argument("--cache-path", default=None)
    p_cc.add_argument("--before", default=None)
    p_cc.add_argument("--yes", action="store_true")
    p_cc.add_argument("--settings", default=None)
    p_cc.add_argument("--config", default=None)
    p_cc.set_defaults(handler=_cmd_cache_clear)

    p_cache.set_defaults(handler=_cmd_cache_help)

    return parser


def _dry_run_evaluator(task: Task) -> TaskResult:
    digest = hashlib.sha256(task.fingerprint.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big", signed=False)
    fitness = (raw / float(1 << 64)) * 3.0
    return TaskResult(
        fingerprint=task.fingerprint,
        status=TaskStatus.SUCCESS,
        fitness=fitness,
        metrics=(("sharpe", fitness),),
    )


def _build_client_with_creds(
    creds: BrainCredentials,
    evaluation_config: EvaluationConfig,
    max_concurrent: int,
    alpha_registry: AlphaRegistry | None = None,
    fitness_fn=feasibility_fitness,
):
    from wqgp.brain.client import BrainClient
    return BrainClient(
        credentials=creds,
        settings=evaluation_config.settings,
        transport=RequestsTransport(),
        max_concurrent=max_concurrent,
        alpha_registry=alpha_registry,
        fitness_fn=fitness_fn,
    )


def _resolve_fitness_strategy(
    args: argparse.Namespace, cfg: dict[str, Any],
) -> str:
    raw = _pick(
        args, cfg, "fitness_strategy", "fitness_strategy", "feasibility",
    )
    if raw not in ("feasibility", "sharpe"):
        raise CliError(
            f"invalid fitness strategy: {raw!r}; "
            f"expected 'feasibility' or 'sharpe'"
        )
    return raw


def _fitness_fn_for(strategy: str):
    if strategy == "sharpe":
        return sharpe_only_fitness
    return feasibility_fitness


def _resolve_seed_count(
    args: argparse.Namespace, cfg: dict[str, Any],
) -> int:
    """Resolve seed_count from flag, then config, then default 3.

    `--no-seed-injection` forces 0 regardless of other sources.
    See D-170.
    """
    if getattr(args, "no_seed_injection", False):
        return 0
    raw = _pick(
        args, cfg, "seed_count", "seed_count", 3,
    )
    try:
        n = int(raw)
    except (TypeError, ValueError) as e:
        raise CliError(
            f"invalid seed_count: {raw!r}"
        ) from e
    if n < 0:
        raise CliError(f"seed_count must be >= 0, got {n}")
    return n


def _make_evaluator(
    args: argparse.Namespace,
    settings: Settings,
    evaluation_config: EvaluationConfig,
    fitness_strategy: str,
    alpha_registry: AlphaRegistry | None = None,
) -> Evaluator:
    if args.dry_run:
        log.warning("dry-run mode: BRAIN will not be contacted")
        return _dry_run_evaluator
    try:
        creds = _resolve_credentials(settings)
    except BrainAuthError as e:
        raise CliError(str(e)) from e

    max_concurrent = getattr(args, "max_concurrent", None)
    if max_concurrent is None:
        max_concurrent = settings.defaults.max_concurrent

    fitness_fn = _fitness_fn_for(fitness_strategy)
    log.info("fitness strategy: %s", fitness_strategy)

    return _build_client_with_creds(
        creds, evaluation_config, max_concurrent,
        alpha_registry=alpha_registry,
        fitness_fn=fitness_fn,
    )


def _make_notifier(args: argparse.Namespace) -> Notifier:
    if not args.telegram:
        return NullNotifier()
    try:
        creds = TelegramCredentials.from_env()
    except TelegramConfigError as e:
        raise CliError(str(e)) from e
    transport = _make_transport(args)
    return TelegramNotifier(credentials=creds, transport=transport)


def _make_cache(
    args: argparse.Namespace,
    settings: Settings,
) -> EvaluationCache | None:
    if getattr(args, "no_cache", False):
        log.info("cache disabled by --no-cache")
        return None
    if not settings.cache.enabled:
        log.info("cache disabled by settings.cache.enabled=false")
        return None
    if getattr(args, "dry_run", False):
        log.info("cache disabled (dry-run does not produce real results)")
        return None

    path = _resolve_cache_path(args, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    return EvaluationCache(path, ttl_days=settings.cache.ttl_days)


def _cmd_build_pool(args: argparse.Namespace) -> int:
    sig_path = Path(args.signatures)
    try:
        signatures = OperatorSignatureTable.from_file(sig_path)
    except SerializationError as e:
        raise CliError(f"cannot load signature table: {e}") from e

    try:
        creds = _resolve_credentials(args._settings)
    except BrainAuthError as e:
        raise CliError(str(e)) from e

    evaluation_config = _build_evaluation_config(args)
    settings = evaluation_config.settings
    transport = _make_transport(args)
    builder = PoolBuilder(
        credentials=creds,
        transport=transport,
        settings=settings,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        max_retries_on_429=args.max_retries_on_429,
    )
    try:
        result = builder.build(signatures)
    except (PoolBuilderError, PoolBuilderProtocolError) as e:
        raise CliError(f"failed to build pool: {e}") from e
    finally:
        builder.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.pool.save(out_path)

    log.info("wrote pool to %s", out_path)
    log.info(
        "settings: instrument_type=%s region=%s universe=%s delay=%d",
        settings.instrument_type, settings.region,
        settings.universe, settings.delay,
    )
    for k, v in result.summary().items():
        log.info("  %s: %s", k, v)
    return EXIT_OK


def _resolve_budget(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> Budget:
    b = cfg.get("budget", {})
    if not isinstance(b, dict):
        raise CliError("config 'budget' must be an object")

    total = _pick(args, b, "total_simulations", "total_simulations", None)
    if total is None:
        raise CliError(
            "--total-simulations or config budget.total_simulations "
            "is required"
        )
    per_gen = _pick(
        args, b, "per_generation_simulations",
        "per_generation_simulations", None,
    )
    wall = _pick(
        args, b, "wall_time_seconds", "wall_time_seconds", None,
    )
    return Budget(
        total_simulations=int(total),
        per_generation_simulations=(
            None if per_gen is None else int(per_gen)
        ),
        wall_time_seconds=(None if wall is None else float(wall)),
    )


def _resolve_selection(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> SelectionConfig:
    s = cfg.get("selection", {})
    if not isinstance(s, dict):
        raise CliError("config 'selection' must be an object")

    defaults = args._settings.defaults
    mode = _pick(args, s, "mode", "mode", "single")

    raw_objectives = s.get("pareto_objectives", ())
    if not isinstance(raw_objectives, (list, tuple)):
        raise CliError("selection.pareto_objectives must be a list")
    pareto_objectives: list[tuple[str, str]] = []
    for entry in raw_objectives:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
        ):
            raise CliError(
                "each pareto_objectives entry must be [metric, direction]"
            )
        pareto_objectives.append((str(entry[0]), str(entry[1])))

    diversity_enabled = bool(_pick(
        args, s, "diversity_enabled", "diversity_enabled", True,
    ))
    if getattr(args, "no_diversity", False):
        diversity_enabled = False

    return SelectionConfig(
        population_size=int(_pick(
            args, s, "population_size", "population_size",
            defaults.population_size,
        )),
        elite_count=int(_pick(
            args, s, "elite_count", "elite_count",
            defaults.elite_count,
        )),
        tournament_size=int(_pick(
            args, s, "tournament_size", "tournament_size",
            defaults.tournament_size,
        )),
        crossover_rate=float(_pick(
            args, s, "crossover_rate", "crossover_rate",
            defaults.crossover_rate,
        )),
        max_breeding_attempts=int(_pick(
            args, s, "max_breeding_attempts",
            "max_breeding_attempts", defaults.max_breeding_attempts,
        )),
        mode=str(mode),
        pareto_objectives=tuple(pareto_objectives),
        diversity_enabled=diversity_enabled,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(_resolve_run_config_path(args))
    settings = args._settings
    defaults = settings.defaults

    pool_raw = _pick(args, cfg, "pool", "pool_path", None)
    if pool_raw is None:
        raise CliError("--pool or config key 'pool_path' is required")
    pool_path = Path(pool_raw)
    try:
        pool = Pool.from_file(pool_path)
    except PoolError as e:
        raise CliError(f"cannot load pool: {e}") from e

    eval_db_raw = _pick(
        args, cfg, "evaluation_db", "evaluation_db", None
    )
    init_pop_raw = _pick(
        args, cfg, "initial_population_size",
        "initial_population_size", None,
    )

    evaluation_config = _build_evaluation_config(args)
    fitness_strategy = _resolve_fitness_strategy(args, cfg)
    seed_count = _resolve_seed_count(args, cfg)

    run_cfg = RunConfig(
        run_id=str(_require(args, cfg, "run_id", "run_id", "--run-id")),
        pool=pool,
        budget=_resolve_budget(args, cfg),
        selection=_resolve_selection(args, cfg),
        seed=int(_pick(args, cfg, "seed", "seed", defaults.seed)),
        evaluation_config=evaluation_config,
        fitness_strategy=fitness_strategy,
        seed_count=seed_count,
        max_generations=int(_pick(
            args, cfg, "max_generations", "max_generations",
            defaults.max_generations,
        )),
        checkpoint_root=Path(_pick(
            args, cfg, "checkpoint_root",
            "checkpoint_root", defaults.checkpoint_root,
        )),
        evaluation_db=(
            None if eval_db_raw is None else Path(eval_db_raw)
        ),
        n_workers=int(_pick(
            args, cfg, "n_workers", "n_workers", defaults.n_workers,
        )),
        initial_population_size=(
            None if init_pop_raw is None else int(init_pop_raw)
        ),
    )

    ckpt_store = CheckpointStore(run_cfg.checkpoint_root)
    try:
        latest = ckpt_store.load_latest(run_cfg.run_id)
    except CorruptCheckpointError as e:
        raise CliError(
            f"checkpoint for {run_cfg.run_id!r} is corrupt: {e}"
        ) from e
    if latest is not None:
        log.info(
            "resuming run %s from checkpoint generation %d",
            run_cfg.run_id, latest.generation,
        )
    else:
        log.info("starting new run %s", run_cfg.run_id)

    alpha_registry: AlphaRegistry | None = None
    if (
        not getattr(args, "no_cache", False)
        and settings.cache.enabled
        and not getattr(args, "dry_run", False)
    ):
        reg_path = run_cfg.checkpoint_root / "alphas.db"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        alpha_registry = AlphaRegistry(reg_path)

    evaluator = _make_evaluator(
        args, settings, evaluation_config,
        fitness_strategy=fitness_strategy,
        alpha_registry=alpha_registry,
    )
    notifier = _make_notifier(args)
    cache = _make_cache(args, settings)

    try:
        result = Coordinator(
            run_cfg, evaluator, notifier, cache=cache
        ).run()
    except CoordinatorError as e:
        raise CliError(f"coordinator error: {e}") from e
    finally:
        if cache is not None:
            cache.close()
        if alpha_registry is not None:
            alpha_registry.close()

    log.info(
        "run %s finished: reason=%s generation=%d total_simulations=%d "
        "cache_hits=%d",
        run_cfg.run_id, result.stopped_reason,
        result.generation, result.total_simulations_used,
        result.cache_hits,
    )
    return EXIT_OK


def _cmd_signatures(args: argparse.Namespace) -> int:
    sig_path = Path(args.signatures)
    try:
        table = OperatorSignatureTable.from_file(sig_path)
    except SerializationError as e:
        raise CliError(f"cannot load signature table: {e}") from e

    try:
        creds = _resolve_credentials(args._settings)
    except BrainAuthError as e:
        raise CliError(str(e)) from e

    transport = _make_transport(args)
    try:
        audit = audit_signatures(creds, transport, table)
    except (PoolBuilderError, PoolBuilderProtocolError) as e:
        raise CliError(f"failed to audit signatures: {e}") from e

    _print_audit(audit, max_suggestions=args.max_suggestions)
    if args.strict and not audit.is_complete():
        return EXIT_ERROR
    return EXIT_OK


def _print_audit(audit: SignatureAudit, *, max_suggestions: int) -> None:
    print(f"BRAIN operators: {audit.brain_operator_count}")
    print(f"Table operators: {audit.table_operator_count}")
    print(f"Common:          {len(audit.common)}")
    print()
    if audit.extra_in_table:
        print(f"In table but not on BRAIN ({len(audit.extra_in_table)}):")
        for name in audit.extra_in_table:
            print(f"  {name}")
        print()
    if audit.missing_from_table:
        print(f"Missing from table ({len(audit.missing_from_table)}):")
        for name in audit.missing_from_table:
            print(f"  {name}")
        print()
        print("Suggested skeletons:")
        print(format_skeletons(
            audit.missing_from_table, limit=max_suggestions,
        ))
    else:
        print("Table is complete.")


def _cmd_list_runs(args: argparse.Namespace) -> int:
    root = Path(args.checkpoint_root)
    if not root.exists():
        print(f"no checkpoint root at {root}", file=sys.stderr)
        return EXIT_OK
    store = CheckpointStore(root)
    runs = store.list_runs()
    if not runs:
        print(f"no runs in {root}", file=sys.stderr)
        return EXIT_OK
    print(f"{'run_id':<30} {'latest_gen':>12}")
    print("-" * 43)
    for run_id in runs:
        try:
            latest = store.load_latest(run_id)
            gen = "?" if latest is None else str(latest.generation)
        except CorruptCheckpointError:
            gen = "corrupt"
        print(f"{run_id:<30} {gen:>12}")
    return EXIT_OK


def _resolve_cache_path(
    args: argparse.Namespace,
    settings: Settings,
) -> Path:
    override = getattr(args, "cache_path", None)
    if override:
        return Path(override)
    if settings.cache.path:
        return Path(settings.cache.path)
    checkpoint_root = Path(settings.defaults.checkpoint_root)
    return checkpoint_root / "cache.db"


def _cmd_cache_help(args: argparse.Namespace) -> int:
    print("usage: wqgp cache {stats,clear}", file=sys.stderr)
    return EXIT_USAGE


def _cmd_cache_stats(args: argparse.Namespace) -> int:
    settings = args._settings
    path = _resolve_cache_path(args, settings)
    if not path.exists():
        print(f"no cache DB at {path}")
        print("(it will be created on the next run)")
        return EXIT_OK
    cache = EvaluationCache(path, ttl_days=settings.cache.ttl_days)
    try:
        s = cache.stats()
    finally:
        cache.close()
    print(f"cache DB:  {path}")
    print(f"ttl:       {settings.cache.ttl_days} days")
    print()
    print(f"total entries:   {s.total_entries}")
    print(f"  success:       {s.success_entries}")
    print(f"  failed:        {s.failed_entries}")
    print(f"  expired:       {s.expired_entries}")
    print()
    print(f"total hits:      {s.total_hits}")
    print(f"avg hits/entry:  {s.avg_hits_per_entry:.2f}")
    print()
    if s.oldest_created_at is not None:
        import datetime as _dt
        oldest = _dt.datetime.fromtimestamp(s.oldest_created_at)
        newest = _dt.datetime.fromtimestamp(s.newest_created_at)
        print(f"oldest entry:    {oldest.isoformat(timespec='seconds')}")
        print(f"newest entry:    {newest.isoformat(timespec='seconds')}")
    return EXIT_OK


def _cmd_cache_clear(args: argparse.Namespace) -> int:
    settings = args._settings
    path = _resolve_cache_path(args, settings)
    if not path.exists():
        print(f"no cache DB at {path}", file=sys.stderr)
        return EXIT_OK
    before_ts: float | None = None
    if args.before is not None:
        import datetime as _dt
        try:
            dt = _dt.datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            raise CliError(
                f"--before must be YYYY-MM-DD, got {args.before!r}"
            )
        before_ts = dt.timestamp()
    if not args.yes:
        if before_ts is None:
            prompt = f"Delete ALL entries in {path}? [y/N] "
        else:
            prompt = (
                f"Delete entries older than {args.before} in "
                f"{path}? [y/N] "
            )
        answer = input(prompt).strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return EXIT_OK
    cache = EvaluationCache(path, ttl_days=settings.cache.ttl_days)
    try:
        removed = cache.clear(before_ts=before_ts)
    finally:
        cache.close()
    if before_ts is None:
        print(f"deleted {removed} entries from {path}")
    else:
        print(
            f"deleted {removed} entries older than "
            f"{args.before} from {path}"
        )
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        settings = _load_settings(args)
        setup_logging(settings.logging)
        args._settings = settings
        log.debug("[cli] command=%s", args.command)
        return args.handler(args)
    except CliError as e:
        log.error("%s", e)
        return EXIT_ERROR
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        return EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001
        log.error("unexpected error: %s", e)
        log.debug("traceback:", exc_info=True)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())