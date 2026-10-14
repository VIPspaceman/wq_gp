"""Command-line interface for wqgp.

Design decisions:
- Subparsers via argparse. See D-087.
- Config is JSON, flag > config > default. See D-088.
- Secrets come ONLY from environment variables. See D-089.
- SIGINT propagates; exit code 130. See D-090.
- Structured logging to stderr; --log-level controls verbosity. See
  D-091.
- --dry-run uses a deterministic evaluator. See D-092.
- --telegram opts into Telegram notifications. See D-097.
- `signatures` audits a signature table; never writes it. See D-102.
- `build-pool` accepts simulation settings flags. See D-107.
- Logs go to the "wqgp.cli" channel. See D-134.
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
from wqgp.brain.client import (
    BrainAuthError,
    BrainCredentials,
    BrainSettings,
    build_default_client,
)
from wqgp.brain.pool_builder import (
    OperatorSignatureTable,
    PoolBuilder,
    PoolBuilderError,
    PoolBuilderProtocolError,
)
from wqgp.brain.transport import HttpTransport, RequestsTransport
from wqgp.budget import Budget
from wqgp.checkpoint import CheckpointStore, CorruptCheckpointError
from wqgp.coordinator import Coordinator, CoordinatorError, RunConfig
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
from wqgp.settings import Settings
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


# --- user-facing error ------------------------------------------------------

class CliError(RuntimeError):
    """Raised for expected user-facing errors."""


# --- logging ---------------------------------------------------------------

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _load_settings(args: argparse.Namespace) -> Settings:
    """Load the settings file, if --config was given.

    --config is defined only on the `run` subcommand. See D-120.
    """
    config_path = getattr(args, "config", None)
    try:
        settings = Settings.load(config_path)
    except (FileNotFoundError, ValueError) as e:
        raise CliError(str(e)) from e

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
            raw=settings.raw,
        )
    return settings


# --- transport factory (patched in tests) -----------------------------------

def _make_transport(args: argparse.Namespace) -> HttpTransport:
    """Return the HTTP transport to use. Tests replace this function."""
    return RequestsTransport()


# --- config -----------------------------------------------------------------

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


def _settings_from_args(args: argparse.Namespace) -> BrainSettings:
    defaults = BrainSettings()
    return BrainSettings(
        instrument_type=getattr(args, "instrument_type", None)
        or defaults.instrument_type,
        region=getattr(args, "region", None) or defaults.region,
        universe=getattr(args, "universe", None) or defaults.universe,
        delay=getattr(args, "delay", None)
        if getattr(args, "delay", None) is not None
        else defaults.delay,
    )


# --- argument parser --------------------------------------------------------

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


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        default=None,
        choices=_LOG_LEVELS,
        help=(
            "override the config file's logging.level "
            "(default: value from config, or INFO)"
        ),
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

    p_bp = sub.add_parser(
        "build-pool",
        parents=[common],
        help="fetch fields + operator names from BRAIN and write a pool",
    )
    p_bp.add_argument(
        "--signatures", required=True,
        help="path to the operator signature table (JSON)",
    )
    p_bp.add_argument(
        "--output", default="pool.json",
        help="where to write the pool (default: pool.json)",
    )
    p_bp.add_argument(
        "--page-limit", type=int, default=50,
        help=(
            "page size for paginated endpoints (default: 50). "
            "BRAIN rejects large page sizes with HTTP 400."
        ),
    )
    p_bp.add_argument(
        "--max-pages", type=int, default=200,
        help="hard cap on pages per endpoint (default: 200)",
    )
    p_bp.add_argument(
        "--max-retries-on-429", type=int, default=3,
        help="how many times to retry a rate-limited request (default: 3)",
    )
    _add_settings_flags(p_bp)
    p_bp.set_defaults(handler=_cmd_build_pool)

    p_run = sub.add_parser(
        "run",
        parents=[common],
        help="run (or resume) a GP experiment",
    )
    p_run.add_argument("--config", default=None, help="JSON config file")
    p_run.add_argument("--pool", default=None, help="pool JSON path")
    p_run.add_argument("--run-id", default=None, help="experiment id")
    p_run.add_argument("--total-simulations", type=int, default=None)
    p_run.add_argument("--per-generation-simulations", type=int, default=None)
    p_run.add_argument("--wall-time-seconds", type=float, default=None)
    p_run.add_argument("--population-size", type=int, default=None)
    p_run.add_argument("--elite-count", type=int, default=None)
    p_run.add_argument("--tournament-size", type=int, default=None)
    p_run.add_argument("--crossover-rate", type=float, default=None)
    p_run.add_argument("--max-breeding-attempts", type=int, default=None)
    p_run.add_argument(
        "--mode",
        choices=("single", "pareto"),
        default=None,
        help=(
            "selection mode: 'single' (sharpe only) or 'pareto' "
            "(multi-objective). Default from config, else 'single'."
        ),
    )
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--max-generations", type=int, default=None)
    p_run.add_argument("--checkpoint-root", default=None)
    p_run.add_argument("--evaluation-db", default=None)
    p_run.add_argument("--n-workers", type=int, default=None)
    p_run.add_argument("--initial-population-size", type=int, default=None)
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="use a deterministic fake evaluator; do not contact BRAIN",
    )
    p_run.add_argument(
        "--telegram", action="store_true",
        help=(
            "send lifecycle events to Telegram "
            "(requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
        ),
    )
    p_run.set_defaults(handler=_cmd_run)

    p_sig = sub.add_parser(
        "signatures",
        parents=[common],
        help="audit a signature table against BRAIN's operator list",
    )
    p_sig.add_argument(
        "--signatures", required=True,
        help="path to the signature table to audit",
    )
    p_sig.add_argument(
        "--max-suggestions", type=int, default=20,
        help="how many paste-ready skeletons to print (default: 20)",
    )
    p_sig.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if the table is missing operators",
    )
    p_sig.set_defaults(handler=_cmd_signatures)

    p_lr = sub.add_parser(
        "list-runs",
        parents=[common],
        help="list runs in the checkpoint root",
    )
    p_lr.add_argument(
        "--checkpoint-root", default="./checkpoints",
        help="checkpoint root (default: ./checkpoints)",
    )
    p_lr.set_defaults(handler=_cmd_list_runs)

    return parser


# --- evaluators -------------------------------------------------------------

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


def _make_evaluator(args: argparse.Namespace) -> Evaluator:
    if args.dry_run:
        log.warning("dry-run mode: BRAIN will not be contacted")
        return _dry_run_evaluator
    try:
        return build_default_client()
    except BrainAuthError as e:
        raise CliError(str(e)) from e


def _make_notifier(args: argparse.Namespace) -> Notifier:
    if not args.telegram:
        return NullNotifier()
    try:
        creds = TelegramCredentials.from_env()
    except TelegramConfigError as e:
        raise CliError(str(e)) from e
    transport = _make_transport(args)
    return TelegramNotifier(credentials=creds, transport=transport)


# --- commands ---------------------------------------------------------------

def _cmd_build_pool(args: argparse.Namespace) -> int:
    sig_path = Path(args.signatures)
    try:
        signatures = OperatorSignatureTable.from_file(sig_path)
    except SerializationError as e:
        raise CliError(f"cannot load signature table: {e}") from e

    try:
        creds = BrainCredentials.from_env()
    except BrainAuthError as e:
        raise CliError(str(e)) from e

    settings = _settings_from_args(args)
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

    return SelectionConfig(
        population_size=int(
            _pick(args, s, "population_size", "population_size", 20)
        ),
        elite_count=int(
            _pick(args, s, "elite_count", "elite_count", 2)
        ),
        tournament_size=int(
            _pick(args, s, "tournament_size", "tournament_size", 3)
        ),
        crossover_rate=float(
            _pick(args, s, "crossover_rate", "crossover_rate", 0.7)
        ),
        max_breeding_attempts=int(
            _pick(
                args, s, "max_breeding_attempts",
                "max_breeding_attempts", 200,
            )
        ),
        mode=str(mode),
        pareto_objectives=tuple(pareto_objectives),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config) if args.config else None)
    defaults = args._settings.defaults

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

    run_cfg = RunConfig(
        run_id=str(_require(args, cfg, "run_id", "run_id", "--run-id")),
        pool=pool,
        budget=_resolve_budget(args, cfg),
        selection=_resolve_selection(args, cfg),
        seed=int(_pick(args, cfg, "seed", "seed", defaults.seed)),
        max_generations=int(
            _pick(
                args, cfg, "max_generations", "max_generations",
                defaults.max_generations,
            )
        ),
        checkpoint_root=Path(
            _pick(
                args, cfg, "checkpoint_root",
                "checkpoint_root", defaults.checkpoint_root,
            )
        ),
        evaluation_db=(
            None if eval_db_raw is None else Path(eval_db_raw)
        ),
        n_workers=int(
            _pick(args, cfg, "n_workers", "n_workers", defaults.n_workers)
        ),
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

    evaluator = _make_evaluator(args)
    notifier = _make_notifier(args)

    try:
        result = Coordinator(run_cfg, evaluator, notifier).run()
    except CoordinatorError as e:
        raise CliError(f"coordinator error: {e}") from e

    log.info(
        "run %s finished: reason=%s generation=%d total_simulations=%d",
        run_cfg.run_id, result.stopped_reason,
        result.generation, result.total_simulations_used,
    )
    return EXIT_OK


def _cmd_signatures(args: argparse.Namespace) -> int:
    sig_path = Path(args.signatures)
    try:
        table = OperatorSignatureTable.from_file(sig_path)
    except SerializationError as e:
        raise CliError(f"cannot load signature table: {e}") from e

    try:
        creds = BrainCredentials.from_env()
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
        print(
            f"In table but not on BRAIN "
            f"({len(audit.extra_in_table)}):"
        )
        for name in audit.extra_in_table:
            print(f"  {name}")
        print()
        print("These may have been renamed or removed by BRAIN.")
        print()

    if audit.missing_from_table:
        print(
            f"Missing from table "
            f"({len(audit.missing_from_table)}):"
        )
        for name in audit.missing_from_table:
            print(f"  {name}")
        print()
        print("Suggested skeletons (paste into signatures.json):")
        print(format_skeletons(
            audit.missing_from_table,
            limit=max_suggestions,
        ))
        print()
        print(
            'Note: type placeholder "MATRIX" is a guess. '
            "Verify each signature"
        )
        print("against BRAIN's documentation before use.")
    else:
        print(
            "Table is complete: every BRAIN operator has a signature."
        )


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


# --- entry point ------------------------------------------------------------

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