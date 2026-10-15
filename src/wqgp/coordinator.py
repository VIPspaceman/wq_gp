"""End-to-end GP coordinator.

Design decisions:
- AST/Individual/Population serialization lives in gp_serialization.py.
  See D-065.
- Expression rendering lives in ast.py. See D-066.
- Checkpoints save the UNEVALUATED next-generation population. See D-067.
- Before submitting, existing evaluations are absorbed from the store.
  See D-068.
- One Executor for the whole run. See D-069.
- The loop ends on budget exhaustion, no-progress, or max_generations.
  See D-070.
- Notifications are optional; the default is NullNotifier. See D-098.
- Per-generation progress logs at INFO to "wqgp.gp". See D-115 / D-134.
- FAILED evaluations propagate to the Individual's status. See D-125.
- If a cache is provided, _evaluate consults it before submitting and
  _merge_evaluations writes every reusable result back to it. The
  run DB is still written in both cases. See D-146.
- RunConfig now carries an EvaluationConfig, the single source of
  truth for a simulation's identity. See D-138.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from wqgp._ids import validate_run_id
from wqgp.ast import render as render_ast
from wqgp.budget import Budget, BudgetTracker
from wqgp.cache import (
    CacheEntry,
    CacheStatus,
    EvaluationCache,
    classify_failure,
)
from wqgp.checkpoint import Checkpoint, CheckpointStore
from wqgp.crossover import Crossover
from wqgp.evaluation_config import EvaluationConfig
from wqgp.execution import Evaluator, Executor, Task
from wqgp.gp_serialization import (
    population_from_dict,
    population_to_dict,
)
from wqgp.mutation import Mutator
from wqgp.notification.events import (
    GenerationEndEvent,
    RunEndEvent,
    RunStartEvent,
)
from wqgp.notification.null import NullNotifier
from wqgp.notification.protocol import Notifier
from wqgp.persistence import EvaluationStatus, EvaluationStore
from wqgp.pool import Pool
from wqgp.population import (
    GenerationError,
    Individual,
    Population,
    RandomAstBuilder,
)
from wqgp.selection import SelectionConfig, next_generation
from wqgp.types import DataType


log = logging.getLogger("wqgp.gp")


class CoordinatorError(RuntimeError):
    """Raised on unrecoverable coordinator failures."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Frozen, validated configuration for one GP run."""
    run_id: str
    pool: Pool
    budget: Budget
    selection: SelectionConfig
    seed: int
    evaluation_config: EvaluationConfig
    fitness_strategy: str = "feasibility"
    max_generations: int = 10
    checkpoint_root: Path = Path("./checkpoints")
    evaluation_db: Path | None = None
    n_workers: int = 4
    initial_population_size: int | None = None
    builder_max_depth: int = 4
    builder_max_nodes: int = 25
    crossover_max_depth: int = 6
    crossover_max_nodes: int = 40

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        if self.max_generations < 1:
            raise ValueError("max_generations must be >= 1")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if (
            self.initial_population_size is not None
            and self.initial_population_size < 1
        ):
            raise ValueError(
                "initial_population_size must be >= 1 if set"
            )
        if self.builder_max_depth < 0:
            raise ValueError("builder_max_depth must be >= 0")
        if self.builder_max_nodes < 1:
            raise ValueError("builder_max_nodes must be >= 1")
        if self.crossover_max_depth < 1:
            raise ValueError("crossover_max_depth must be >= 1")
        if self.crossover_max_nodes < 1:
            raise ValueError("crossover_max_nodes must be >= 1")
        if self.crossover_max_nodes < 1:
            raise ValueError("crossover_max_nodes must be >= 1")
        if self.fitness_strategy not in ("feasibility", "sharpe"):
            raise ValueError(
                f"fitness_strategy must be 'feasibility' or 'sharpe', "
                f"got {self.fitness_strategy!r}"
            )


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a completed run."""
    generation: int
    population: Population
    total_simulations_used: int
    budget_exhausted: bool
    stopped_reason: str
    cache_hits: int = 0


class Coordinator:
    """Drives the GP loop end to end."""

    __slots__ = (
        "_config",
        "_evaluator",
        "_notifier",
        "_cache",
        "_store",
        "_ckpt_store",
        "_mutator",
        "_crossover",
        "_cache_hits",
    )

    def __init__(
        self,
        config: RunConfig,
        evaluator: Evaluator,
        notifier: Notifier | None = None,
        cache: EvaluationCache | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._notifier: Notifier = (
            notifier if notifier is not None else NullNotifier()
        )
        self._cache = cache
        self._cache_hits = 0

        if config.evaluation_db is not None:
            db_path = Path(config.evaluation_db)
        else:
            db_path = (
                Path(config.checkpoint_root) / "runs"
                / f"{config.run_id}.db"
            )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._store = EvaluationStore(db_path)

        self._ckpt_store = CheckpointStore(config.checkpoint_root)
        self._mutator = Mutator(config.pool)
        self._crossover = Crossover(
            config.pool,
            max_depth=config.crossover_max_depth,
            max_nodes=config.crossover_max_nodes,
        )

    # --- public ---------------------------------------------------------

    def run(self) -> RunResult:
        pop, tracker, rng, start_gen = self._setup()

        log.info(
            "[%s] run starting: generation=%d population=%d "
            "max_generations=%d cache=%s",
            self._config.run_id, start_gen, len(pop),
            self._config.max_generations,
            "on" if self._cache is not None else "off",
        )

        executor = Executor(
            run_id=self._config.run_id,
            store=self._store,
            evaluator=self._evaluator,
            tracker=tracker,
            n_workers=self._config.n_workers,
        )
        executor.start()

        stopped_reason = "max_generations"
        budget_exhausted = False
        last_generation = start_gen

        self._safe_notify(
            self._notifier.on_run_start,
            RunStartEvent(
                run_id=self._config.run_id,
                generation=start_gen,
                population_size=len(pop),
                resuming=start_gen > 0,
            ),
        )

        try:
            generation = start_gen
            while generation < self._config.max_generations:
                tracker.start_generation()
                last_generation = generation

                pop = self._merge_evaluations(pop)
                pending = sum(
                    1 for i in pop
                    if not i.is_evaluated()
                    and i.status is not EvaluationStatus.FAILED
                )
                log.info(
                    "[%s] gen %d: submitting %d individuals",
                    self._config.run_id, generation, pending,
                )
                self._evaluate(executor, pop, tracker)
                pop = self._merge_evaluations(pop)
                best = self._best_fitness(pop)
                best_str = "n/a" if best is None else f"{best:.4f}"
                log.info(
                    "[%s] gen %d: evaluated=%d best=%s sims_used=%d",
                    self._config.run_id, generation,
                    len(pop.evaluated()), best_str,
                    tracker.total_simulations_used,
                )

                if tracker.is_exhausted():
                    stopped_reason = "budget_exhausted"
                    budget_exhausted = True
                elif len(pop.evaluated()) == 0:
                    stopped_reason = "no_progress"
                elif generation + 1 >= self._config.max_generations:
                    stopped_reason = "max_generations"

                self._safe_notify(
                    self._notifier.on_generation_end,
                    GenerationEndEvent(
                        run_id=self._config.run_id,
                        generation=generation,
                        evaluated_count=len(pop.evaluated()),
                        best_fitness=self._best_fitness(pop),
                        total_simulations_used=(
                            tracker.total_simulations_used
                        ),
                        stopped_reason=(
                            None if stopped_reason == "max_generations"
                            and generation + 1 < self._config.max_generations
                            else stopped_reason
                        ),
                    ),
                )

                if stopped_reason != "max_generations":
                    break
                if generation + 1 >= self._config.max_generations:
                    break

                pop = next_generation(
                    pop,
                    self._mutator,
                    self._crossover,
                    self._config.selection,
                    rng,
                )
                generation += 1
                self._save_checkpoint(pop, tracker, rng, generation)
        finally:
            executor.shutdown()

        result = RunResult(
            generation=last_generation,
            population=pop,
            total_simulations_used=tracker.total_simulations_used,
            budget_exhausted=budget_exhausted,
            stopped_reason=stopped_reason,
            cache_hits=self._cache_hits,
        )

        try:
            self._log_summary(result)
        finally:
            self._store.close()

        self._safe_notify(
            self._notifier.on_run_end,
            RunEndEvent(
                run_id=self._config.run_id,
                generation=result.generation,
                stopped_reason=result.stopped_reason,
                total_simulations_used=result.total_simulations_used,
                budget_exhausted=result.budget_exhausted,
                population_size=len(result.population),
            ),
        )
        return result

    # --- setup ----------------------------------------------------------

    def _setup(
        self,
    ) -> tuple[Population, BudgetTracker, random.Random, int]:
        existing = self._ckpt_store.load_latest(self._config.run_id)
        if existing is None:
            return self._initial_setup()
        log.info(
            "[%s] resuming from checkpoint generation %d",
            self._config.run_id, existing.generation,
        )
        return self._resume_setup(existing)

    def _initial_setup(
        self,
    ) -> tuple[Population, BudgetTracker, random.Random, int]:
        rng = random.Random(self._config.seed)
        tracker = BudgetTracker(self._config.budget)

        size = (
            self._config.initial_population_size
            if self._config.initial_population_size is not None
            else self._config.selection.population_size
        )
        builder = RandomAstBuilder(
            self._config.pool,
            max_depth=self._config.builder_max_depth,
            max_nodes=self._config.builder_max_nodes,
        )

        individuals: list[Individual] = []
        seen_fps: set[str] = set()
        attempts = 0
        generation_failures = 0
        max_attempts = size * 20
        while len(individuals) < size and attempts < max_attempts:
            attempts += 1
            try:
                ast = builder.build(DataType.MATRIX, rng)
            except GenerationError:
                generation_failures += 1
                continue
            fp = ast.fingerprint()
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            individuals.append(Individual.from_ast(ast, generation=0))

        if not individuals:
            raise CoordinatorError(
                f"could not generate any initial individuals "
                f"after {max_attempts} attempts"
            )

        if len(individuals) < size:
            log.warning(
                "[%s] initial population short: target=%d actual=%d "
                "attempts=%d generation_failures=%d duplicates=%d",
                self._config.run_id, size, len(individuals),
                attempts, generation_failures,
                attempts - generation_failures - len(individuals),
            )
        else:
            log.info(
                "[%s] initial population: %d individuals in %d attempts",
                self._config.run_id, len(individuals), attempts,
            )

        pop = Population(individuals)
        self._save_checkpoint(pop, tracker, rng, generation=0)
        return pop, tracker, rng, 0

    def _resume_setup(
        self, ckpt: Checkpoint
    ) -> tuple[Population, BudgetTracker, random.Random, int]:
        pop = self._load_population_from_checkpoint(ckpt)

        snap = ckpt.budget
        used = snap.total_simulations_used
        limit = self._config.budget.total_simulations

        # BudgetTracker raises a bare ValueError when `used > limit`,
        # which surfaces as "unexpected error" in the CLI. Catch it
        # here and raise a CoordinatorError with a message the user
        # can act on. See D-162.
        if used > limit:
            raise CoordinatorError(
                f"cannot resume run {self._config.run_id!r} from "
                f"generation {ckpt.generation}: it has already used "
                f"{used} simulations, but the new budget only allows "
                f"{limit}. Increase --total-simulations to at least "
                f"{used}, or delete the checkpoint at "
                f"{self._config.checkpoint_root}/runs/"
                f"{self._config.run_id}/ to start fresh."
            )

        tracker = BudgetTracker(
            self._config.budget,
            elapsed_offset_seconds=snap.elapsed_seconds,
            total_simulations_used=used,
            generation_simulations_used=0,
        )
        rng = ckpt.restore_rng()
        start_gen = ckpt.generation
        return pop, tracker, rng, start_gen

    def _load_population_from_checkpoint(
        self, ckpt: Checkpoint
    ) -> Population:
        raw = ckpt.payload.get("population")
        if raw is None:
            raise CoordinatorError(
                f"checkpoint {ckpt.generation} has no 'population' payload"
            )
        try:
            return population_from_dict(
                raw,
                field_lookup=self._config.pool.fields.by_id,
                operator_lookup=self._config.pool.operators.by_name,
            )
        except Exception as e:
            raise CoordinatorError(
                f"failed to load population from checkpoint "
                f"{ckpt.generation}: {e}"
            ) from e

    # --- evaluation -----------------------------------------------------

    def _evaluate(
        self,
        executor: Executor,
        pop: Population,
        tracker: BudgetTracker,
    ) -> int:
        submitted = 0
        cache_hits_this_gen = 0

        for ind in pop:
            if ind.is_evaluated():
                continue
            if ind.status is EvaluationStatus.FAILED:
                continue

            if self._cache is not None:
                entry = self._lookup_cache(ind)
                if entry is not None:
                    self._record_cache_hit(ind, entry)
                    cache_hits_this_gen += 1
                    self._cache_hits += 1
                    continue

            task = Task(
                fingerprint=ind.fingerprint,
                expression=render_ast(ind.ast),
                generation=ind.generation,
            )
            if not executor.submit(task):
                break
            submitted += 1

        if submitted > 0:
            executor.run_until_drained()

        if cache_hits_this_gen > 0:
            log.info(
                "[%s] cache: %d hits this generation "
                "(no BRAIN submissions)",
                self._config.run_id, cache_hits_this_gen,
            )
        return submitted

    def _lookup_cache(self, ind: Individual) -> CacheEntry | None:
        assert self._cache is not None
        key = self._config.evaluation_config.cache_key(
            ind.fingerprint,
            self._config.fitness_strategy,
        )
        return self._cache.get(key)

    def _record_cache_hit(
        self, ind: Individual, entry: CacheEntry
    ) -> None:
        from wqgp.persistence import EvaluationRecord

        expression = render_ast(ind.ast)
        if entry.status is CacheStatus.SUCCESS:
            rec = EvaluationRecord(
                run_id=self._config.run_id,
                generation=ind.generation,
                fingerprint=ind.fingerprint,
                expression=expression,
                status=EvaluationStatus.SUCCESS,
                fitness=entry.fitness,
                metrics=entry.metrics,
            )
        else:
            rec = EvaluationRecord(
                run_id=self._config.run_id,
                generation=ind.generation,
                fingerprint=ind.fingerprint,
                expression=expression,
                status=EvaluationStatus.FAILED,
                error=entry.error,
            )
        self._store.record(rec)

    def _merge_evaluations(self, pop: Population) -> Population:
        updated: list[Individual] = []
        for ind in pop:
            if ind.is_evaluated():
                updated.append(ind)
                continue
            if ind.status is EvaluationStatus.FAILED:
                updated.append(ind)
                continue
            rec = self._store.get(self._config.run_id, ind.fingerprint)
            if rec is None:
                updated.append(ind)
                continue
            if rec.status is EvaluationStatus.SUCCESS:
                updated.append(
                    ind.with_evaluation(
                        fitness=rec.fitness,  # type: ignore[arg-type]
                        metrics=rec.metrics,
                    )
                )
                self._cache_record(rec)
                continue
            if rec.status is EvaluationStatus.FAILED:
                updated.append(ind.with_status(EvaluationStatus.FAILED))
                self._cache_record(rec)
                continue
            updated.append(ind)
        return Population(updated)

    def _cache_record(self, rec) -> None:
        if self._cache is None:
            return

        cfg = self._config.evaluation_config
        cache_key = cfg.cache_key(
            rec.fingerprint,
            self._config.fitness_strategy,
        )
        created_at = rec.created_at if rec.created_at > 0 else time.time()

        if rec.status is EvaluationStatus.SUCCESS:
            if rec.fitness is None:
                return
            self._cache.put(CacheEntry(
                cache_key=cache_key,
                fingerprint=rec.fingerprint,
                settings_hash=cfg.settings_hash(),
                settings_json=json.dumps(
                    cfg.to_api_dict(), sort_keys=True,
                ),
                status=CacheStatus.SUCCESS,
                fitness=rec.fitness,
                metrics=rec.metrics,
                created_at=created_at,
            ))
            return

        if rec.status is EvaluationStatus.FAILED:
            error_type = classify_failure(rec.error)
            if error_type is None:
                return
            self._cache.put(CacheEntry(
                cache_key=cache_key,
                fingerprint=rec.fingerprint,
                settings_hash=cfg.settings_hash(),
                settings_json=json.dumps(
                    cfg.to_api_dict(), sort_keys=True,
                ),
                status=CacheStatus.FAILED,
                error=rec.error,
                error_type=error_type,
                created_at=created_at,
            ))

    # --- helpers --------------------------------------------------------

    def _best_fitness(self, pop: Population) -> float | None:
        best: float | None = None
        for ind in pop:
            if ind.fitness is None:
                continue
            if best is None or ind.fitness > best:
                best = ind.fitness
        return best

    def _log_summary(self, result: RunResult) -> None:
        rid = self._config.run_id
        counts = self._store.count_by_status(rid)
        success = counts.get("SUCCESS", 0)
        failed = counts.get("FAILED", 0)
        total = success + failed
        rate = (success / total * 100.0) if total else 0.0

        log.info(
            "[%s] run finished: reason=%s generations=%d "
            "total_sims=%d success=%d failed=%d rate=%.1f%% "
            "cache_hits=%d",
            rid, result.stopped_reason, result.generation,
            result.total_simulations_used, success, failed, rate,
            result.cache_hits,
        )

        top = self._store.top_by_fitness(rid, 3)
        for i, rec in enumerate(top, start=1):
            to = dict(rec.metrics).get("turnover")
            to_str = f"{to:.4f}" if to is not None else "?"
            expr = _truncate(rec.expression, 100)
            log.info(
                "[%s]   top %d: sharpe=%.4f turnover=%s expr=%s",
                rid, i, rec.fitness, to_str, expr,
            )

    def _safe_notify(self, fn, event) -> None:
        try:
            fn(event)
        except Exception:  # noqa: BLE001 -- D-095
            log.warning(
                "notifier raised; continuing without notification",
                exc_info=True,
            )

    # --- checkpoint -----------------------------------------------------

    def _save_checkpoint(
        self,
        pop: Population,
        tracker: BudgetTracker,
        rng: random.Random,
        generation: int,
    ) -> None:
        payload = {"population": population_to_dict(pop)}
        ckpt = Checkpoint.capture(
            run_id=self._config.run_id,
            generation=generation,
            budget=tracker.snapshot(),
            rng=rng,
            payload=payload,
            note=f"start of generation {generation}",
        )
        self._ckpt_store.save(ckpt)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."