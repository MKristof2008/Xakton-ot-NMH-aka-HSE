"""Компонент 4 — Orchestrator: цикл «сбор -> анализ -> дообучение» 24/7.

    python -m autopilot run           # бесконечный цикл
    python -m autopilot run --once    # один цикл (проверка)

Цикл:
1. Collector обходит источники и пишет новые материалы в SQLite.
2. Analyzer считает P(фейк) активной локальной моделью (быстро, без сети).
3. Verifier проверяет спорные материалы по базе знаний и ставит метки
   с обоснованием (см. verifier.py).
4. Trainer дообучает модель в отдельном процессе, когда накопилось N новых
   подтверждённых меток, и подменяет рабочую версию только при росте метрик.

Ошибка любого шага логируется и не останавливает цикл; Ctrl+C завершает
работу после текущего шага.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from .collector import Collector
from .db import Database
from .logsetup import get_logger, setup_logging
from .model import ModelRegistry
from .settings import (AppConfig, LOGS_DIR, REPO_DIR, ensure_dirs,
                       llm_config_status, load_llm_env)
from .trainer import Trainer
from .verifier import Verifier

log = get_logger("orchestrator")

LOCK_PATH = LOGS_DIR / "autopilot.lock"


@dataclass
class CycleReport:
    started_at: float
    collected: int = 0
    scored: int = 0
    fake: int = 0
    real: int = 0
    uncertain: int = 0
    trained: bool = False
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class Orchestrator:
    def __init__(self, cfg: AppConfig):
        ensure_dirs()
        load_llm_env()
        setup_logging(cfg.log_level, LOGS_DIR, cfg.log_to_file)

        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.registry = ModelRegistry(self.db)
        self.registry.load()
        self.collector = Collector(self.db, cfg.collector, cfg.source_specs)
        self.verifier = Verifier(self.db, cfg.verifier, self.registry)
        self.trainer = Trainer(self.db, cfg.trainer, self.registry)
        self._stop = asyncio.Event()
        self._last_housekeeping = 0.0

    # -- шаги ------------------------------------------------------------
    async def collect(self) -> int:
        reports = await self.collector.collect_once()
        return sum(r.inserted for r in reports)

    def score_new(self, limit: int = 2000) -> int:
        """Локальная модель: быстрая оценка всех новых материалов."""
        rows = self.db.pending_for_scoring(limit)
        version = self.registry.version
        scored = 0
        for row in rows:
            prob = self.registry.predict(row["text"])
            self.db.set_score(int(row["id"]), prob, version)
            scored += 1
        return scored

    async def verify(self) -> dict[str, int]:
        return await self.verifier.process_pending()

    def maybe_train(self) -> bool:
        if not self.trainer.should_retrain():
            return False
        log.info("запускаю дообучение в отдельном процессе")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "autopilot", "train"],
                cwd=str(REPO_DIR), env=env, capture_output=True, text=True,
                timeout=3600, encoding="utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("дообучение не запустилось: %s", exc)
            return False
        tail = (proc.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            log.info("trainer | %s", line)
        if proc.returncode != 0:
            log.warning("trainer завершился с кодом %s: %s", proc.returncode,
                        (proc.stderr or "")[-400:])
            return False
        self.registry.load(force=True)
        return True

    def housekeeping(self) -> None:
        if time.time() - self._last_housekeeping < 3600:
            return
        self._last_housekeeping = time.time()
        archived = self.db.archive_stale_uncertain(self.cfg.verifier.uncertain_ttl_hours)
        if archived:
            log.info("в архив ушло %d материалов: TTL %.0f ч без решения",
                     archived, self.cfg.verifier.uncertain_ttl_hours)
        removed = self.db.purge_old(self.cfg.retention_days)
        if removed:
            log.info("удалено старых неразмеченных материалов: %d", removed)

    # -- цикл ------------------------------------------------------------
    async def run_cycle(self) -> CycleReport:
        report = CycleReport(started_at=time.time())
        log.info("=== цикл: сбор ===")
        try:
            report.collected = await self.collect()
            log.info("новых материалов: %d", report.collected)
        except Exception as exc:
            log.exception("сбор упал: %s", exc)
            report.errors.append(f"collect: {exc}")

        log.info("=== цикл: скоринг локальной моделью ===")
        try:
            report.scored = self.score_new()
            log.info("оценено моделью: %d (версия %s)", report.scored, self.registry.version)
        except Exception as exc:
            log.exception("скоринг упал: %s", exc)
            report.errors.append(f"score: {exc}")

        log.info("=== цикл: проверка и разметка ===")
        try:
            counters = await self.verify()
            report.fake = counters.get("fake", 0)
            report.real = counters.get("real", 0)
            report.uncertain = counters.get("uncertain", 0)
        except Exception as exc:
            log.exception("проверка упала: %s", exc)
            report.errors.append(f"verify: {exc}")

        log.info("=== цикл: дообучение ===")
        try:
            report.trained = self.maybe_train()
        except Exception as exc:
            log.exception("обучение упало: %s", exc)
            report.errors.append(f"train: {exc}")

        try:
            self.housekeeping()
        except Exception as exc:
            log.debug("обслуживание не удалось: %s", exc)

        report.duration_s = round(time.time() - report.started_at, 1)
        log.info("=== цикл завершён за %.1f с: собрано %d, проверено fake=%d real=%d "
                 "uncertain=%d, обучение=%s ===",
                 report.duration_s, report.collected, report.fake, report.real,
                 report.uncertain, "да" if report.trained else "нет")
        return report

    async def run_forever(self) -> None:
        interval = max(60, int(self.cfg.collector.interval_s))
        log.info("автопилот запущен: интервал %d с, источников %d, модель %s",
                 interval, len(self.collector.sources), self.registry.version)
        llm = llm_config_status()
        if not llm["has_key"]:
            log.warning("ключ модели не найден (%s): работают только "
                        "детерминированные проверки, меток будет мало",
                        llm["env_file"])
        else:
            log.info("модель: %s @ %s", llm["model"], llm["base_url"])
        log.info("остановка: Ctrl+C")
        while not self._stop.is_set():
            try:
                await self.run_cycle()
            except Exception as exc:  # цикл не должен умирать
                log.exception("непредвиденная ошибка цикла: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
        log.info("автопилот остановлен")
        await self.aclose()

    def request_stop(self) -> None:
        self._stop.set()

    async def aclose(self) -> None:
        await self.collector.close()


def install_signal_handlers(orchestrator: Orchestrator,
                            loop: asyncio.AbstractEventLoop) -> None:
    def handler(*_args):
        log.info("получен сигнал остановки")
        orchestrator.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handler)
        except (NotImplementedError, RuntimeError, AttributeError):
            try:
                signal.signal(sig, lambda *_: orchestrator.request_stop())
            except (ValueError, OSError):
                pass


async def run_once(cfg: AppConfig) -> CycleReport:
    orchestrator = Orchestrator(cfg)
    try:
        return await orchestrator.run_cycle()
    finally:
        await orchestrator.aclose()


async def run_forever(cfg: AppConfig) -> None:
    orchestrator = Orchestrator(cfg)
    install_signal_handlers(orchestrator, asyncio.get_running_loop())
    await orchestrator.run_forever()
