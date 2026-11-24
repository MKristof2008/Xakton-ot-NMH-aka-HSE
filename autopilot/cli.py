"""CLI автопилота BAYKAM.

    python -m autopilot run [--once] [--no-llm]
    python -m autopilot collect
    python -m autopilot verify [--limit N]
    python -m autopilot train [--no-en]
    python -m autopilot status
    python -m autopilot items [--label fake|real|uncertain] [--limit N]
    python -m autopilot models [--promote VERSION]
    python -m autopilot sources
    python -m autopilot export --out dataset.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from .collector import Collector
from .db import Database
from .logsetup import get_logger, setup_logging
from .model import ModelRegistry
from .orchestrator import run_forever, run_once
from .settings import AppConfig, LOGS_DIR, MODELS_DIR, ensure_dirs, load_config, load_llm_env
from .statusline import print_report
from .trainer import Trainer
from .verifier import Verifier

log = get_logger("cli")


def _prepare(cfg_path: str | None, *, quiet: bool = False) -> AppConfig:
    ensure_dirs()
    load_llm_env()
    cfg = load_config(Path(cfg_path) if cfg_path else None)
    if not quiet:
        setup_logging(cfg.log_level, LOGS_DIR, cfg.log_to_file)
    return cfg


def cmd_run(args) -> int:
    cfg = _prepare(args.config)
    if args.no_llm:
        cfg.verifier.llm_enabled = False
    if args.once:
        report = asyncio.run(run_once(cfg))
        print(json.dumps({
            "collected": report.collected, "scored": report.scored,
            "fake": report.fake, "real": report.real, "uncertain": report.uncertain,
            "trained": report.trained, "errors": report.errors,
            "duration_s": report.duration_s,
        }, ensure_ascii=False, indent=2))
        return 0 if not report.errors else 1
    try:
        asyncio.run(run_forever(cfg))
    except KeyboardInterrupt:
        log.info("остановлено пользователем")
    return 0


def cmd_collect(args) -> int:
    cfg = _prepare(args.config)

    async def go():
        collector = Collector(Database(cfg.db_path), cfg.collector, cfg.source_specs)
        try:
            return await collector.collect_once()
        finally:
            await collector.close()

    reports = asyncio.run(go())
    for r in reports:
        print(f"{r.source_id:<16} ok={str(r.ok):<5} новых={r.inserted:<3} "
              f"дублей={r.duplicates:<3} ошибка={r.error or '-'}")
    print(f"\nвсего новых материалов: {sum(r.inserted for r in reports)}")
    return 0


def cmd_verify(args) -> int:
    cfg = _prepare(args.config)
    db = Database(cfg.db_path)
    registry = ModelRegistry(db)
    registry.load()
    verifier = Verifier(db, cfg.verifier, registry)
    counters = asyncio.run(verifier.process_pending(args.limit))
    print(json.dumps(counters, ensure_ascii=False))
    return 0


def cmd_train(args) -> int:
    cfg = _prepare(args.config)
    db = Database(cfg.db_path)
    registry = ModelRegistry(db)
    registry.load()
    trainer = Trainer(db, cfg.trainer, registry)
    print(f"бэкенд: {trainer.backend}")
    result = trainer.train(reason="cli", include_en=not args.no_en)
    if result is None:
        print("обучение не выполнено: недостаточно данных")
        return 1
    print(json.dumps({
        "promoted": result.promoted, "version": result.version,
        "metrics": result.metrics, "baseline": result.baseline,
        "n_train": result.n_train, "n_test": result.n_test, "notes": result.notes,
    }, ensure_ascii=False, indent=2))
    return 0 if result.promoted else 2


def cmd_status(args) -> int:
    cfg = _prepare(args.config, quiet=True)
    db = Database(cfg.db_path)
    registry = ModelRegistry(db, models_dir=MODELS_DIR)
    registry.load()
    if getattr(args, "json", False):
        print_report(db, registry, cfg, as_json=True)
        return 0
    print_report(db, registry, cfg, as_json=False,
                 color=None if not getattr(args, "no_color", False) else False)
    if getattr(args, "holdout", False):
        from .trainer import Trainer

        summary = Trainer(db, cfg.trainer, registry).holdout_summary()
        print("\n  ОТЛОЖЕННАЯ ВЫБОРКА")
        print(f"  Примеров:  {summary['n']}  (fake {summary['fake']}, "
              f"real {summary['real']}, свежих {summary['fresh']})")
        if summary["saturated"]:
            print("  F1 = 1.0: выборка слишком простая — версии сравниваются "
                  "по LogLoss, ROC-AUC и объёму данных")
        if summary["small"]:
            print(f"  Внимание: меньше {summary['min_comparison_n']} примеров — "
                  "метрики шумные, добавляйте подтверждённые метки")
        print()
    if getattr(args, "history", 0):
        print("\n  ИСТОРИЯ ОБУЧЕНИЯ")
        for row in db.training_history(args.history):
            flag = "принята " if row["promoted"] else "отклонена"
            print(f"  {_fmt_time(row['started_at'])}  {flag}  "
                  f"F1 {row['f1']} (было {row['baseline_f1']})  "
                  f"· {row['notes'] or ''}"[:120])
        print()
    return 0


def _fmt_time(ts) -> str:
    import time as _time
    return _time.strftime("%Y-%m-%d %H:%M", _time.localtime(float(ts))) if ts else "—"


def cmd_items(args) -> int:
    cfg = _prepare(args.config, quiet=True)
    db = Database(cfg.db_path)
    where, params = ("WHERE label=?", (args.label,)) if args.label else ("", ())
    rows = db.query(
        f"SELECT id, source_name, title, label, label_confidence, fake_prob, "
        f"trust_score, trust_level, label_source, url FROM items {where} "
        f"ORDER BY id DESC LIMIT ?", (*params, args.limit))
    for row in rows:
        title = (row["title"] or row["url"] or "")[:78]
        prob = row["fake_prob"]
        print(f"#{row['id']:<5} {(row['label'] or '-'):<9} "
              f"conf={(row['label_confidence'] or 0):.2f} "
              f"p={prob if prob is not None else -1:.2f} "
              f"trust={row['trust_score'] if row['trust_score'] is not None else '-'} "
              f"src={(row['label_source'] or '-'):<8} {title}")
    if not rows:
        print("нет материалов по фильтру")
    return 0


def cmd_models(args) -> int:
    cfg = _prepare(args.config, quiet=True)
    db = Database(cfg.db_path)
    registry = ModelRegistry(db)
    registry.load()
    if args.promote:
        row = next((r for r in db.model_versions() if r["version"] == args.promote), None)
        if row is None:
            print(f"версия {args.promote} не найдена")
            return 1
        registry.promote(row["version"], Path(row["path"]))
        print(f"активная модель: {row['version']}")
        return 0
    print(f"активная модель: {registry.version}")
    for row in db.model_versions():
        mark = "*" if row["active"] else " "
        print(f"{mark} {row['version']:<18} F1={row['f1']} "
              f"P={row['precision_score']} R={row['recall']} "
              f"n={row['n_train']} {row['path']}")
    return 0


def cmd_sources(args) -> int:
    cfg = _prepare(args.config, quiet=True)
    db = Database(cfg.db_path)
    states = {r["source_id"]: r for r in db.last_fetch_by_source()}
    for spec in cfg.source_specs:
        state = states.get(spec.id)
        status = "—" if state is None else ("ok" if state["ok"] else "ошибка")
        err = "" if state is None or not state["error"] else f" | {state['error'][:60]}"
        print(f"{spec.id:<16} {spec.kind:<9} {spec.role:<9} rel={spec.reliability:.2f} "
              f"{'вкл' if spec.enabled else 'выкл'} | {status}{err}")
    return 0


def cmd_export(args) -> int:
    cfg = _prepare(args.config, quiet=True)
    db = Database(cfg.db_path)
    rows = db.query(
        "SELECT id, source_name, url, title, text, label, label_confidence, "
        "label_source, label_reason FROM items WHERE label IN ('fake','real') "
        "AND label_confidence >= ? ORDER BY id", (cfg.trainer.min_label_confidence,))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "source", "url", "title", "text", "label",
                         "confidence", "decided_by", "reason"])
        for row in rows:
            writer.writerow([row["id"], row["source_name"], row["url"], row["title"],
                             row["text"], row["label"], row["label_confidence"],
                             row["label_source"], row["label_reason"]])
    print(f"экспортировано {len(rows)} размеченных материалов -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopilot", description="BAYKAM Autopilot: сбор, проверка, дообучение")
    parser.add_argument("--config", help="путь к config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="запустить цикл 24/7")
    p_run.add_argument("--once", action="store_true", help="один цикл и выход")
    p_run.add_argument("--no-llm", action="store_true",
                       help="без вызовов модели (только локальный классификатор)")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("collect", help="один проход по источникам").set_defaults(func=cmd_collect)

    p_verify = sub.add_parser("verify", help="проверить ожидающие материалы")
    p_verify.add_argument("--limit", type=int, default=10)
    p_verify.set_defaults(func=cmd_verify)

    p_train = sub.add_parser("train", help="обучить и при росте метрик сменить модель")
    p_train.add_argument("--no-en", action="store_true", help="без английского корпуса")
    p_train.set_defaults(func=cmd_train)

    p_status = sub.add_parser("status", help="наглядная сводка по базе и модели")
    p_status.add_argument("--json", action="store_true", help="машинный формат")
    p_status.add_argument("--no-color", action="store_true")
    p_status.add_argument("--history", type=int, default=0,
                          help="показать N последних запусков обучения")
    p_status.add_argument("--holdout", action="store_true",
                          help="состав отложенной выборки и её насыщение")
    p_status.set_defaults(func=cmd_status)

    p_items = sub.add_parser("items", help="последние материалы")
    p_items.add_argument("--label", choices=["fake", "real", "uncertain"])
    p_items.add_argument("--limit", type=int, default=20)
    p_items.set_defaults(func=cmd_items)

    p_models = sub.add_parser("models", help="версии модели")
    p_models.add_argument("--promote", help="сделать версию активной")
    p_models.set_defaults(func=cmd_models)

    sub.add_parser("sources", help="состояние источников").set_defaults(func=cmd_sources)

    p_export = sub.add_parser("export", help="экспорт размеченных данных в CSV")
    p_export.add_argument("--out", default="autopilot/data/labeled.csv")
    p_export.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        print(f"ошибка конфигурации: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
