"""Человекочитаемая сводка состояния автопилота для PowerShell.

    python -m autopilot status --pretty
    .\\autopilot\\STATUS.ps1

Показывает: сколько собрано, сколько размечено (real / fake / uncertain),
активную модель с метриками, последнее дообучение и состояние источников.
Цвета — ANSI-коды, Windows Terminal и PowerShell 7 их понимают.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .db import Database
from .model import ModelRegistry
from .settings import AppConfig

BAR = "─" * 64
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
CYAN = "\x1b[36m"
DIM = "\x1b[2m"


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))


def _ago(ts: float | None) -> str:
    if not ts:
        return "никогда"
    delta = max(0, time.time() - float(ts))
    if delta < 3600:
        return f"{int(delta // 60)} мин назад"
    if delta < 86400:
        return f"{delta / 3600:.1f} ч назад"
    return f"{delta / 86400:.1f} дн назад"


def _pct(part: int, total: int) -> str:
    return f"{100 * part / total:.0f}%" if total else "—"


def build_report(db: Database, registry: ModelRegistry,
                 cfg: AppConfig) -> dict[str, Any]:
    stats = db.stats()
    by_status = stats["by_status"]
    by_label = stats["by_label"]
    total = stats["items_total"]

    labeled = by_label.get("fake", 0) + by_label.get("real", 0)
    fetches = {r["source_id"]: r for r in db.last_fetch_by_source()}
    sources = []
    for spec in cfg.source_specs:
        state = fetches.get(spec.id)
        sources.append({
            "id": spec.id,
            "name": spec.name,
            "role": spec.role,
            "enabled": spec.enabled,
            "ok": bool(state["ok"]) if state else None,
            "last_seen": _ago(state["ts"] if state else None),
            "new_last": int(state["items_new"]) if state else 0,
            "error": (state["error"] or "") if state else "",
        })

    active = stats.get("active_model") or {}
    last = stats.get("last_training") or {}
    model = {
        "version": registry.version,
        "origin": registry.source,
        "f1": active.get("f1"),
        "precision": active.get("precision_score"),
        "recall": active.get("recall"),
        "n_train": active.get("n_train"),
        "created_at": _fmt_ts(active.get("created_at")),
        "last_training_at": _fmt_ts(last.get("started_at")),
        "last_decision": ("принята" if last.get("promoted")
                          else "отклонена" if last else "—"),
        "last_notes": last.get("notes") or "",
        "last_f1": last.get("f1"),
        "last_baseline_f1": last.get("baseline_f1"),
    }

    return {
        "items_total": total,
        "verified": labeled,
        "fake": by_label.get("fake", 0),
        "real": by_label.get("real", 0),
        "uncertain": by_status.get("uncertain", 0),
        "archived_uncertain": by_status.get("archived_uncertain", 0),
        "pending": by_status.get("scored", 0) + by_status.get("new", 0),
        "labels_share": _pct(labeled, total),
        "llm_calls_last_hour": stats.get("llm_calls_last_hour", 0),
        "llm_budget": cfg.verifier.llm_hourly_budget,
        "model": model,
        "sources": sources,
        "by_source": stats.get("by_source", []),
    }


def render(report: dict[str, Any], color: bool | None = None) -> str:
    if color is None:
        color = os.environ.get("NO_COLOR") is None

    def c(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if color and code else text

    m = report["model"]
    out: list[str] = [c(BAR, CYAN),
                      c("  BAYKAM · АВТОПИЛОТ", CYAN) + "   сбор · проверка · дообучение",
                      c(BAR, CYAN), ""]

    out.append(f"  Собрано материалов:        {report['items_total']}")
    out.append(f"  Размечено (fake/real):     {report['verified']} "
               f"({report['labels_share']} от собранного)")
    out.append(f"    {c('fake', RED)}: {report['fake']:<4} "
               f"{c('real', GREEN)}: {report['real']}")
    out.append(f"  Ожидают проверки:          {report['pending']}")
    out.append(f"  Неопределено (uncertain):  {report['uncertain']}")
    out.append(f"  В архиве:                  {report['archived_uncertain']}")
    out.append("")

    out.append(c(BAR, CYAN))
    out.append("  МОДЕЛЬ")
    out.append(f"  Активная версия:    {m['version']}  ({m['origin']})")
    if m.get("created_at") and m["created_at"] != "—":
        out.append(f"  Обучена:            {m['created_at']}")
    if m.get("n_train"):
        out.append(f"  Примеров в обучении: {m['n_train']}")
    if m.get("f1") is not None:
        out.append(f"  Метрики:            F1 {m['f1']}  ·  "
                   f"precision {m['precision']}  ·  recall {m['recall']}")
    else:
        out.append("  Метрики:            — (автопилот ещё не переобучал модель)")
    out.append(f"  Последнее обучение: {m['last_training_at']} — {m['last_decision']}")
    if m.get("last_notes"):
        out.append(f"    {c(m['last_notes'][:110], DIM)}")
    out.append(f"  Вызовов модели за час: {report['llm_calls_last_hour']} "
               f"из {report['llm_budget']}")
    out.append("")

    out.append(c(BAR, CYAN))
    out.append("  ИСТОЧНИКИ")
    for src in report["sources"]:
        if not src["enabled"]:
            mark, code = "выкл", ""
        elif src["ok"] is None:
            mark, code = "—   ", ""
        elif src["ok"]:
            mark, code = "ok  ", GREEN
        else:
            mark, code = "ошиб", RED
        role = "фактчек" if src["role"] == "factcheck" else "новости"
        detail = f"+{src['new_last']}" if src["new_last"] else src["last_seen"]
        line = f"  {c(mark, code)} {src['id']:<16} {role:<8} {detail}"
        if src["error"]:
            line += f"  · {c(src['error'][:40], DIM)}"
        out.append(line)
    out += ["", c(BAR, CYAN)]
    return "\n".join(out)


def print_report(db: Database, registry: ModelRegistry, cfg: AppConfig,
                 as_json: bool = False, color: bool | None = None) -> int:
    report = build_report(db, registry, cfg)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(render(report, color=color))
    return 0
