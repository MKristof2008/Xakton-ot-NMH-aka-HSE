"""Чтение статистики автопилота для фронтенда — только на чтение.

Автопилот работает 24/7 и пишет в свою базу `autopilot/data/autopilot.db`
в режиме WAL. Бэкенд открывает её через URI `?mode=ro`: SQLite не даёт
такому соединению менять файл, а WAL позволяет читать во время записи.

Если автопилот не запускался, модуль возвращает понятный «пусто»-ответ,
а не ошибку: страница должна открываться и без него.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

AUTOPILOT_DIR = Path(__file__).resolve().parent.parent / "autopilot"
AUTOPILOT_DB = Path(
    os.environ.get("AUTOPILOT_DB", AUTOPILOT_DIR / "data" / "autopilot.db")
)
ACTIVE_MODEL_POINTER = Path(
    os.environ.get(
        "AUTOPILOT_MODEL_POINTER", AUTOPILOT_DIR / "data" / "models" / "active.json"
    )
)

#: Сколько последних материалов отдавать в ленте по умолчанию.
DEFAULT_LIMIT = 30
MAX_LIMIT = 100


def _connect() -> sqlite3.Connection | None:
    """Соединение только на чтение. None, если базы ещё нет."""
    if not AUTOPILOT_DB.exists():
        return None
    uri = f"file:{AUTOPILOT_DB.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def available() -> bool:
    """Запускался ли автопилот: база создана и читается."""
    conn = _connect()
    if conn is None:
        return False
    try:
        conn.execute("SELECT 1 FROM items LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row is not None else None


def _scalar_map(conn: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {row[0]: row[1] for row in conn.execute(sql).fetchall()}


def _active_model() -> dict[str, Any] | None:
    """Версия модели из `active.json` и её метрики из базы."""
    if not ACTIVE_MODEL_POINTER.exists():
        return None
    try:
        pointer = json.loads(ACTIVE_MODEL_POINTER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "version": pointer.get("version"),
        "promoted_at": pointer.get("promoted_at"),
    }


def _model_metrics(conn: sqlite3.Connection, version: str | None) -> dict[str, Any]:
    """Метрики активной версии из `model_versions`, иначе последний прогон."""
    if version:
        row = conn.execute(
            "SELECT version, f1, precision_score, recall, n_train, created_at "
            "FROM model_versions WHERE version=?",
            (version,),
        ).fetchone()
        if row is not None:
            return dict(row)
    row = conn.execute(
        "SELECT model_version AS version, f1, precision_score, recall, "
        "n_train, finished_at AS created_at FROM training_runs "
        "WHERE promoted=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row is not None else {}


def _llm_usage(conn: sqlite3.Connection) -> dict[str, int]:
    """Сколько вызовов модели сделано за час и за сутки."""
    now = time.time()
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    return {
        "last_hour": int(_one(conn, "SELECT COUNT(*) FROM llm_calls WHERE ts > ?",
                              (now - 3600,)) or 0),
        "today": int(_one(conn, "SELECT COUNT(*) FROM llm_calls WHERE ts >= ?",
                          (midnight,)) or 0),
    }


def summary() -> dict[str, Any]:
    """Счётчики автопилота: собрано, размечено, в очереди, в архиве."""
    if not available():
        return {
            "available": False,
            "items_total": 0,
            "by_label": {},
            "fake": 0,
            "real": 0,
            "uncertain": 0,
            "archived": 0,
            "pending": 0,
            "model": None,
            "llm": {"last_hour": 0, "today": 0},
            "sources": [],
        }

    conn = _connect()
    assert conn is not None
    try:
        by_status = _scalar_map(conn, "SELECT status, COUNT(*) FROM items GROUP BY status")
        by_label = _scalar_map(
            conn, "SELECT label, COUNT(*) FROM items WHERE label IS NOT NULL GROUP BY label"
        )
        model = _active_model()
        version = (model or {}).get("version")
        metrics = _model_metrics(conn, version)
        sources = [
            dict(row)
            for row in conn.execute(
                "SELECT source_id, source_name, role, COUNT(*) AS n FROM items "
                "GROUP BY source_id, source_name, role ORDER BY n DESC"
            ).fetchall()
        ]
        # «Неясно» — это и явная метка uncertain, и архив: материал, по
        # которому оснований так и не нашлось. Иначе счётчик всегда был бы 0.
        uncertain_total = int(by_label.get("uncertain", 0)) + int(
            by_status.get("archived_uncertain", 0))
        return {
            "available": True,
            "items_total": int(_one(conn, "SELECT COUNT(*) FROM items") or 0),
            "by_label": by_label,
            "fake": int(by_label.get("fake", 0)),
            "real": int(by_label.get("real", 0)),
            "uncertain": uncertain_total,
            "archived": int(by_status.get("archived_uncertain", 0)),
            "pending": int(by_status.get("scored", 0) + by_status.get("new", 0)),
            "by_status": by_status,
            "model": {**(model or {}), **metrics} if (model or metrics) else None,
            "llm": _llm_usage(conn),
            "sources": sources,
        }
    except sqlite3.Error:
        return {"available": False, "items_total": 0, "by_label": {}, "fake": 0,
                "real": 0, "uncertain": 0, "archived": 0, "pending": 0,
                "model": None, "llm": {"last_hour": 0, "today": 0}, "sources": []}
    finally:
        conn.close()


def recent(limit: int = DEFAULT_LIMIT, label: str | None = None) -> dict[str, Any]:
    """Последние проверенные материалы автопилота с вердиктом и обоснованием."""
    limit = max(1, min(MAX_LIMIT, int(limit)))
    if label not in ("fake", "real", "uncertain"):
        label = None

    if not available():
        return {"items": [], "total": 0}

    conn = _connect()
    assert conn is not None
    try:
        # материал считается разобранным, если у него есть метка ИЛИ он ушёл
        # в архив без решения
        where = "WHERE (label IS NOT NULL OR status = 'archived_uncertain')"
        params: list[Any] = []
        if label == "uncertain":
            where += " AND (label = 'uncertain' OR status = 'archived_uncertain')"
        elif label:
            where += " AND label = ?"
            params.append(label)
        total = int(_one(conn, f"SELECT COUNT(*) FROM items {where}", tuple(params)) or 0)
        rows = conn.execute(
            f"""SELECT id, source_id, source_name, url, title, text, label,
                       label_confidence, label_reason, label_source, trust_score,
                       trust_level, verified_at
                FROM items {where}
                ORDER BY COALESCE(verified_at, fetched_at) DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["excerpt"] = (item.pop("text") or "")[:320]
            item["title"] = item.get("title") or ""
            if not item.get("label"):
                item["label"] = "uncertain"
            items.append(item)
        return {"items": items, "total": total}
    except sqlite3.Error:
        return {"items": [], "total": 0}
    finally:
        conn.close()
