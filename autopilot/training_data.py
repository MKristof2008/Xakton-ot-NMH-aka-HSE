"""Обучающий корпус из данных проекта BAYKAM.

Модель должна учиться на том, что проект уже считает достоверным, а не на
посторонних датасетах. Источники по убыванию веса:

1. **Обратная связь пользователей** (`backend/store.py`, таблица `feedback`):
   там есть `expected` — какой вердикт ждали. Самый ценный сигнал.
2. **Эталонный набор** (`ml/data/eval_set.json`): кейсы с ожидаемым
   вердиктом; `правда` -> real, `фейк` -> fake, `непонятно` не берём.
3. **Курируемая база фактов** (`ml/lakmus/data/kb_facts.json`): утверждения
   с `verdict=true` -> real, `false_claim` -> fake.
4. **Собранный автопилотом корпус**: карточки фактчекеров -> fake,
   сообщения агентств -> real. Это самые слабые метки (стиль издания
   влияет), поэтому у них низкий вес.

Функции модуля ничего не меняют в проекте: только читают файлы и, если
база бэкенда существует, её таблицу `feedback`.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .logsetup import get_logger
from .settings import ML_DIR, REPO_DIR

log = get_logger("training_data")

#: Файлы проекта, которые пробуем прочитать (разные ветки кладут их по-разному).
EVAL_SET_CANDIDATES = (
    ML_DIR / "data" / "eval_set.json",
    ML_DIR / "eval_set.json",
)
KB_FACTS_CANDIDATES = (
    ML_DIR / "lakmus" / "data" / "kb_facts.json",
    ML_DIR / "data" / "kb_facts.json",
)
FEEDBACK_DB_CANDIDATES = (
    REPO_DIR / "backend" / "data" / "app.db",
    REPO_DIR / "backend" / "data" / "baуkam.db",
    REPO_DIR / "data" / "app.db",
    REPO_DIR / "backend" / "store.db",
)

#: Веса источников: живая обратная связь ценнее статического корпуса.
WEIGHT_FEEDBACK = 3.0
WEIGHT_EVAL = 2.0
WEIGHT_KB = 1.5
WEIGHT_CORPUS = 0.5

VERDICT_TO_LABEL = {
    "правда": 0, "true": 0, "real": 0, "верно": 0,
    "фейк": 1, "fake": 1, "ложь": 1, "false": 1,
}


@dataclass
class TrainExample:
    text: str
    label: int          # 1 = fake, 0 = real
    origin: str         # feedback | eval | kb | corpus
    weight: float = 1.0
    item_id: int | None = None


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("%s не прочитан: %s", path.name, exc)
        return None


def _verdict_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 0 if value else 1
    return VERDICT_TO_LABEL.get(str(value).strip().lower())


# ---------------------------------------------------------------------------
# Источники
# ---------------------------------------------------------------------------

def feedback_examples(db_path: Path | None = None,
                      limit: int = 5000) -> list[TrainExample]:
    """Обратная связь из базы бэкенда: `expected` — что ждали."""
    path = db_path or _first_existing(FEEDBACK_DB_CANDIDATES)
    if path is None or not Path(path).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT text, expected, verdict_ok FROM feedback "
                "WHERE expected IS NOT NULL ORDER BY created_at DESC LIMIT ?",
                (int(limit),)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        log.info("обратная связь недоступна (%s): %s", path.name, exc)
        return []

    out: list[TrainExample] = []
    for row in rows:
        text = (row["text"] or "").strip()
        label = _verdict_label(row["expected"])
        if len(text) >= 60 and label is not None:
            out.append(TrainExample(text, label, "feedback", WEIGHT_FEEDBACK))
    if out:
        log.info("обратная связь: %d примеров", len(out))
    return out


def eval_examples(path: Path | None = None) -> list[TrainExample]:
    """Эталонный набор проекта: кейсы с ожидаемым вердиктом."""
    path = path or _first_existing(EVAL_SET_CANDIDATES)
    if path is None:
        return []
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    out: list[TrainExample] = []
    for case in data.get("cases") or []:
        if not isinstance(case, dict):
            continue
        text = str(case.get("text") or "").strip()
        label = _verdict_label(case.get("expect"))
        if len(text) >= 40 and label is not None:
            out.append(TrainExample(text, label, "eval", WEIGHT_EVAL))
    if out:
        log.info("эталонный набор: %d примеров", len(out))
    return out


def kb_examples(path: Path | None = None) -> list[TrainExample]:
    """Курируемая база фактов: true -> real, false_claim -> fake."""
    path = path or _first_existing(KB_FACTS_CANDIDATES)
    if path is None:
        return []
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    out: list[TrainExample] = []
    for topic in data.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        for fact in topic.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("claim") or "").strip()
            verdict = str(fact.get("verdict") or "").strip().lower()
            if len(claim) < 40:
                continue
            if verdict == "true":
                out.append(TrainExample(claim, 0, "kb", WEIGHT_KB))
            elif verdict in ("false_claim", "false", "fake"):
                out.append(TrainExample(claim, 1, "kb", WEIGHT_KB))
            # mixed — неоднозначный факт: не учим, чтобы не шуметь
    if out:
        log.info("база фактов: %d примеров", len(out))
    return out


def corpus_examples(db, limit: int = 2000) -> list[TrainExample]:
    """Собранный корпус: карточки фактчекеров -> fake, агентства -> real."""
    rows = db.query(
        """SELECT id, role, source_id, title, text FROM items
           WHERE length(text) >= 200 AND label IS NULL
           ORDER BY id DESC LIMIT ?""", (int(limit),))
    out: list[TrainExample] = []
    for row in rows:
        text = f"{row['title'] or ''}. {row['text']}".strip()
        if row["role"] == "factcheck":
            out.append(TrainExample(text, 1, "corpus", WEIGHT_CORPUS,
                                    int(row["id"])))
        else:
            out.append(TrainExample(text, 0, "corpus", WEIGHT_CORPUS,
                                    int(row["id"])))
    if out:
        log.info("собранный корпус: %d примеров", len(out))
    return out


def collect(db=None) -> list[TrainExample]:
    """Все источники вместе. Порядок — по убыванию ценности."""
    out: list[TrainExample] = []
    out.extend(feedback_examples())
    out.extend(eval_examples())
    out.extend(kb_examples())
    if db is not None:
        out.extend(corpus_examples(db))
    return out


def describe(examples: Iterable[TrainExample]) -> dict[str, Any]:
    """Сводка по составу корпуса — для логов и статуса."""
    from collections import Counter

    rows = list(examples)
    by_origin = Counter(ex.origin for ex in rows)
    by_label = Counter(ex.label for ex in rows)
    return {
        "n": len(rows),
        "fake": by_label.get(1, 0),
        "real": by_label.get(0, 0),
        "by_origin": dict(by_origin),
    }
