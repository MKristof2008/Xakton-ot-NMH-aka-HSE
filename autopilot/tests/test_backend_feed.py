"""Чтение статистики автопилота: только на чтение и без падений.

Проверяем, что бэкенд не может испортить базу автопилота, корректно
отвечает при её отсутствии и отдаёт те поля, которые ждёт фронтенд.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import autopilot_feed  # noqa: E402

SCHEMA = """
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL,
    source_name TEXT DEFAULT '', kind TEXT DEFAULT '', url TEXT DEFAULT '',
    url_hash TEXT DEFAULT '', content_hash TEXT UNIQUE, simhash INTEGER,
    title TEXT DEFAULT '', text TEXT NOT NULL, published_at TEXT,
    fetched_at REAL NOT NULL, lang TEXT, chars INTEGER DEFAULT 0,
    meta TEXT DEFAULT '{}', role TEXT DEFAULT 'news', status TEXT DEFAULT 'new',
    fake_prob REAL, trust_score INTEGER, uncertainty REAL, trust_level TEXT,
    label TEXT, label_confidence REAL, label_reason TEXT, label_source TEXT,
    model_version TEXT, evidence TEXT, verified_at REAL,
    training_used INTEGER DEFAULT 0, llm_calls INTEGER DEFAULT 0,
    verify_attempts INTEGER DEFAULT 0
);
CREATE TABLE llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    item_id INTEGER, step TEXT, ok INTEGER, latency_ms INTEGER, error TEXT);
CREATE TABLE training_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL,
    finished_at REAL, n_train INTEGER, n_new INTEGER, n_test INTEGER, f1 REAL,
    precision_score REAL, recall REAL, baseline_f1 REAL, promoted INTEGER,
    model_version TEXT, notes TEXT);
CREATE TABLE model_versions (version TEXT PRIMARY KEY, path TEXT, f1 REAL,
    precision_score REAL, recall REAL, n_train INTEGER, created_at REAL,
    active INTEGER);
"""


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    rows = [
        ("a", "ТАСС", "news", "Срочно! Всё отменяют, перешли всем!", "fake", 0.9),
        ("b", "РИА", "news", "Агентство сообщает о решении совета директоров.", "real", 0.8),
        ("c", "StopFake", "factcheck", "Фейк: запретили наличные с первого марта.", "fake", 0.9),
        ("d", "Lenta", "news", "Материал без метки, ждёт проверки.", None, None),
    ]
    for i, (src, name, role, text, label, conf) in enumerate(rows):
        conn.execute(
            "INSERT INTO items(source_id, source_name, role, text, content_hash, "
            "fetched_at, status, label, label_confidence, label_reason, "
            "label_source, verified_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (src, name, role, text, f"h{i}", 1_700_000_000.0 + i,
             "verified" if label else "scored", label, conf,
             "опровергнуто" if label == "fake" else "подтверждено",
             "claims", 1_700_000_100.0 + i if label else None),
        )
    conn.execute(
        "INSERT INTO items(source_id, source_name, role, text, content_hash, "
        "fetched_at, status) VALUES('e','Коммерсантъ','news','В очереди.',?,?, 'scored')",
        ("h9", 1_700_000_500.0))
    conn.execute(
        "INSERT INTO items(source_id, source_name, role, text, content_hash, "
        "fetched_at, status, label) VALUES('f','Baza','news','Старое.',?,?,"
        " 'archived_uncertain','uncertain')",
        ("h10", 1_700_000_600.0))
    conn.execute(
        "INSERT INTO model_versions VALUES('v1','/tmp/v1.joblib',0.88,0.79,1.0,284,?,1)",
        (1_700_000_000.0,))
    conn.execute(
        "INSERT INTO training_runs(started_at, f1, precision_score, recall, "
        "promoted, model_version) VALUES(?,0.88,0.79,1.0,1,'v1')",
        (1_700_000_000.0,))
    conn.execute("INSERT INTO llm_calls(ts, step, ok) VALUES(?, 'arbiter', 1)",
                 (__import__("time").time(),))
    conn.commit()
    conn.close()


@pytest.fixture()
def ap_db(tmp_path, monkeypatch):
    path = tmp_path / "autopilot.db"
    _make_db(path)
    pointer = tmp_path / "active.json"
    pointer.write_text('{"version": "v1", "promoted_at": "2026-11-22T15:27:47"}',
                       encoding="utf-8")
    monkeypatch.setattr(autopilot_feed, "AUTOPILOT_DB", path)
    monkeypatch.setattr(autopilot_feed, "ACTIVE_MODEL_POINTER", pointer)
    return path


def test_summary_counts(ap_db):
    data = autopilot_feed.summary()
    assert data["available"] is True
    assert data["items_total"] == 6
    assert data["fake"] == 2 and data["real"] == 1
    # «неясно» = явная метка uncertain (нет) + архив без метки (1) + архив
    # с меткой uncertain (1)
    assert data["uncertain"] == 2
    assert data["pending"] == 2           # scored + new
    assert data["archived"] == 1
    assert data["model"]["version"] == "v1"
    assert data["model"]["f1"] == 0.88
    assert data["llm"]["last_hour"] >= 1


def test_recent_returns_labels_and_excerpt(ap_db):
    data = autopilot_feed.recent(limit=10)
    assert data["total"] == 4              # 3 метки + 1 архив без метки
    assert all(item["label"] in ("fake", "real", "uncertain")
               for item in data["items"])
    assert all("excerpt" in item and "text" not in item for item in data["items"])
    # архивные записи без метки не имеют verified_at — сортировка их терпит
    stamps = [item["verified_at"] for item in data["items"]]
    assert sum(1 for s in stamps if s is not None) == 3
    dated = [s for s in stamps if s is not None]
    assert dated == sorted(dated, reverse=True)


def test_recent_filters_by_label(ap_db):
    fakes = autopilot_feed.recent(limit=10, label="fake")
    assert fakes["total"] == 2
    assert {item["label"] for item in fakes["items"]} == {"fake"}
    # «неясно» — это архивные материалы без решения
    unclear = autopilot_feed.recent(limit=10, label="uncertain")
    assert unclear["total"] == 1
    assert unclear["items"][0]["label"] == "uncertain"


def test_recent_limit_is_clamped(ap_db):
    assert len(autopilot_feed.recent(limit=999)["items"]) == 4
    assert len(autopilot_feed.recent(limit=1)["items"]) == 1


def test_connection_is_read_only(ap_db):
    conn = autopilot_feed._connect()
    assert conn is not None
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM items")
    finally:
        conn.close()


def test_missing_db_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(autopilot_feed, "AUTOPILOT_DB", tmp_path / "nope.db")
    assert autopilot_feed.available() is False
    data = autopilot_feed.summary()
    assert data["available"] is False and data["items_total"] == 0
    assert autopilot_feed.recent(limit=5) == {"items": [], "total": 0}
