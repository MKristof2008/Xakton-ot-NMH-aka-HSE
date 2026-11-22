"""SQLite-хранилище автопилота (WAL, короткие транзакции, потокобезопасно).

Таблицы:
- `items`         — собранный контент, ML-скор, метка и её обоснование;
- `fetch_log`     — журнал обходов источников;
- `llm_calls`     — расход бюджета вызовов модели;
- `arbitrations`  — решения LLM-арбитра по спорным случаям;
- `training_runs` — история дообучения и метрики;
- `model_versions`— версии модели и активная версия.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        TEXT NOT NULL,
    source_name      TEXT NOT NULL DEFAULT '',
    kind             TEXT NOT NULL DEFAULT '',
    url              TEXT NOT NULL DEFAULT '',
    url_hash         TEXT NOT NULL DEFAULT '',
    content_hash     TEXT NOT NULL UNIQUE,
    simhash          INTEGER,
    title            TEXT NOT NULL DEFAULT '',
    text             TEXT NOT NULL,
    published_at     TEXT,
    fetched_at       REAL NOT NULL,
    lang             TEXT,
    chars            INTEGER NOT NULL DEFAULT 0,
    meta             TEXT NOT NULL DEFAULT '{}',
    role             TEXT NOT NULL DEFAULT 'news',
    status           TEXT NOT NULL DEFAULT 'new',
    fake_prob        REAL,
    trust_score      INTEGER,
    uncertainty      REAL,
    trust_level      TEXT,
    label            TEXT,
    label_confidence REAL,
    label_reason     TEXT,
    label_source     TEXT,
    model_version    TEXT,
    evidence         TEXT,
    verified_at      REAL,
    training_used    INTEGER NOT NULL DEFAULT 0,
    llm_calls        INTEGER NOT NULL DEFAULT 0,
    verify_attempts  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status, id);
CREATE INDEX IF NOT EXISTS idx_items_simhash ON items(simhash);
CREATE INDEX IF NOT EXISTS idx_items_fetched ON items(fetched_at);
CREATE INDEX IF NOT EXISTS idx_items_label ON items(label, verified_at);

CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    source_id   TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    http_status INTEGER,
    items_seen  INTEGER NOT NULL DEFAULT 0,
    items_new   INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_fetch_source ON fetch_log(source_id, ts);

CREATE TABLE IF NOT EXISTS llm_calls (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    item_id    INTEGER,
    step       TEXT NOT NULL DEFAULT '',
    ok         INTEGER NOT NULL,
    latency_ms INTEGER,
    error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_ts ON llm_calls(ts);

CREATE TABLE IF NOT EXISTS arbitrations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    INTEGER NOT NULL,
    ts         REAL NOT NULL,
    verdict    TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale  TEXT,
    cited      TEXT,
    ok         INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_arb_item ON arbitrations(item_id);

CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      REAL NOT NULL,
    finished_at     REAL,
    n_train         INTEGER,
    n_new           INTEGER,
    n_test          INTEGER,
    f1              REAL,
    precision_score REAL,
    recall          REAL,
    baseline_f1     REAL,
    promoted        INTEGER NOT NULL DEFAULT 0,
    model_version   TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version         TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    f1              REAL,
    precision_score REAL,
    recall          REAL,
    n_train         INTEGER,
    created_at      REAL NOT NULL,
    active          INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    """Одно короткое соединение на операцию — WAL это спокойно переживает."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate_columns(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _migrate_columns(conn) -> None:
        """Досоздаёт колонки в базе, созданной более старой версией."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(items)")}
        for column, ddl in {
            "role": "TEXT NOT NULL DEFAULT 'news'",
            "verify_attempts": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE items ADD COLUMN {column} {ddl}")

    # -- запросы ---------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(sql, tuple(params))
            return cur.lastrowid or cur.rowcount

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock, self._connect() as conn:
            conn.executemany(sql, list(rows))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            return conn.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = 0) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def close(self) -> None:
        pass  # соединения короткоживущие

    # -- источники -------------------------------------------------------
    def source_state(self, source_id: str) -> sqlite3.Row | None:
        rows = self.query(
            """SELECT source_id,
                      MAX(CASE WHEN ok=1 THEN ts END) AS last_ok_at,
                      SUM(CASE WHEN ok=0 AND ts >= ? THEN 1 ELSE 0 END) AS failures,
                      (SELECT error FROM fetch_log WHERE source_id=? AND ok=0
                       ORDER BY id DESC LIMIT 1) AS last_error
               FROM fetch_log WHERE source_id=? GROUP BY source_id""",
            (time.time() - 6 * 3600, source_id, source_id))
        return rows[0] if rows else None

    def record_fetch(self, source_id: str, ok: bool, http_status: int | None,
                     items_seen: int, items_new: int, error: str | None,
                     duration_ms: int) -> None:
        self.execute(
            """INSERT INTO fetch_log(ts, source_id, ok, http_status, items_seen,
                   items_new, error, duration_ms) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), source_id, 1 if ok else 0, http_status, items_seen,
             items_new, (error or "")[:500] or None, duration_ms))

    def last_fetch_by_source(self) -> list[sqlite3.Row]:
        return self.query(
            """SELECT f.source_id, f.ts, f.ok, f.items_new, f.error,
                      (SELECT COUNT(*) FROM fetch_log x
                       WHERE x.source_id=f.source_id AND x.ok=1) AS ok_total,
                      (SELECT COUNT(*) FROM fetch_log x
                       WHERE x.source_id=f.source_id AND x.ok=0) AS fail_total
               FROM fetch_log f
               JOIN (SELECT source_id, MAX(id) AS id FROM fetch_log
                     GROUP BY source_id) last ON last.id=f.id
               ORDER BY f.source_id""")

    # -- элементы --------------------------------------------------------
    def content_hash_exists(self, content_hash: str) -> bool:
        return self.query_one(
            "SELECT 1 FROM items WHERE content_hash=? LIMIT 1", (content_hash,)) is not None

    def near_duplicate_ids(self, simhash_int: int | None, max_distance: int,
                           window_days: int, limit: int = 3000) -> list[int]:
        if simhash_int is None:
            return []
        since = time.time() - window_days * 86400
        rows = self.query(
            "SELECT id, simhash FROM items WHERE fetched_at >= ? "
            "AND simhash IS NOT NULL ORDER BY id DESC LIMIT ?", (since, limit))
        mask = 0xFFFFFFFFFFFFFFFF
        out = []
        for row in rows:
            dist = bin((int(row["simhash"]) ^ int(simhash_int)) & mask).count("1")
            if dist <= max_distance:
                out.append(int(row["id"]))
        return out

    def insert_item(self, item: dict[str, Any]) -> int | None:
        text = item.get("text", "")
        try:
            return self.execute(
                """INSERT INTO items(source_id, source_name, kind, url, url_hash,
                       content_hash, simhash, title, text, published_at, fetched_at,
                       lang, chars, meta, role)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (item["source_id"], item.get("source_name", ""), item.get("kind", ""),
                 item.get("url", ""), item.get("url_hash", ""), item["content_hash"],
                 item.get("simhash"), item.get("title", ""), text,
                 item.get("published_at"), item.get("fetched_at") or time.time(),
                 item.get("lang"), len(text),
                 json.dumps(item.get("meta") or {}, ensure_ascii=False),
                 item.get("role", "news")))
        except sqlite3.IntegrityError:
            return None

    def pending_for_scoring(self, limit: int = 2000) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM items WHERE status='new' ORDER BY id LIMIT ?", (limit,))

    def set_score(self, item_id: int, prob: float | None, version: str) -> None:
        self.execute(
            "UPDATE items SET fake_prob=?, model_version=?, status='scored' WHERE id=?",
            (prob, version, item_id))

    def select_for_verification(self, limit: int,
                                ambiguous_share: float = 0.6) -> list[sqlite3.Row]:
        """Active Learning: вперёд идут случаи, где модель ближе всего к 0.5."""
        ambiguous = max(1, int(limit * ambiguous_share))
        rows = self.query(
            """SELECT * FROM items WHERE status='scored'
               ORDER BY ABS(COALESCE(fake_prob, 0.5) - 0.5) ASC, id DESC LIMIT ?""",
            (ambiguous,))
        seen = {int(r["id"]) for r in rows}
        if len(rows) < limit:
            for row in self.query(
                    "SELECT * FROM items WHERE status='scored' ORDER BY RANDOM() LIMIT ?",
                    (limit * 2,)):
                if int(row["id"]) in seen:
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows

    def save_verification(self, item_id: int, *, label: str, confidence: float,
                          reason: str, source: str, fake_prob: float | None,
                          trust_score: int | None, uncertainty: float | None,
                          level: str | None, evidence: dict | None,
                          llm_calls: int) -> None:
        self.execute(
            """UPDATE items SET status=?, label=?, label_confidence=?, label_reason=?,
                   label_source=?, fake_prob=COALESCE(?, fake_prob), trust_score=?,
                   uncertainty=?, trust_level=?, evidence=?, verified_at=?,
                   llm_calls=llm_calls+?
               WHERE id=?""",
            (label, label, confidence, reason[:4000], source, fake_prob, trust_score,
             uncertainty, level, json.dumps(evidence or {}, ensure_ascii=False),
             time.time(), llm_calls, item_id))

    def requeue(self, item_id: int, max_attempts: int = 3,
                ttl_hours: float = 48.0) -> str:
        """Возвращает материал в очередь или архивирует.

        Возвращает:
        - `"requeued"` — попытки ещё есть, материал снова в очереди;
        - `"archived"` — попытки исчерпаны или материал старше TTL,
          он уходит в `archived_uncertain`.

        Так очередь не забивается вечными повторами: каждый материал
        проверяется не больше `max_attempts` раз и не дольше `ttl_hours`.
        """
        cutoff = time.time() - ttl_hours * 3600
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT verify_attempts, fetched_at FROM items WHERE id=?",
                (item_id,)).fetchone()
            if row is None:
                return "archived"
            attempts = int(row["verify_attempts"] or 0)
            too_old = float(row["fetched_at"] or 0) < cutoff
            if attempts >= max_attempts or too_old:
                conn.execute(
                    "UPDATE items SET status='archived_uncertain' WHERE id=?", (item_id,))
                return "archived"
            conn.execute(
                "UPDATE items SET status='scored', verify_attempts=? WHERE id=?",
                (attempts + 1, item_id))
            return "requeued"

    def mark_archived(self, item_id: int, reason: str, evidence: dict | None,
                      fake_prob: float | None, llm_calls: int) -> None:
        """Переводит материал в архив без метки: он не идёт в обучение."""
        self.execute(
            """UPDATE items SET status='archived_uncertain', label_reason=?,
                   evidence=?, fake_prob=COALESCE(?, fake_prob), verified_at=?,
                   llm_calls=llm_calls+?
               WHERE id=?""",
            (reason[:4000], json.dumps(evidence or {}, ensure_ascii=False),
             fake_prob, time.time(), llm_calls, item_id))

    def archive_stale_uncertain(self, ttl_hours: float) -> int:
        """Архивирует `uncertain`, которые висят дольше TTL.

        Материал, по которому так и не нашлось основания, не должен
        перепроверяться бесконечно: он уходит в `archived_uncertain`
        и остаётся в базе для отчётности.
        """
        cutoff = time.time() - ttl_hours * 3600
        cur = self.execute(
            "UPDATE items SET status='archived_uncertain' "
            "WHERE status='uncertain' AND fetched_at < ?", (cutoff,))
        return int(cur or 0)

    def set_status(self, item_id: int, status: str, reason: str = "") -> None:
        self.execute(
            "UPDATE items SET status=?, label_reason=COALESCE(NULLIF(?, ''), label_reason) "
            "WHERE id=?", (status, reason, item_id))

    def get_item(self, item_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM items WHERE id=?", (item_id,))

    def labeled_examples(self, min_confidence: float) -> list[sqlite3.Row]:
        return self.query(
            """SELECT id, text, label, label_confidence, label_source FROM items
               WHERE label IN ('fake','real') AND label_confidence >= ?
               AND length(text) >= 60 ORDER BY id""", (min_confidence,))

    def new_labels_since(self, ts: float, min_confidence: float) -> int:
        return int(self.scalar(
            "SELECT COUNT(*) FROM items WHERE label IN ('fake','real') "
            "AND label_confidence >= ? AND verified_at > ?", (min_confidence, ts)))

    def mark_trained(self, ids: Sequence[int]) -> None:
        self.executemany("UPDATE items SET training_used=1 WHERE id=?",
                         [(int(i),) for i in ids if i is not None])

    # -- бюджет модели ---------------------------------------------------
    def log_llm_call(self, item_id: int | None, step: str, ok: bool,
                     latency_ms: int, error: str | None = None) -> None:
        self.execute(
            "INSERT INTO llm_calls(ts, item_id, step, ok, latency_ms, error) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (time.time(), item_id, step, 1 if ok else 0, latency_ms,
             (error or "")[:300] or None))

    def llm_calls_last_hour(self) -> int:
        return int(self.scalar("SELECT COUNT(*) FROM llm_calls WHERE ts > ?",
                               (time.time() - 3600,)))

    # -- арбитраж --------------------------------------------------------
    def save_arbitration(self, item_id: int, verdict: str, confidence: float,
                         rationale: str, cited: list[str], ok: bool) -> None:
        self.execute(
            """INSERT INTO arbitrations(item_id, ts, verdict, confidence, rationale,
                   cited, ok) VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (item_id, time.time(), verdict, confidence, rationale[:2000],
             json.dumps(cited, ensure_ascii=False), 1 if ok else 0))

    # -- обучение --------------------------------------------------------
    def start_training_run(self, n_train: int, n_new: int, n_test: int) -> int:
        return self.execute(
            "INSERT INTO training_runs(started_at, n_train, n_new, n_test) "
            "VALUES(?, ?, ?, ?)", (time.time(), n_train, n_new, n_test))

    def finish_training_run(self, run_id: int, *, metrics: dict, baseline_f1: float,
                            promoted: bool, model_version: str | None,
                            notes: str) -> None:
        self.execute(
            """UPDATE training_runs SET finished_at=?, f1=?, precision_score=?,
                   recall=?, baseline_f1=?, promoted=?, model_version=?, notes=?
               WHERE id=?""",
            (time.time(), metrics.get("f1"), metrics.get("precision"),
             metrics.get("recall"), baseline_f1, 1 if promoted else 0,
             model_version, notes[:2000], run_id))

    def last_training_run(self) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1")

    def last_training_finished_at(self) -> float:
        row = self.query_one(
            "SELECT MAX(COALESCE(finished_at, started_at)) AS ts FROM training_runs")
        return float(row["ts"] or 0) if row else 0.0

    def training_history(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))

    # -- версии модели ---------------------------------------------------
    def register_model(self, version: str, path: str, metrics: dict,
                       n_train: int, active: bool = False) -> None:
        self.execute(
            """INSERT INTO model_versions(version, path, f1, precision_score, recall,
                   n_train, created_at, active) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(version) DO UPDATE SET path=excluded.path, f1=excluded.f1,
                   precision_score=excluded.precision_score, recall=excluded.recall,
                   n_train=excluded.n_train, active=excluded.active""",
            (version, path, metrics.get("f1"), metrics.get("precision"),
             metrics.get("recall"), n_train, time.time(), 1 if active else 0))
        if active:
            self.execute("UPDATE model_versions SET active=0 WHERE version<>?", (version,))

    def active_model(self) -> sqlite3.Row | None:
        return self.query_one(
            "SELECT * FROM model_versions WHERE active=1 ORDER BY created_at DESC LIMIT 1")

    def model_versions(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM model_versions ORDER BY created_at DESC")

    # -- обслуживание ----------------------------------------------------
    def purge_old(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        removed = self.execute(
            "DELETE FROM items WHERE fetched_at < ? AND status IN ('new','scored')",
            (cutoff,))
        self.execute("DELETE FROM llm_calls WHERE ts < ?", (cutoff,))
        self.execute("DELETE FROM fetch_log WHERE ts < ?", (cutoff,))
        return int(removed or 0)

    # -- сводка ----------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        by_status = {r["status"]: r["n"] for r in self.query(
            "SELECT status, COUNT(*) AS n FROM items GROUP BY status")}
        by_label = {r["label"]: r["n"] for r in self.query(
            "SELECT label, COUNT(*) AS n FROM items WHERE label IS NOT NULL GROUP BY label")}
        by_source = [dict(r) for r in self.query(
            "SELECT source_id, source_name, role, COUNT(*) AS n FROM items "
            "GROUP BY source_id, source_name, role ORDER BY n DESC")]
        active = self.active_model()
        last = self.last_training_run()
        return {
            "items_total": int(self.scalar("SELECT COUNT(*) FROM items")),
            "by_status": by_status,
            "by_label": by_label,
            "by_source": by_source,
            "verified_total": int(self.scalar(
                "SELECT COUNT(*) FROM items WHERE label IS NOT NULL")),
            "archived_uncertain": int(self.scalar(
                "SELECT COUNT(*) FROM items WHERE status='archived_uncertain'")),
            "llm_calls_last_hour": self.llm_calls_last_hour(),
            "active_model": dict(active) if active else None,
            "last_training": dict(last) if last else None,
        }
