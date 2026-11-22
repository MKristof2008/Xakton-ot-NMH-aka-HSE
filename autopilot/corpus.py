"""Поиск по собранному корпусу: TF-IDF индекс в памяти.

Нужен, чтобы проверять материалы не только по статической базе знаний
проекта, но и по тому, что автопилот собрал сам: другие издания об одном
событии, карточки фактчекеров, Telegram-каналы.

Индекс пересобирается, когда корпус заметно вырос или прошло 10 минут.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .db import Database
from .logsetup import get_logger

log = get_logger("corpus")

REBUILD_INTERVAL_S = 600
REBUILD_GROWTH = 0.15
MAX_DOCS = 20_000


@dataclass
class Hit:
    id: int
    source_id: str
    source_name: str
    role: str          # news | factcheck
    url: str
    title: str
    text: str
    score: float


class CorpusIndex:
    def __init__(self, db: Database, max_docs: int = MAX_DOCS):
        self.db = db
        self.max_docs = max_docs
        self._lock = threading.RLock()
        self._vectorizer = None
        self._matrix = None
        self._rows: list[dict] = []
        self._built_at = 0.0
        self._built_count = -1

    def _load_rows(self) -> list[dict]:
        rows = self.db.query(
            "SELECT id, source_id, source_name, role, url, title, text FROM items "
            "WHERE length(text) >= 100 ORDER BY id DESC LIMIT ?", (self.max_docs,))
        return [dict(r) for r in rows]

    def _stale(self, count: int) -> bool:
        if self._matrix is None:
            return True
        if time.time() - self._built_at > REBUILD_INTERVAL_S:
            return True
        return count > max(50, int(self._built_count * (1 + REBUILD_GROWTH)))

    def maybe_rebuild(self) -> None:
        count = int(self.db.scalar("SELECT COUNT(*) FROM items"))
        if not self._stale(count):
            return
        self.rebuild(count)

    def rebuild(self, count: int | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        rows = self._load_rows()
        if len(rows) < 3:
            with self._lock:
                self._rows, self._matrix, self._vectorizer = rows, None, None
                self._built_at = time.time()
                self._built_count = len(rows)
            return
        texts = [f"{r['title']} {r['text'][:1500]}" for r in rows]
        # на маленьком корпусе min_df=2 выкидывает все слова — индекс пустеет
        vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                     min_df=1 if len(rows) < 50 else 2,
                                     max_features=120_000, sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            log.warning("индекс корпуса не построен: %s", exc)
            return
        with self._lock:
            self._vectorizer, self._matrix, self._rows = vectorizer, matrix, rows
            self._built_at = time.time()
            self._built_count = count if count is not None else len(rows)
        log.info("индекс корпуса: %d документов", len(rows))

    def search(self, query: str, top_k: int = 6, min_score: float = 0.15,
               exclude_id: int | None = None) -> list[Hit]:
        if not query.strip():
            return []
        self.maybe_rebuild()
        with self._lock:
            if self._matrix is None or self._vectorizer is None:
                return []
            vectorizer, matrix, rows = self._vectorizer, self._matrix, list(self._rows)

        q = vectorizer.transform([query[:3000]])
        scores = (matrix @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][: max(top_k * 4, top_k)]
        out: list[Hit] = []
        for idx in order:
            if scores[idx] < min_score:
                continue
            row = rows[int(idx)]
            if exclude_id is not None and int(row["id"]) == exclude_id:
                continue
            out.append(Hit(
                id=int(row["id"]), source_id=row["source_id"],
                source_name=row["source_name"] or "", role=row["role"] or "news",
                url=row["url"] or "", title=row["title"] or "",
                text=row["text"], score=round(float(scores[idx]), 3)))
            if len(out) >= top_k:
                break
        return out
