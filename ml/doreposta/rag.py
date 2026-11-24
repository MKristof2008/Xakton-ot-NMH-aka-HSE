"""RAG-база знаний для фактчекинга.

Простой локальный retrieval без внешних сервисов: TF-IDF (word+char) по
документам базы + косинусная близость. Документы — реальные новости,
сатира (с пометкой) и опровержения фейков.

Хранение:
- kb_docs.jsonl — сами документы {id, title, url, date, kind, text}
- kb_index.joblib — векторизатор + матрица (перестраивается скриптом
  ml/build_kb.py после добавления документов)

Если индекса нет — retrieve() возвращает пустой список, пайплайн
продолжает работать без RAG.
"""
from __future__ import annotations

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(_HERE, "kb_docs.jsonl")
INDEX_PATH = os.path.join(_HERE, "kb_index.joblib")

# снижаем вес сатирических документов: их стиль похож на фейк,
# но факты внутри выдуманы — они полезны как «антипримеры», а не доказательства
_KIND_RELIABILITY = {"news": 1.0, "debunk": 1.0, "satire": 0.2}

_docs_cache: list[dict] | None = None
_index_cache = None
_index_mtime: float | None = None


def load_docs() -> list[dict]:
    """Прочитать базу знаний (кэшируется по mtime файла)."""
    global _docs_cache
    if not os.path.exists(DOCS_PATH):
        _docs_cache = []
        return _docs_cache
    if _docs_cache is not None:
        return _docs_cache
    docs = []
    with open(DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("text"):
                docs.append(d)
    _docs_cache = docs
    return docs


def _load_index():
    """Ленивая загрузка индекса; перечитываем, если файл обновился."""
    global _index_cache, _index_mtime
    if not os.path.exists(INDEX_PATH):
        return None
    mtime = os.path.getmtime(INDEX_PATH)
    if _index_cache is not None and _index_mtime == mtime:
        return _index_cache
    try:
        import joblib
        _index_cache = joblib.load(INDEX_PATH)
        _index_mtime = mtime
    except Exception:
        _index_cache = None
    return _index_cache


def build_index() -> dict:
    """Перестроить индекс по kb_docs.jsonl. Возвращает статистику."""
    import joblib
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = load_docs()
    if not docs:
        return {"ok": False, "error": "база знаний пуста", "docs": 0}
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=1, sublinear_tf=True, max_features=300_000)
    matrix = vectorizer.fit_transform(d["text"][:3000] for d in docs)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, INDEX_PATH)
    global _index_cache, _index_mtime
    _index_cache = None  # сброс, чтобы _load_index перечитал свежий файл
    by_kind: dict[str, int] = {}
    for d in docs:
        by_kind[d.get("kind", "?")] = by_kind.get(d.get("kind", "?"), 0) + 1
    return {"ok": True, "docs": len(docs), "by_kind": by_kind}


def retrieve(query: str, top_k: int = 3, min_score: float = 0.08) -> list[dict]:
    """Найти документы базы, похожие на запрос.

    Возвращает [{doc, score}] по убыванию близости. Пустой список —
    если индекса нет или ничего релевантного не нашлось.
    """
    index = _load_index()
    docs = load_docs()
    if index is None or not docs or not query.strip():
        return []
    q = index["vectorizer"].transform([query[:3000]])
    matrix = index["matrix"]
    scores = (matrix @ q.T).toarray().ravel()
    order = scores.argsort()[::-1][:top_k]
    out = []
    for i in order:
        if scores[i] < min_score:
            continue
        d = docs[i]
        out.append({
            "score": round(float(scores[i]), 3),
            "id": d.get("id", ""),
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "date": d.get("date", ""),
            "kind": d.get("kind", "news"),
            "reliability": _KIND_RELIABILITY.get(d.get("kind", "news"), 1.0),
            "text": d.get("text", "")[:1500],
        })
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, min_len: int = 200, max_len: int = 800) -> list[str]:
    """Нарезка на смысловые фрагменты для индексации (по предложениям)."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    out, buf = [], ""
    for s in sents:
        buf = (buf + " " + s).strip()
        if len(buf) >= min_len:
            out.append(buf[:max_len])
            buf = ""
    if len(buf) >= min_len // 2:
        out.append(buf)
    return out
