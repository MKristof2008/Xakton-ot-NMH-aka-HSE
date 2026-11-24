"""Локальный классификатор фейковых новостей.

Артефакт: fake_model.joblib рядом с этим файлом (обучается
`ml/train_fake_classifier.py` на Toloka/russian_fake_news).

Если модель не обучена — все функции тихо возвращают None,
пайплайн продолжает работать без этого сигнала.
"""
from __future__ import annotations

import os

_MODEL = None
_LOADED = False
_MODEL_MTIME = None
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_model.joblib")


def _load():
    """Ленивая загрузка модели (joblib). Если файл модели обновился
    (переобучение по обратной связи) — перезагружаем автоматически."""
    global _MODEL, _LOADED, _MODEL_MTIME
    if not os.path.exists(_MODEL_PATH):
        _LOADED = True
        _MODEL = None
        return None
    mtime = os.path.getmtime(_MODEL_PATH)
    if _LOADED and _MODEL_MTIME == mtime:
        return _MODEL
    try:
        import joblib
        _MODEL = joblib.load(_MODEL_PATH)
        _MODEL_MTIME = mtime
    except Exception:
        _MODEL = None
    _LOADED = True
    return _MODEL


def reload_model() -> None:
    """Принудительная перезагрузка (после дообучения)."""
    global _LOADED
    _LOADED = False
    _load()


def is_available() -> bool:
    return _load() is not None


def fake_probability(text: str) -> float | None:
    """P(текст — фейковая новость) по локальной модели, 0..1. None, если нет модели."""
    model = _load()
    if model is None or not text or len(text.strip()) < 20:
        return None
    try:
        return float(model.predict_proba([text])[0][1])
    except Exception:
        return None
