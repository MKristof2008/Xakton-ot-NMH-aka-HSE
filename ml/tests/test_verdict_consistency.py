"""Вердикт и объяснение не должны противоречить друг другу.

Пользователь читает две вещи: крупный вердикт («скорее фейк») и словесный
разбор под ним. Если рубрика считает по статусам утверждений, а правила —
по признакам жанра, эти два вывода расходятся, и в одном ответе стоят
«фейк» и «это не значит, что новость ложная». Тесты фиксируют согласие.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from lakmus.full_pipeline import run_check


def run(text: str, **kwargs):
    """Офлайн-проверка: без ключей модели и поиска, детерминированно."""
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "SERPER_API_KEY")}
    try:
        return asyncio.run(run_check(text, **kwargs))
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


@pytest.mark.parametrize("text", [
    # вирусная выдумка без источников: вердикт ставится по признакам жанра
    "СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года. Перешлите всем!",
    # известный факт из курируемой базы
    "МРОТ с 1 января 2025 года — 22 440 рублей.",
    # прогноз: «фейк» запрещён правилом темы
    "Завтра в Москве ожидается снег.",
    # отрицание: правда, а не фейк
    "ЦБ не запрещал наличные.",
    # строгий режим: тема в базе есть, события в ней нет
    "В России отменяют ЕГЭ с 2027 года.",
    # составное утверждение с отрицанием второй части
    "ЦБ запретил наличные, но потом отменил решение.",
])
def test_reasoning_verdict_matches_shown_verdict(text):
    outcome = run(text)
    assert outcome.reasoning["verdict"] == outcome.verdict, (
        f"вердикт «{outcome.verdict}», а объяснение говорит "
        f"«{outcome.reasoning['verdict']}»"
    )


def test_fake_verdict_does_not_say_news_is_not_false():
    """«Фейк» и «это не значит, что новость ложная» в одном ответе — брак."""
    outcome = run(
        "СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года. Перешлите всем!"
    )
    if outcome.verdict != "фейк":
        pytest.skip("вердикт зависит от базы фактов окружения")
    text = " ".join(outcome.limitations) + " " + outcome.reasoning["summary"]
    assert "не значит, что новость ложная" not in text
    assert "ограничение поиска" not in text


def test_forecast_is_never_fake():
    outcome = run("Завтра в Москве ожидается снег.")
    assert outcome.verdict != "фейк"
