"""Присланная ссылка не должна превращаться в «фейк» сама по себе.

Сценарий: пользователь присылает адрес новости. Страница скачалась,
статью разобрали, но независимых подтверждений поиск не нашёл — свежая
новость ещё не перепечатана. Строгий режим раньше ставил «фейк»: материал
сам оказывался уликой против себя. Честный ответ — «непонятно» с
объяснением, что подтверждений нет.

При этом сатира и домены из чёрного списка по-прежнему дают «фейк»:
там основание — репутация источника, а не отсутствие поиска.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from lakmus import reputation
from lakmus.full_pipeline import run_check
from lakmus.sources import FetchedSource, fetch_article


def run(text: str, **kwargs):
    saved = {k: os.environ.pop(k, None) for k in ("LLM_API_KEY", "SERPER_API_KEY")}
    try:
        return asyncio.run(run_check(text, **kwargs))
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_link_without_confirmation_is_not_fake(monkeypatch):
    """Ссылка без подтверждений — «непонятно», а не «фейк»."""
    article_text = (
        "Администрация президента США планирует объявить о повышении импортных "
        "пошлин на товары из Китая и Евросоюза. Об этом сообщает издание "
        "The Wall Street Journal со ссылкой на источники."
    )
    extra = [FetchedSource(
        id="u1", title="Материал по ссылке", url="https://lenta.ru/news/2026/11/24/x/",
        content=article_text, domain="lenta.ru",
    )]
    outcome = run(article_text, url="https://lenta.ru/news/2026/11/24/x/",
                  extra_sources=extra)
    assert outcome.verdict != "фейк", outcome.reasoning.get("summary")
    assert outcome.reasoning["verdict"] == outcome.verdict


def test_text_without_sources_still_uses_strict_mode():
    """Обычный текст без ссылки строгий режим не потерял."""
    outcome = run("ЦБ ввёл комиссию за снятие наличных 12 процентов с апреля.")
    assert outcome.verdict in ("фейк", "непонятно")


def test_satire_link_is_still_fake():
    """Сатирическое издание — основание в репутации домена, а не в поиске."""
    text = "Пенсионерам выплатят по миллиону рублей на Новый год."
    outcome = run(text, url="https://panorama.pub/news/123")
    assert outcome.verdict == "фейк"


def test_reputation_marks_satire_domain():
    info = reputation.classify("https://panorama.pub/news/1")
    assert info.reputation == "satire"
