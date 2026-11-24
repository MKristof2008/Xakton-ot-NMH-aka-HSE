"""RAG-слой: отбор релевантных фрагментов источников перед оценкой.

Когда тексты источников не помещаются в контекст модели целиком, вместо
грубой обрезки по лимиту символов мы выбираем из каждого источника
предложения, наиболее близкие к утверждениям и поисковым запросам.
Модель получает компактный, но содержательный блок — это дешевле и
снижает риск «потери середины» длинного контекста.

Ничего не выдумывает: фрагменты — дословные куски исходного текста,
поэтому проверка цитат на дословность продолжает работать.
"""
from __future__ import annotations

import re
from typing import Sequence

from . import checks
from .schemas import Source

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")

FRAGMENTS_PER_SOURCE = 4
FRAGMENT_SOURCE_LIMIT = 1200
# Если суммарный текст короче — ретрив не нужен, отдаём полные тексты.
FULL_TEXT_BUDGET = 6000


def split_fragments(content: str) -> list[str]:
    """Делит текст на предложения/абзацы без потери содержимого."""
    parts = [checks.norm_ws(p) for p in _SENT_SPLIT.split(content or "")]
    return [p for p in parts if p]


def _claim_terms(claims: Sequence) -> set[str]:
    terms: set[str] = set()
    for claim in claims:
        terms |= checks.content_tokens(getattr(claim, "text", ""))
        for query in getattr(claim, "queries", []) or []:
            terms |= checks.content_tokens(query)
    return terms


def rank_fragments(
    fragments: Sequence[str],
    terms: set[str],
    claim_numbers: set[str],
    claim_years: set[int],
    limit: int = FRAGMENTS_PER_SOURCE,
) -> list[str]:
    """Топ фрагментов по лексическому пересечению; порядок — как в источнике.

    Бонус за совпадение чисел и годов: в новостях о ставках, суммах и датах
    именно они несут фактическую нагрузку.
    """
    scored = []
    for index, frag in enumerate(fragments):
        score = len(terms & checks.content_tokens(frag))
        score += len(claim_numbers & checks.numbers(frag))
        score += len(claim_years & checks.years(frag))
        scored.append((-score, index, frag))
    scored.sort(key=lambda item: (item[0], item[1]))
    best = [item for item in scored[:limit] if item[0] < 0]
    if not best:  # ничего не совпало — берём начало документа как контекст
        best = scored[: min(2, len(scored))]
    return [frag for _, _, frag in sorted(best, key=lambda item: item[1])]


def retrieved_sources_block(
    sources: Sequence[Source],
    claims: Sequence,
    per_source: int = FRAGMENTS_PER_SOURCE,
) -> tuple[str, bool]:
    """Строит блок «релевантные фрагменты» для промпта B.

    Возвращает (текст блока, признак, что ретрив применён и модель видела
    не полные тексты). Если весь материал и так помещается в бюджет,
    возвращает ("", False) — тогда pipeline использует обычный sources_block.
    """
    total = sum(len(s.content or "") for s in sources)
    if total <= FULL_TEXT_BUDGET:
        return "", False

    terms = _claim_terms(claims)
    numbers: set[str] = set()
    years: set[int] = set()
    for claim in claims:
        numbers |= checks.numbers(getattr(claim, "text", ""))
        years |= checks.years(getattr(claim, "text", ""))

    from .context_build import _TEMPLATE  # тот же формат <source …>

    parts = []
    for source in sources:
        fragments = split_fragments(source.content)
        picked = rank_fragments(fragments, terms, numbers, years, per_source)
        content = "\n".join(f"… {frag}" for frag in picked)
        if len(content) > FRAGMENT_SOURCE_LIMIT:
            content = content[:FRAGMENT_SOURCE_LIMIT].rstrip() + " …"
        published = f' published_at="{source.published_at}"' if source.published_at else ""
        parts.append(_TEMPLATE.format(
            id=source.id,
            kind=source.content_kind.value,
            origin=source.origin.value,
            title=source.title,
            published=published,
            content=content,
        ))
    header = ("Ниже — НАИБОЛЕЕ РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ источников, отобранные "
              "автоматически под утверждения. Это дословные выдержки, но не "
              "полные тексты документов.\n\n")
    return header + "\n\n".join(parts), True
