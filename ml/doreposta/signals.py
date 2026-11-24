"""Детерминированные сигналы для текста (ТЗ «Лакмус», разделы 6.2, 6.4).

Сигнал — независимая улика с честной оценкой надёжности: вердикт, сила
и reliability. Сигналы не вызывают модель и ничего не выдумывают —
считают то, что можно посчитать кодом.

T4 (риторика давления) реализован полностью здесь: reliability ≤ 0.3,
вердикт не меняет score напрямую — это флаг «стиль, характерный для
манипуляций», а не признак лжи.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from . import checks

_STAGE_PROVENANCE = "provenance"
_STAGE_ARTIFACTS = "artifacts"
_STAGE_SOURCES = "sources"
_STAGE_CONTEXT = "context"

VERDICT_REAL = "supports_real"
VERDICT_FAKE = "supports_fake"
VERDICT_NEUTRAL = "neutral"
VERDICT_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SignalResult:
    id: str
    name: str
    stage: str
    verdict: str
    strength: float          # 0–1, насколько выражен эффект
    reliability: float       # 0–1, насколько сигналу можно верить
    summary: str             # 1–2 предложения по-русски, без жаргона
    details: dict = field(default_factory=dict)


# --- T4: риторика давления ----------------------------------------------

_PRESSURE_PATTERNS = {
    "urgency_forward": re.compile(
        r"срочно|перешли|перешлите|переслать|максимальный репост|распростран", re.IGNORECASE),
    "anonymous_source": re.compile(
        r"(источник\w*\s+(?:пожелал\w*\s+)?анонимн|по\s+(?:неофициальным|секретным)\s+данным|"
        r"говорят,?\s+что|все\s+молчат|это\s+скрывают)", re.IGNORECASE),
    "no_dates_refs": re.compile(r"(1[89]\d{2}|20\d{2})\b"),
    "emotional_caps": re.compile(r"\b[А-ЯЁA-Z]{4,}\b"),
    "exclamation": re.compile(r"!{1,}|!?‼"),
    "clickbait": re.compile(
        r"(шок|сенсаци|вы\s+не\s+поверите|этого\s+от\s+нас\s+скрывали|срочно\s+всем\s+читать)",
        re.IGNORECASE),
}

T4_RELIABILITY = 0.3  # по ТЗ: флаг стиля, а не признак лжи


def rhetoric_signal(text: str) -> SignalResult:
    """T4: стиль давления и манипуляций в тексте."""
    text = text or ""
    flags: dict[str, bool] = {}
    flags["срочность и призывы переслать"] = bool(_PRESSURE_PATTERNS["urgency_forward"].search(text))
    flags["анонимные «источники»"] = bool(_PRESSURE_PATTERNS["anonymous_source"].search(text))
    flags["кликбейт"] = bool(_PRESSURE_PATTERNS["clickbait"].search(text))
    words = text.split()
    caps = len(_PRESSURE_PATTERNS["emotional_caps"].findall(text))
    flags["эмоциональный КАПС"] = bool(words) and caps / max(len(words), 1) > 0.05
    flags["много восклицаний"] = len(_PRESSURE_PATTERNS["exclamation"].findall(text)) >= 3
    # даты и ссылки — признаки проверяемости; их отсутствие при длинном тексте — флаг
    has_date = bool(_PRESSURE_PATTERNS["no_dates_refs"].search(text))
    has_link = "http" in text.lower() or ".ru" in text.lower() or ".com" in text.lower()
    flags["нет дат и ссылок"] = len(text) > 200 and not has_date and not has_link

    fired = [name for name, hit in flags.items() if hit]
    if not fired:
        return SignalResult(
            id="rhetoric", name="Риторика давления", stage=_STAGE_CONTEXT,
            verdict=VERDICT_NEUTRAL, strength=0.0, reliability=T4_RELIABILITY,
            summary="Явных признаков манипулятивного стиля не найдено.",
            details={"flags": flags},
        )
    strength = min(1.0, len(fired) / 3.0)
    summary = ("Стиль, характерный для манипуляций: " + ", ".join(fired) +
               ". Это не доказательство лжи — повод проверить факты внимательнее.")
    return SignalResult(
        id="rhetoric", name="Риторика давления", stage=_STAGE_CONTEXT,
        verdict=VERDICT_FAKE, strength=strength, reliability=T4_RELIABILITY,
        summary=summary, details={"flags": flags, "fired_count": len(fired)},
    )


# --- репутация домена источника -------------------------------------------

# Известные сатирические ресурсы и распространители выдуманных новостей.
# Поддомены покрываются: panorama.pub ловит и www.panorama.pub.
SATIRE_DOMAINS = {
    "panorama.pub",
    "newsmuzz.com",
    "sm-news.ru",
    "oilrus.info",
    "therussiantimes.com",
    "ria-smi.ru",
    "ivpress.ru",
}

SATIRE_RELIABILITY = 0.9  # список поддерживается вручную — сигнал сильный


def _host_of(url: str) -> str:
    """Домен без схемы, www. и пути."""
    host = re.sub(r"^[a-z]+://", "", (url or "").strip().lower())
    host = host.split("/")[0].split("?")[0].split(":")[0]
    return host.removeprefix("www.")


def domain_reputation_signal(url: str) -> SignalResult | None:
    """Если URL принадлежит известному сатирику — сильный сигнал supports_fake."""
    host = _host_of(url)
    if not host:
        return None
    hit = host in SATIRE_DOMAINS or any(
        host.endswith("." + d) for d in SATIRE_DOMAINS)
    if not hit:
        return None
    return SignalResult(
        id="domain_reputation", name="Репутация источника", stage=_STAGE_PROVENANCE,
        verdict=VERDICT_FAKE, strength=0.95, reliability=SATIRE_RELIABILITY,
        summary=(f"Домен {host} — известный сатирический ресурс: его «новости» "
                 "выдуманы и публикуются как юмор, а не как факты."),
        details={"host": host},
    )


# --- локальный ML-классификатор фейков ------------------------------------

ML_MODEL_RELIABILITY = 0.5  # локальная модель — вспомогательный сигнал, не истина


def ml_fake_signal(text: str) -> SignalResult | None:
    """Вероятность фейка по локальному классификатору (fake_model.joblib).

    None — если модель не обучена или текст слишком короткий. Модель
    выступает слабым сигналом: уверенный результат двигает score,
    пограничный остаётся neutral.
    """
    from .fake_classifier import fake_probability
    p = fake_probability(text)
    if p is None:
        return None
    if p >= 0.7:
        return SignalResult(
            id="ml_classifier", name="ML-классификатор фейков", stage=_STAGE_ARTIFACTS,
            verdict=VERDICT_FAKE, strength=min(0.9, p), reliability=ML_MODEL_RELIABILITY,
            summary=(f"Локальная модель оценивает текст как фейковый с вероятностью "
                     f"{p:.0%}. Это статистическая оценка стиля, а не проверка фактов."),
            details={"fake_probability": round(p, 3)},
        )
    if p <= 0.3:
        return SignalResult(
            id="ml_classifier", name="ML-классификатор фейков", stage=_STAGE_ARTIFACTS,
            verdict=VERDICT_REAL, strength=min(0.9, 1 - p), reliability=ML_MODEL_RELIABILITY,
            summary=(f"Локальная модель оценивает текст как правдивый с вероятностью "
                     f"{1 - p:.0%}. Это статистическая оценка стиля, а не проверка фактов."),
            details={"fake_probability": round(p, 3)},
        )
    return SignalResult(
        id="ml_classifier", name="ML-классификатор фейков", stage=_STAGE_ARTIFACTS,
        verdict=VERDICT_NEUTRAL, strength=0.3, reliability=ML_MODEL_RELIABILITY,
        summary=f"Локальная модель не уверена: вероятность фейка {p:.0%}.",
        details={"fake_probability": round(p, 3)},
    )


# --- сигналы из результата оценки утверждений ----------------------------

_STATUS_TO_VERDICT = {
    "supported": VERDICT_REAL,
    "contradicted": VERDICT_FAKE,
    "context": VERDICT_NEUTRAL,
    "insufficient": VERDICT_NEUTRAL,
}


def claims_signals(evaluation, claims: Sequence) -> list[SignalResult]:
    """Каждое оценённое утверждение — сигнал стадии sources.

    Reliability зависит от качества доказательств: есть ли независимые
    источники (перепечатки уже убраны дедупликацией до вызова модели).
    """
    out: list[SignalResult] = []
    for cr in evaluation.claims:
        source_ids = {ev.source_id for ev in cr.evidence}
        verdict = _STATUS_TO_VERDICT[cr.status.value]
        if not cr.evidence:
            reliability = 0.3
            strength = 0.4
        else:
            reliability = min(0.9, 0.4 + 0.25 * len(source_ids))
            strength = 0.8 if verdict in (VERDICT_REAL, VERDICT_FAKE) else 0.4
        short = checks.norm_ws(cr.text)[:80]
        out.append(SignalResult(
            id=f"claim_{cr.id}", name=f"Проверка: «{short}»", stage=_STAGE_SOURCES,
            verdict=verdict, strength=0.9, reliability=reliability,
            summary=cr.explanation, details={"status": cr.status.value,
                                             "sources": sorted(source_ids)},
        ))
    return out
