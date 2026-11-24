"""Полный пайплайн проверки: извлечение → поиск источников → оценка → валидация.

Три звена, каждое важно:

  1. Извлечь утверждение. Нужна модель (LLM). Без ключа используется
     детерминированный экстрактор по маркерам (года, числа, «запрет»,
     «повысят») — он грубее, но честно работает офлайн.
  2. Найти источники. Провайдеры из ``sources.py``: база фактов, каталог,
     ссылка пользователя. Модель ничего не выдумывает — ей дают текст.
  3. Оценить и проверить. Модель выносит статус, но код перепроверяет:
     цитата должна дословно найтись в источнике. Не нашлась — статус
     понижается. Так ловятся галлюцинации.

Режим определяется наличием ключа: есть LLM_API_KEY — работает модель,
нет — детерминированный путь. Оба режима возвращают один контракт.
"""

from __future__ import annotations

import os
import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import bank as bank_module
from . import reputation
from . import verdict_rules
from .reasoning import judge
from .tags import classify as classify_tag
from .tariffs import BASE, Tier
from .sources import (
    CatalogProvider,
    FetchedSource,
    LocalKbProvider,
    NullProvider,
    SerperProvider,
    SourceProvider,
    UrlProvider,
    is_trusted,
)

# --------------------------------------------------------------- контракт

@dataclass
class ClaimFinding:
    """Одно проверенное утверждение."""

    text: str
    status: str  # supported | contradicted | context | insufficient
    explanation: str = ""
    quote: str = ""
    source_title: str = ""
    source_url: str = ""
    trusted: bool = False
    limitations: list[str] = field(default_factory=list)
    #: Что человеку делать дальше. Появляется с тарифа «Плюс».
    action: str = ""
    #: Готовая фраза для пересылки в чат. Появляется на «Максе».
    resend: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "text": self.text,
            "status": self.status,
            "explanation": self.explanation,
            "quote": self.quote,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "trusted": self.trusted,
            "limitations": self.limitations,
        }
        # Пустые поля не отдаём: у базы их нет, и это честно.
        if self.action:
            out["action"] = self.action
        if self.resend:
            out["resend"] = self.resend
        return out


@dataclass
class CheckOutcome:
    """Результат полной проверки."""

    claims: list[ClaimFinding] = field(default_factory=list)
    verdict: str = "непонятно"  # правда | фейк | непонятно
    level: str = "fog"
    score: int = 50
    uncertainty: float = 1.0
    mode: str = "offline"  # offline | llm
    sources_used: int = 0
    limitations: list[str] = field(default_factory=list)
    duration_ms: int = 0
    #: Тема утверждения (система тегов) — от неё зависят правила проверки.
    tag: dict[str, Any] = field(default_factory=dict)
    #: Пошаговая цепочка: почему именно такой вердикт и балл.
    reasoning: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "claims": [c.as_dict() for c in self.claims],
            "verdict": self.verdict,
            "level": self.level,
            "score": self.score,
            "uncertainty": round(self.uncertainty, 4),
            "mode": self.mode,
            "sources_used": self.sources_used,
            "limitations": self.limitations,
            "duration_ms": self.duration_ms,
        }
        if self.tag:
            out["tag"] = self.tag
        if self.reasoning:
            out["reasoning"] = self.reasoning
        return out


# --------------------------------------------------- детерминированный путь

#: Маркеры, по которым текст похож на проверяемое утверждение.
_CLAIM_MARKERS = (
    "запрет", "запрещ", "запретил", "запретят", "отмен", "повыс", "пониз", "введ",
    "введут", "увелич", "сниз", "штраф", "ставк", "тариф", "выплат",
    "пенси", "налог", "разреш", "обяза", "подорожа", "подарож", "мрот",
)

_NEGATIONS = ("не ", "нет ", "никогда", "не будет", "не принимал", "не планир")

#: Противопоставление внутри одного утверждения: «запретил, но отменил».
_CONTRAST = re.compile(r"\b(но|однако|зато|хотя|впрочем)\b", re.I)

#: Признаки будущего: прогноз, обещание, ожидание. Такое нельзя назвать
#: ложью — событие ещё не наступило.
#: Признаки будущего: прогноз, обещание, ожидание. Такое нельзя назвать
#: ложью — событие ещё не наступило.
_FUTURE = re.compile(
    "(завтра|послезавтра|ожидается|ожидают|объявят|"
    "планируют|планируется|собираются|в следующем году|со следующей недели|"
    "скоро|в ближайшее время|подорожает|подешевеет|"
    "упадёт|упадут|снизится|введут|отменят|повысят|снизят|"
    # «повысится», но не «повысил»: прошедшее время — уже случившийся факт.
    "вырастет|вырастут|повысится|прогнозируется|по прогнозу)",
    re.I,
)


#: Безымянные заглушки: утверждение о «компании X» проверить нельзя.
_VAGUE = re.compile(
    r"\b(компания\s+[a-zа-яё]\b|некая\s+компания|некий|некие|"
    r"один\s+человек|какой-то)\b",
    re.I,
)


def _is_vague(text: str) -> bool:
    return bool(_VAGUE.search(text or ""))


def _is_future(text: str) -> bool:
    return bool(_FUTURE.search(text or ""))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 12]


def extract_offline(text: str, limit: int = 3) -> list[str]:
    """Утверждения без модели: предложения с числами или маркерами.

    Это заведомо грубее модели: не отсекает иронию, не понимает контекст.
    Зато работает офлайн и не выдумывает.
    """
    out: list[str] = []
    for sentence in _split_sentences(text):
        low = _norm(sentence)
        has_digit = bool(re.search(r"\d", sentence))
        has_marker = any(m in low for m in _CLAIM_MARKERS)
        # Короткие сообщения без маркеров всё равно проверяем: «Вчера
        # открыли мост» — утверждение, даже если в нём нет числа.
        if has_digit or has_marker or len(sentence) >= 20:
            if sentence not in out:
                out.append(sentence)
        if len(out) >= limit:
            break
    return out


def _quote_in_source(quote: str, source_text: str) -> bool:
    """Дословная ли цитата в источнике (с нормализацией оформления)."""
    def norm(value: str) -> str:
        value = value.translate(
            str.maketrans({
                "«": '"', "»": '"', "„": '"', "“": '"', "”": '"',
                "—": "-", "–": "-", "’": "'", "‘": "'",
            })
        )
        value = re.sub(r"\s+", " ", value).strip().lower().replace("ё", "е")
        # Кавычки вокруг отдельных слов не должны ломать поиск: убираем их,
        # но только прямые двойные, оставшиеся после нормализации.
        value = value.replace('"', " ").replace("'", " ")
        return re.sub(r"\s+", " ", value).strip()

    needle = norm(quote)
    if len(needle) < 8:
        return False
    return needle in norm(source_text)


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for raw in re.findall(r"\d[\d\s]*(?:[.,]\d+)?", text or ""):
        try:
            out.add(round(float(raw.replace(" ", "").replace(",", ".")), 3))
        except ValueError:
            continue
    return out


_MONTHS = (
    "январ", "феврал", "март", "апрел", "ма", "июн", "июл",
    "август", "сентябр", "октябр", "ноябр", "декабр",
)

#: «25 октября», «1 марта» — число перед названием месяца это день, не значение.
_DAY_MONTH = re.compile(
    r"\b(\d{1,2})\s+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|"
    r"сентябр|октябр|ноябр|декабр)",
    re.I,
)


def _content_numbers(text: str) -> set[float]:
    """Значимые числа: годы и дни месяца не в счёт.

    «25 октября 2024 года ставка 21%» → {21.0}: день и год уходят,
    процент остаётся. День определяем по позиции — число перед названием
    месяца, а не по диапазону 1..31, иначе процент «21%» тоже выпадал бы.
    """
    days = {float(m.group(1)) for m in _DAY_MONTH.finditer(text or "")}
    out: set[float] = set()
    for value in _numbers(text):
        if 1900 <= value <= 2100:
            continue
        if value in days:
            continue
        out.add(value)
    return out


def _negation_matters(claim: str, source_text: str) -> bool:
    """Говорит ли источник о том же и с противоположным знаком.

    Проверять «есть ли в тексте слово не» бессмысленно: в статье на
    тысячу слов оно встретится всегда, и утверждение «открыли мост»
    превращалось в «фейк». Смотрим только предложения, где речь о том же:
    если в них есть отрицание, а в утверждении нет (или наоборот) — это
    настоящее расхождение.
    """
    claim_words = {w for w in re.findall(r"[а-яa-z]{5,}", _norm(claim))}
    if not claim_words:
        return False

    claim_negated = _is_negated(claim)
    for sentence in re.split(r"[.!?]+", source_text or ""):
        words = {w for w in re.findall(r"[а-яa-z]{5,}", _norm(sentence))}
        if not words:
            continue
        if len(claim_words & words) / len(claim_words) < 0.5:
            continue
        if _is_negated(sentence) != claim_negated:
            return True
    return False


def _is_negated(text: str) -> bool:
    low = _norm(text)
    return any(m in low for m in _NEGATIONS)


def _apply_text_guards(text: str, finding: ClaimFinding) -> ClaimFinding:
    """Защиты, которые зависят от исходного текста, а не от источника.

    Вынесены в функцию: раньше они жили только в полном пути и в быстром
    (по базе фактов) не применялись вообще.
    """
    if _is_vague(text):
        finding.status = "insufficient"
        finding.explanation = "В утверждении нет конкретных имён и чисел: проверить нечего."
        return finding
    # Прогноз нельзя назвать ложью — но только когда он о том, чего ещё
    # нет. «Пенсионный возраст повысят до 70» опровергается действующим
    # законом: факт известен, и это не прогноз, а выдумка. Поэтому
    # опровержение базы фактов сильнее слова «повысят».
    # Прогноз нельзя назвать ложью. Исключение одно: база фактов уже знает
    # действующее значение, и прогноз ему прямо противоречит числом
    # («повысят до 70» против «сейчас 65»). Это не прогноз, а выдумка.
    contradicted_by_number = (
        finding.status == "contradicted"
        and bool(finding.quote)
        and bool(_content_numbers(text))
    )
    if _is_future(text) and finding.status == "contradicted" and not contradicted_by_number:
        finding.status = "insufficient"
        finding.explanation = "Это прогноз о будущем: проверить его сейчас нельзя."
    return finding


def judge_offline(claim: str, sources: Sequence[FetchedSource]) -> ClaimFinding:
    """Сверить утверждение с источниками без модели.

    Сначала пробуем базу фактов по её собственным правилам (там известны
    «опровергающие» записи и маркеры фактов). Если тема не найдена —
    работаем по общим правилам: пересечение слов, числа, отрицание.
    """
    kb_finding = _judge_with_kb(claim)
    if kb_finding is not None:
        return kb_finding
    return _judge_by_overlap(claim, sources)


def _judge_with_kb(claim: str) -> Optional[ClaimFinding]:
    """Проверка по курируемой базе — та же логика, что в сигнале kb_facts."""
    try:
        from .signals.kb_facts import check_against_topic, find_topics, load_kb
    except ImportError:
        return None
    try:
        kb = load_kb()
    except Exception:
        return None

    # Составное утверждение с противопоставлением («запретил, но отменил»)
    # само себе противоречит. Опровергать его одной записью базы нельзя:
    # правильный ответ — «непонятно», а не «фейк».
    compound = bool(_CONTRAST.search(claim or ""))

    topics = find_topics(kb, claim)
    if not topics:
        return None
    for topic in topics:
        verdict, _reliability, summary, details = check_against_topic(topic, claim)
        if details.get("reason") == "different_fact":
            continue
        if compound and verdict.value == "supports_fake":
            return ClaimFinding(
                text=claim,
                status="insufficient",
                explanation=(
                    "Утверждение составное: одна часть противоречит базе, "
                    "другая — нет. Нужна ручная проверка."
                ),
                source_title=str((details.get("matched_fact") or {}).get("source", "")),
                source_url=str((details.get("matched_fact") or {}).get("url", "")),
                trusted=False,
            )
        status = {
            "supports_real": "supported",
            "supports_fake": "contradicted",
        }.get(verdict.value, "insufficient")
        fact = details.get("matched_fact") or {}
        return ClaimFinding(
            text=claim,
            status=status,
            explanation=summary,
            quote=str(fact.get("claim", "")),
            source_title=str(fact.get("source", "")),
            source_url=str(fact.get("url", "")),
            trusted=is_trusted(str(fact.get("url", ""))),
        )
    return None


def _judge_by_overlap(claim: str, sources: Sequence[FetchedSource]) -> ClaimFinding:
    claim_numbers = _content_numbers(claim)
    claim_negated = _is_negated(claim)
    claim_words = {w for w in re.findall(r"[а-яa-z]{5,}", _norm(claim))}
    # Одно-два слова — не утверждение, а тема. «Путин показал» подходит
    # к любой новости про Путина, поэтому совпадение по ним ничего не
    # доказывает. Такие запросы уходят в «недостаточно данных».
    if len(claim_words) <= 2 and not claim_numbers:
        return ClaimFinding(
            text=claim,
            status="insufficient",
            explanation=(
                "Слишком короткое утверждение: одних этих слов "
                "недостаточно, чтобы подтвердить событие."
            ),
            limitations=["Нужны подробности: что, где и когда."],
        )

    best: Optional[ClaimFinding] = None
    for source in sources:
        text = _norm(source.content)
        source_words = {w for w in re.findall(r"[а-яa-z]{5,}", text)}
        if not claim_words or not source_words:
            continue
        overlap = len(claim_words & source_words) / len(claim_words)
        if overlap < 0.35:
            continue

        source_numbers = _content_numbers(source.content)
        source_negated = _is_negated(source.content)
        trusted = is_trusted(source.url)

        numbers_agree: Optional[bool] = None
        if claim_numbers and source_numbers:
            numbers_agree = all(
                any(abs(c - s) / max(abs(s), 1e-9) <= 0.005 for s in source_numbers)
                for c in claim_numbers
            )

        if numbers_agree is False:
            finding = ClaimFinding(
                text=claim,
                status="contradicted",
                explanation=f"Число в сообщении не совпадает с источником: {source.title}.",
                quote="",
                source_title=source.title,
                source_url=source.url,
                trusted=trusted,
            )
        elif claim_negated != source_negated and _negation_matters(claim, source.content):
            finding = ClaimFinding(
                text=claim,
                status="contradicted",
                explanation=f"Источник говорит обратное: {source.title}.",
                quote="",
                source_title=source.title,
                source_url=source.url,
                trusted=trusted,
            )
        else:
            # Пересечение слов — не доказательство. «Путин показал кулак»
            # и «Путин показал пенис» имеют общие слова, но это разные
            # события. Подтверждение возможно только с дословной цитатой,
            # поэтому здесь его не ставим вовсе.
            finding = ClaimFinding(
                text=claim,
                status="insufficient",
                explanation=(
                    f"В источнике «{source.title}» нет подтверждения: "
                    "событие не совпадает."
                ),
                quote="",
                source_title=source.title,
                source_url=source.url,
                trusted=trusted,
            )
        if best is None:
            best = finding

    if best is not None:
        return best
    return ClaimFinding(
        text=claim,
        status="insufficient",
        explanation="В доступных источниках нет подтверждения или опровержения.",
        limitations=["Источники не найдены — утверждение не проверено."],
    )


# ------------------------------------------------------------------- режим

_CACHE: dict[str, tuple[float, "CheckOutcome"]] = {}
_CACHE_TTL = 6 * 3600
_CACHE_MAX = 500


def _cache_key(
    text: str, tier_key: str = "base", url: Optional[str] = None
) -> str:
    """Ключ кэша: текст + ссылка + тариф.

    URL входит в ключ: иначе проверка присланной ссылки возвращала результат
    прошлого прогона того же текста без источников. Тариф тоже входит:
    у «Плюса» и «Макса» другой разбор, и отдавать им короткий ответ нельзя.
    """
    import hashlib

    norm_text = " ".join((text or "").lower().split())
    norm_url = (url or "").strip().lower()
    return hashlib.sha256(
        f"{norm_text}\x00{norm_url}\x00{tier_key}".encode()
    ).hexdigest()


def _cache_get(
    text: str, tier_key: str = "base", url: Optional[str] = None
) -> Optional["CheckOutcome"]:
    key = _cache_key(text, tier_key, url)
    hit = _CACHE.get(key)
    if hit is None:
        return None
    stamp, outcome = hit
    import time

    if time.time() - stamp > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return outcome


def _cache_put(
    text: str,
    outcome: "CheckOutcome",
    tier_key: str = "base",
    url: Optional[str] = None,
) -> None:
    import time

    if not (text or "").strip():
        return
    if len(_CACHE) >= _CACHE_MAX:
        oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest, None)
    _CACHE[_cache_key(text, tier_key, url)] = (time.time(), outcome)


def llm_enabled() -> bool:
    """Включён ли LLM. Только по переменным окружения, ключей в коде нет."""
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"))


def _is_suspicious_for_bank(text: str) -> bool:
    """Похоже ли утверждение на вирусную выдумку.

    Такие записи в банк не пишем: если система ошиблась и подтвердила
    фейк, запись закрепит ошибку навсегда — следующая такая же проверка
    ответит из банка «правда».
    """
    try:
        from . import fake_markers
    except ImportError:
        return False
    try:
        return bool(fake_markers.find_markers(text))
    except Exception:
        return False


def _remember(bank: Any, finding: "ClaimFinding", tag_match: Any) -> None:
    """Запомнить проверенный факт в банке знаний и сохранить файл.

    Пишем только то, что имеет основание: вердикт supported/contradicted
    и ссылку на источник. Ошибка записи не ломает проверку — банк нужен
    для ускорения, а не для ответа.
    """
    # Не запоминаем вирусные фейки: подтверждённая ошибка закрепится.
    if _is_suspicious_for_bank(finding.text):
        return
    try:
        saved = bank.learn(
            finding.text,
            finding.status,
            tag=getattr(getattr(tag_match, "tag", None), "key", ""),
            explanation=finding.explanation,
            quote=finding.quote,
            source_title=finding.source_title,
            source_url=finding.source_url,
        )
        if saved is not None:
            bank.save()
    except Exception:
        # Банк не критичен: если диск недоступен, проверка всё равно верна.
        pass


#: Строгий режим: если поиск был, но подтверждений нет — считаем фейком.
#: Включается STRICT_NOT_FOUND=1 (по умолчанию включён: так просил заказчик).
#: Отличие от обычного режима только в вердикте, объяснение остаётся честным.
def _strict_not_found() -> bool:
    return os.environ.get("STRICT_NOT_FOUND", "1") not in ("0", "false", "no")


def _level_for(
    findings: Sequence[ClaimFinding],
    *,
    searched: bool = False,
    strict_allowed: bool = True,
    sources: int = 0,
    trusted: int = 0,
    tag: Any = None,
    tag_confident: bool = False,
    kb_hit: bool = False,
    bank_hit: bool = False,
    rule_result: Any = None,
) -> tuple[str, str, int, float]:
    """Уровень, вердикт, балл и неуверенность из набора находок.

    Балл берётся не из таблицы «50 по умолчанию», а из рубрики
    (``reasoning.judge``): цитаты, согласие источников, надёжность доменов,
    полнота. Поэтому число объяснимо, а при отсутствии данных оно честно
    низкое, а не нейтральное.

    ``rule_result`` — результат явных правил (``verdict_rules``). Если
    правила нашли весомый признак, их вердикт главнее: они объясняют
    причину («следы генератора», «признаки вирусной выдумки»), тогда как
    рубрика только считает баллы.

    ``searched`` — были ли источники вообще. Только тогда отсутствие
    подтверждения можно считать признаком выдумки: если искать было негде,
    честный ответ — «непонятно».
    """
    if not findings:
        return "fog", "непонятно", 0, 1.0

    rubric = judge(
        claims=findings,
        sources=sources,
        trusted=trusted,
        tag=tag,
        tag_confident=tag_confident,
        kb_hit=kb_hit,
        bank_hit=bank_hit,
        strict_allowed=strict_allowed,
    )
    confidence = rubric.confidence

    supported = sum(1 for f in findings if f.status == "supported")
    contradicted = sum(1 for f in findings if f.status == "contradicted")
    unknown = sum(1 for f in findings if f.status == "insufficient")

    # Правила могут вынести вердикт сами — но только если он обоснован
    # сильным признаком, а не одним стилистическим маркером. Порог 55:
    # столько даёт абсурдная мера (0.7) или два независимых маркера (0.6).
    rule_verdict = getattr(rule_result, "verdict", "") if rule_result else ""
    rule_confidence = getattr(rule_result, "confidence", 0) if rule_result else 0
    if rule_verdict in ("фейк", "правда") and rule_confidence >= 55:
        level = "likely_fake" if rule_verdict == "фейк" else "likely_real"
        score = (
            max(0, 100 - rule_confidence)
            if rule_verdict == "фейк"
            else rule_confidence
        )
        return level, rule_verdict, score, 0.25

    if contradicted and not supported:
        # Доверие к утверждению — обратная величина уверенности опровержения.
        return "likely_fake", "фейк", max(0, 100 - confidence), 0.2
    if supported and not contradicted:
        return "likely_real", "правда", confidence, 0.2
    if supported and contradicted:
        return "fog", "непонятно", confidence, 0.6

    # Только нехватка данных. Строгий режим может счесть это фейком —
    # но лишь если поиск состоялся и тема не прогноз.
    if unknown:
        if searched and strict_allowed and _strict_not_found():
            return "likely_fake", "фейк", max(0, min(45, confidence)), 0.55
        return "fog", "непонятно", confidence, 0.7
    return "fog", "непонятно", confidence, 0.5


async def run_check(
    text: str,
    *,
    url: Optional[str] = None,
    extra_sources: Optional[Sequence[FetchedSource]] = None,
    kb_path: Any = None,
    tier: Tier = BASE,
) -> CheckOutcome:
    """Полная проверка текста. Единая точка входа для backend.

    Порядок: извлечь утверждения → собрать источники → оценить → валидация.
    ``tier`` — уровень тарифа: он задаёт модель, глубину ответа и объём
    источника. Вердикт от уровня не зависит, разбор — зависит.
    """
    import time

    started = time.monotonic()

    # Кэш: одинаковый текст должен давать одинаковый ответ. Поиск и модель
    # нестабильны между вызовами, а пользователь вправе получить тот же
    # результат, что и час назад.
    cached = _cache_get(text, tier.key, url)
    if cached is not None:
        cached.duration_ms = int((time.monotonic() - started) * 1000)
        return cached

    # 0. Тема утверждения. Определяется до поиска: от темы зависят правила
    #    проверки и список приоритетных источников. Без модели — по словам,
    #    чтобы тег был одинаковым между прогонами и тарифами.
    tag_match = classify_tag(text)

    # 0a. Банк знаний: то, что система уже проверяла. Точное совпадение
    #     утверждения даёт готовый ответ без сети — быстро и стабильно.
    bank = bank_module.shared()
    bank_entry = bank.lookup(text)
    if bank_entry is not None:
        finding = ClaimFinding(
            text=text[:300],
            status=bank_entry.status,
            explanation=bank_entry.explanation
            or "Утверждение уже проверялось ранее.",
            quote=bank_entry.quote,
            source_title=bank_entry.source_title,
            source_url=bank_entry.source_url,
            trusted=is_trusted(bank_entry.source_url),
        )
        level, verdict, score, uncertainty = _level_for(
            [finding],
            searched=True,
            sources=1,
            trusted=1 if finding.trusted else 0,
            tag=tag_match.tag,
            tag_confident=tag_match.is_confident,
            bank_hit=True,
        )
        reasoning = judge(
            claims=[finding],
            sources=1,
            trusted=1 if finding.trusted else 0,
            tag=tag_match.tag,
            tag_confident=tag_match.is_confident,
            bank_hit=True,
        )
        outcome = CheckOutcome(
            claims=[finding],
            verdict=verdict,
            level=level,
            score=score,
            uncertainty=uncertainty,
            mode="bank",
            sources_used=1,
            limitations=[],
            tag=tag_match.as_dict(),
            reasoning=reasoning.as_dict(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        _cache_put(text, outcome, tier.key, url)
        return outcome

    # 1. Курируемая база фактов — проверка до всякой модели.
    #    Так результат стабилен (модель не переформулирует утверждение в
    #    неузнаваемое) и быстр (0 вызовов модели на известных фактах).
    kb_direct = _judge_with_kb(text)
    if kb_direct is not None and kb_direct.status in ("supported", "contradicted"):
        # Быстрый путь тоже обязан соблюдать те же защиты, что и полный:
        # прогноз нельзя опровергать («мобилизацию объявят» — база
        # «опровергает», хотя событие ещё не наступило), а составное
        # утверждение нельзя подтверждать одной записью.
        kb_direct = _apply_text_guards(text, kb_direct)
        if kb_direct.status in ("supported", "contradicted"):
            # Правила вывода нужны и на быстром пути: без них вердикт
            # («фейк» из строгого режима) и объяснение («непонятно» по
            # рубрике) расходятся, и человек видит два разных вывода.
            kb_rules = verdict_rules.rules_for_text(
                claims=[kb_direct],
                sources=1,
                trusted=1 if kb_direct.trusted else 0,
                kb_hit=kb_direct.trusted,
                is_future=_is_future(text),
                is_vague=_is_vague(text),
                text=text,
            )
            level, verdict, score, uncertainty = _level_for(
                [kb_direct],
                searched=True,
                sources=1,
                trusted=1 if kb_direct.trusted else 0,
                tag=tag_match.tag,
                tag_confident=tag_match.is_confident,
                kb_hit=True,
                rule_result=kb_rules,
            )
            reasoning = judge(
                claims=[kb_direct],
                sources=1,
                trusted=1 if kb_direct.trusted else 0,
                tag=tag_match.tag,
                tag_confident=tag_match.is_confident,
                kb_hit=True,
                rule_result=kb_rules,
                # «Фейк» без опровержения — строгий режим, и объяснение
                # обязано называть эту причину, а не «признаков мало».
                strict_fake=verdict == "фейк" and kb_direct.status != "contradicted",
            )
            outcome = CheckOutcome(
                claims=[kb_direct],
                verdict=verdict,
                level=level,
                score=score,
                uncertainty=uncertainty,
                mode="offline",
                sources_used=1,
                limitations=list(kb_direct.limitations),
                tag=tag_match.as_dict(),
                reasoning=reasoning.as_dict(),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            _remember(bank, kb_direct, tag_match)
            _cache_put(text, outcome, tier.key, url)
            return outcome

    # 1. Утверждения
    claims: list[str] = []
    mode = "offline"
    llm_used = False
    evaluate_one_with_llm = None

    # Модель включается для любого текста: именно она отличает реальную
    # новость от выдумки. Раньше стоял порог в 200 символов — короткие
    # сообщения («комендантский час с 1 декабря») уходили в офлайн-путь
    # и получали «Туман» вместо «фейк». Экономия секунды не стоит
    # потерянного вердикта.
    if llm_enabled():
        try:
            from .llm_pipeline import evaluate_one_with_llm as _evaluate, extract_with_llm

            evaluate_one_with_llm = _evaluate
            # Одно короткое сообщение — одно утверждение. Дробить его на
            # три части вредно: части начинают противоречить друг другу,
            # и итог уходит в «Туман» из-за конфликта, которого нет.
            # Извлечение утверждений — всегда на базовом профиле: набор
            # утверждений должен совпадать на всех тарифах, иначе вердикт
            # разойдётся не из-за глубины, а из-за разбиения текста.
            if len(text) > 400:
                extracted = await extract_with_llm(text, tier=BASE) or []
                if extracted:
                    claims = extracted
                    mode = "llm"
            # Модель оценки нужна и без извлечения: если она не смогла
            # разбить текст на утверждения, оцениваем весь текст целиком.
            llm_used = True
        except Exception:
            claims = []
            evaluate_one_with_llm = None  # type: ignore

    if not claims:
        # Пустой текст — не утверждение. Заглушку не подставляем.
        if text.strip():
            claims = extract_offline(text) or [text.strip()]

    if not claims:
        return CheckOutcome(
            mode=mode,
            limitations=["В сообщении нет проверяемых утверждений."],
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # 2. Источники: база → интернет → ссылка пользователя.
    providers: list[SourceProvider] = [LocalKbProvider(kb_path)]
    if url:
        providers.insert(0, UrlProvider())
    if extra_sources:
        providers.append(CatalogProvider([{"url": s.url, "title": s.title} for s in extra_sources]))

    # Интернет подключаем только если есть ключ поиска и тема не покрыта
    # базой: на известных фактах лишние запросы стоят времени и денег.
    serper = SerperProvider()
    # Результат kb_direct уже посчитан выше — второй прогон по базе ничего
    # не меняет, а стоит миллисекунды на каждой проверке.
    if serper.enabled and kb_direct is None:
        providers.append(serper)

    # Провайдеры опрашиваются ПАРАЛЛЕЛЬНО. Раньше они шли по очереди, и
    # время складывалось: база 1 мс + интернет 3-10 с + ссылка пользователя
    # до 12 с. Пользователь видел «долго думает» даже когда база ответила
    # сразу. Теперь самый медленный провайдер задаёт общее время, а не сумма.
    async def _ask(provider: SourceProvider) -> list[FetchedSource]:
        query = url if provider.name == "url" and url else claims[0]
        try:
            return list(await provider.find(query, limit=3))
        except Exception:
            # Сбой одного провайдера не должен ронять проверку.
            return []

    per_provider = await asyncio.gather(*(_ask(p) for p in providers))

    sources: list[FetchedSource] = []
    seen_ids: set[str] = set()
    # Порядок провайдеров сохраняем: база и присланная ссылка весят больше
    # интернет-выдачи, поэтому идут первыми в списке источников.
    for found in per_provider:
        for item in found:
            if item.id not in seen_ids:
                seen_ids.add(item.id)
                sources.append(item)
        if len(sources) >= 5:
            break

    # 2a. Репутация доменов. Сатирический или известный фейковый сайт
    # нельзя использовать как доказательство: иначе модель «подтвердит»
    # выдумку ссылкой на саму выдумку. Такие страницы остаются только как
    # проверяемый объект — их содержимое помечается и не идёт в опору.
    satire_sources: list[FetchedSource] = []
    trusted_sources: list[FetchedSource] = []
    for source in sources:
        info = reputation.classify(source.url or source.domain)
        source.reputation = info.reputation  # type: ignore[attr-defined]
        if info.must_not_be_evidence:
            satire_sources.append(source)
        else:
            trusted_sources.append(source)
    # Модели отдаём только те источники, которым можно доверять как опоре.
    evidence_sources = trusted_sources

    # 3. Оценка
    findings: list[ClaimFinding] = []
    model_silent = False

    async def assess(claim: str) -> tuple[ClaimFinding, bool]:
        """Одно утверждение: база → источники → модель. Возвращает находку и
        признак «модель молчала»."""
        kb_finding = _judge_with_kb(claim)
        # База уверенно ответила — доверяем ей. Но если она нашла лишь
        # соседнюю тему («ЕГЭ» есть, а отмены нет), это не ответ:
        # отдаём решение модели.
        if kb_finding is not None and kb_finding.status in ("supported", "contradicted"):
            return kb_finding, False

        # Модель, а не эвристика, решает вердикт: пересечение слов не
        # отличает «указ о первом заседании» от «указ о роспуске».
        finding = ClaimFinding(text=claim, status="insufficient",
                               explanation="В доступных источниках нет подтверждения.")
        if llm_used and evaluate_one_with_llm is not None:
            answer = await evaluate_one_with_llm(
                claim, evidence_sources, tier, tag_key=tag_match.tag.key
            )
            if answer is None:
                return _judge_by_overlap(claim, sources), True
            finding.status = answer["status"]
            finding.explanation = answer["explanation"] or finding.explanation
            finding.quote = answer["quote"]
            # Уровень «Плюс» и выше приносит «что делать», «Макс» — ещё и
            # готовую формулировку для пересылки. Вердикт от этого не меняется.
            if answer.get("action"):
                finding.action = answer["action"]
            if answer.get("resend"):
                finding.resend = answer["resend"]
            # «Нет события в источниках» — это не опровержение. Опровергает
            # только прямая противоположность, её ставит модель сама.
            # Подтверждение обязано опираться на дословную цитату.
            # «В источниках что-то похожее есть» — не доказательство:
            # модель подтверждала и «воду на Марсе», и «доллар по 120».
            if finding.status == "supported":
                # Подтверждение без цитаты — не подтверждение. Модель
                # умеет «согласиться» с абсурдом, если в выдаче есть
                # что-то похожее по словам.
                source = next(
                    (s for s in evidence_sources
                     if finding.quote and _quote_in_source(finding.quote, s.content)),
                    None,
                )
                if source is None:
                    finding.status = "insufficient"
                    finding.explanation = (
                        "Подтверждения в источниках нет: дословной цитаты не нашлось."
                    )
                    finding.quote = ""
                else:
                    finding.source_title = source.title
                    finding.source_url = source.url
                    finding.trusted = is_trusted(source.url)
            elif finding.quote:
                source = next(
                    (s for s in evidence_sources
                     if _quote_in_source(finding.quote, s.content)),
                    None,
                )
                # Цитата без привязки к статусу вводит в заблуждение:
                # в карточке появляется фраза, которая ничего не доказывает.
                # Оставляем её только при осмысленном вердикте.
                if source is not None and finding.status in ("contradicted", "context"):
                    finding.source_title = source.title
                    finding.source_url = source.url
                    finding.trusted = is_trusted(source.url)
                else:
                    finding.quote = ""
        return finding, False

    # Утверждения оцениваются параллельно: три запроса к модели идут
    # одновременно, а не друг за другом.
    results = await asyncio.gather(*(assess(c) for c in claims))
    for finding, silent in results:
        findings.append(finding)
        model_silent = model_silent or silent

    # Защиты применяются ПОСЛЕ оценки: раньше они стояли выше и проходили
    # по ещё пустому списку findings — то есть не работали никогда.
    # _apply_text_guards снимает и безымянные утверждения, и прогнозы:
    # прогноз нельзя ни подтвердить, ни опровергнуть — событие не наступило.
    for finding in findings:
        _apply_text_guards(text, finding)

    # Если входная ссылка ведёт на сатирический или фейковый сайт —
    # это прямой ответ, независимо от того, что нашла модель.
    satire_note = ""
    if url:
        url_info = reputation.classify(url)
        if url_info.reputation == "satire":
            findings = [ClaimFinding(
                text=text[:300],
                status="contradicted",
                explanation=(
                    "Источник — сатирическое издание. Такие материалы "
                    "выдумываются намеренно и не являются фактами."
                ),
                source_title=url_info.host,
                source_url=url,
                trusted=False,
                limitations=["Сайт относится к сатирическим изданиям."],
            )]
            satire_note = "Сатирическое издание"
        elif url_info.reputation == "untrusted":
            findings = [ClaimFinding(
                text=text[:300],
                status="contradicted",
                explanation=(
                    "Источник известен публикацией выдумок и не может "
                    "считаться доказательством."
                ),
                source_title=url_info.host,
                source_url=url,
                trusted=False,
                limitations=["Домен в списке недостоверных."],
            )]
            satire_note = "Недостоверный источник"

    # Строгий режим не применяется к прогнозам и безымянным утверждениям:
    # «завтра ожидается снег» — не фейк, даже если подтверждения нет.
    #
    # Прогноз определяется по проверяемым утверждениям, а не по всему
    # тексту: в статье на тысячу слов почти всегда найдётся «планируется»
    # или «вырастет», и тогда прошлый факт («ставку повысили») объявлялся
    # бы прогнозом, а «фейк» тихо снимался бы. Для короткого сообщения
    # утверждения и есть текст, поэтому поведение не меняется.
    claims_text = " ".join(c.text for c in findings) if findings else text
    is_future = _is_future(claims_text)
    strict_allowed = not is_future and not _is_vague(claims_text)

    # И к присланной ссылке. Материал не улика против себя: свежая новость
    # может ещё не быть перепечатана, а отсутствие подтверждений — это
    # «не проверили», а не «выдумали». Сатира и домены из чёрного списка
    # обрабатываются выше отдельно и по-прежнему дают «фейк».
    if url:
        strict_allowed = False

    # Явные правила вывода: признаки жанра вирусной выдумки, опора на
    # цитаты, готовое знание. Работают поверх находок и не зависят от
    # модели, поэтому вердикт воспроизводим.
    rule_result = verdict_rules.rules_for_text(
        claims=findings,
        sources=len(sources),
        trusted=len(evidence_sources),
        # База фактов уже ответила? Тогда её слово перевешивает правило
        # «прогноз нельзя объявить ложью»: «пенсионный возраст повысят
        # до 70» опровергается действующим законом.
        kb_hit=any(f.source_url and is_trusted(f.source_url) for f in findings),
        bank_hit=False,
        is_future=is_future,
        is_vague=_is_vague(claims_text),
        text=text,
    )

    level, verdict, score, uncertainty = _level_for(
        findings,
        searched=bool(evidence_sources),
        strict_allowed=strict_allowed,
        sources=len(sources),
        trusted=len(evidence_sources),
        tag=tag_match.tag,
        tag_confident=tag_match.is_confident,
        rule_result=rule_result,
    )
    limitations = [lim for f in findings for lim in f.limitations]
    # Понятная причина «непонятно»: пользователь должен знать, почему
    # система не дала ответ, а не гадать, сломалась она или нет.
    if verdict == "непонятно":
        if not sources:
            verdict_reason = (
                "Недостаточно подтверждений: в доступных источниках "
                "ничего не нашлось. Это не значит, что новость ложная."
            )
        elif not evidence_sources:
            verdict_reason = (
                "Все найденные источники недостоверны: подтверждать нечем."
            )
        else:
            verdict_reason = (
                "Недостаточно подтверждений: источники тему знают, "
                "но самого события в них нет."
            )
        limitations.append(verdict_reason)

    if satire_sources and not satire_note:
        hosts = ", ".join(sorted({s.domain or s.title for s in satire_sources})[:3])
        limitations.append(
            f"Среди найденного есть недостоверные сайты ({hosts}) — "
            "они исключены из доказательств."
        )
    # Причина «фейка» без источников — это признак жанра, а не поиск:
    # формулировка должна совпадать с вердиктом, иначе человек читает
    # «скорее фейк» и тут же «это ограничение поиска, а не вывод».
    if not evidence_sources and sources:
        limitations.append(
            "Все найденные источники недостоверны: подтверждать нечем."
        )
    elif not sources and verdict == "фейк":
        limitations.append(
            "Источников по теме не нашлось. Вердикт поставлен по признакам "
            "жанра вирусной выдумки, а не по опровержению источника."
        )
    elif not sources:
        limitations.append(
            "Границы проверки: источники не найдены, сверять утверждение не с чем. "
            "Это ограничение поиска, а не вывод о ложности."
        )
    if model_silent:
        limitations.append(
            "Модель не ответила — вердикт по детерминированной проверке."
        )

    reasoning = judge(
        claims=findings,
        sources=len(sources),
        trusted=len(evidence_sources),
        tag=tag_match.tag,
        tag_confident=tag_match.is_confident,
        kb_hit=False,
        bank_hit=False,
        strict_allowed=strict_allowed,
        # Рубрика должна знать итог правил: иначе в одном ответе «скорее
        # фейк» и «источников нет, это не вывод о ложности» — два разных
        # вывода. Правило с весомым основанием перебивает рубрику.
        rule_result=rule_result,
        # «Фейк» без единого опровержения и не по правилу — это строгий
        # режим. Если вердикт дало правило (признаки жанра), объяснение
        # берётся из правила: там причина точнее.
        strict_fake=(
            verdict == "фейк"
            and getattr(rule_result, "verdict", "") != "фейк"
            and not any(f.status == "contradicted" for f in findings)
        ),
    )

    outcome = CheckOutcome(
        claims=findings,
        verdict=verdict,
        level=level,
        score=score,
        uncertainty=uncertainty,
        mode=mode,
        sources_used=len(sources),
        limitations=limitations,
        tag=tag_match.as_dict(),
        reasoning=reasoning.as_dict(),
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    # Проверенный факт уходит в банк знаний: в следующий раз ответ будет
    # быстрее и стабильнее, а модель учится на собственных подтверждениях.
    for finding in findings:
        _remember(bank, finding, tag_match)
    _cache_put(text, outcome, tier.key, url)
    return outcome
