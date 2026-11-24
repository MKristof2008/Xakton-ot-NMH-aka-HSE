"""Как система приходит к вердикту: рубрика вместо «50 из 100».

Раньше в ответе стояло число, взятое из таблицы уровней. Человек видел
«50 / 100» и не понимал, откуда оно. Теперь решение собирается по рубрике —
набору критериев с прозрачными весами, как в экзаменационной работе:

  1. Основание: есть ли дословная цитата из источника.
  2. Согласие источников: все говорят одно, расходятся или молчат.
  3. Надёжность: официальный домен, крупное медиа, неизвестный блог.
  4. Полнота: сколько утверждений удалось проверить из всех.
  5. Противоречие: не спорят ли утверждения между собой.
  6. Контекст: не устарел ли факт и не вырван ли из контекста.

Каждый критерий даёт баллы и обязательное словесное обоснование. Их сумма
и есть итоговая уверенность — не «на глаз», а по правилам, которые можно
проверить, оспорить и улучшить.

Вердикт тоже выводится из рубрики, а не задаётся таблицей: «фейк» требует
прямого опровержения, «правда» — дословного подтверждения, всё остальное —
«непонятно». Поэтому цифры больше не появляются «на всякий случай».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class Criterion:
    """Один критерий рубрики: сколько дал и почему."""

    key: str
    title: str
    points: int
    max_points: int
    reason: str
    #: ok — всё хорошо; warn — данных мало; bad — против вердикта.
    tone: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "points": self.points,
            "max_points": self.max_points,
            "reason": self.reason,
            "tone": self.tone,
        }


@dataclass
class VerdictReasoning:
    """Итог рубрики: баллы, вердикт и словесная цепочка."""

    confidence: int = 0
    verdict: str = "непонятно"
    criteria: list[Criterion] = field(default_factory=list)
    #: Одна фраза, которую человек прочитает первой.
    summary: str = ""
    #: Тег темы, к которой отнесено утверждение.
    tag_key: str = "unknown"
    tag_title: str = "Тема не определена"
    rules_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "verdict": self.verdict,
            "summary": self.summary,
            "tag": {"key": self.tag_key, "title": self.tag_title},
            "rules_applied": self.rules_applied,
            "criteria": [c.as_dict() for c in self.criteria],
        }


#: Правила темы → как их объяснить человеку.
RULE_TEXT = {
    "official_only": "тема проверяется по официальному источнику",
    "numbers_must_match": "числа и даты должны совпасть точно",
    "future_never_fake": "прогноз нельзя объявить ложью",
    "count_from_official": "цифры по происшествию — только от ведомства",
    "no_miracle_cure": "обещания «лечит всё» не принимаются",
    "no_sources_no_fake": "без источников «фейк» не ставится",
    "weak_evidence": "тема со слабыми источниками: нужно два независимых",
}


#: Порог, с которого правило выносит вердикт само (совпадает с
#: ``full_pipeline``: столько даёт абсурдная мера или два независимых
#: маркера, но не один стилистический признак).
RULE_DECISION = 55


def _count_claim_status(claims: Sequence[Any]) -> tuple[int, int, int, int]:
    supported = sum(1 for c in claims if getattr(c, "status", "") == "supported")
    contradicted = sum(
        1 for c in claims if getattr(c, "status", "") == "contradicted"
    )
    unknown = sum(1 for c in claims if getattr(c, "status", "") == "insufficient")
    quotes = sum(1 for c in claims if getattr(c, "quote", ""))
    return supported, contradicted, unknown, quotes


def _tag_rules(tag: Any) -> list[str]:
    rules = getattr(tag, "rules", ()) or ()
    return [r for r in rules if r in RULE_TEXT]


def judge(
    *,
    claims: Sequence[Any],
    sources: int,
    trusted: int,
    tag: Any = None,
    tag_confident: bool = False,
    kb_hit: bool = False,
    bank_hit: bool = False,
    strict_allowed: bool = True,
    rule_result: Any = None,
    strict_fake: bool = False,
) -> VerdictReasoning:
    """Собрать вердикт по рубрике. Без модели, только из посчитанного.

    Возвращает баллы с обоснованием. Никаких «50 по умолчанию»: если
    доказательств нет, балл низкий и это прямо написано.

    ``rule_result`` — итог явных правил (``verdict_rules``). Рубрика видит
    только статусы утверждений и не знает о признаках жанра вирусной
    выдумки. Без этой связи в одном ответе стояли бы «скорее фейк» и
    «это не значит, что новость ложная» — то есть два противоположных
    вывода. Правило с весомым основанием перебивает рубрику, а правило
    «непонятно» снимает обвинение, которого рубрика не видит.

    ``strict_fake`` — вердикт «фейк» поставлен строгим режимом: поиск был,
    но подтверждений не нашлось. Рубрика в этом случае не спорит и не
    пишет «признаков недостаточно»: она называет настоящую причину.
    """
    tag_key = getattr(tag, "key", "unknown") if tag is not None else "unknown"
    tag_title = (
        getattr(tag, "title", "Тема не определена")
        if tag is not None
        else "Тема не определена"
    )
    out = VerdictReasoning(tag_key=tag_key or "unknown", tag_title=tag_title)
    out.rules_applied = [RULE_TEXT[r] for r in _tag_rules(tag)]

    supported, contradicted, unknown, quotes = _count_claim_status(claims)
    total = len(claims)

    # 1. Основание: дословная цитата.
    if quotes:
        out.criteria.append(Criterion(
            "quote", "Основание",
            30 if quotes >= 2 else 25, 30,
            f"Дословных цитат проверено кодом: {quotes}.",
        ))
    elif sources:
        out.criteria.append(Criterion(
            "quote", "Основание", 5, 30,
            "Источники есть, но дословной цитаты в них не нашлось — довод слабый.",
            "warn",
        ))
    else:
        out.criteria.append(Criterion(
            "quote", "Основание", 0, 30,
            "Источников нет, подтверждать нечем.",
            "warn",
        ))

    # 2. Согласие источников.
    if supported and contradicted:
        out.criteria.append(Criterion(
            "agreement", "Согласие источников", 0, 20,
            "Утверждения спорят между собой: часть подтверждается, часть опровергается.",
            "bad",
        ))
    elif supported or contradicted:
        direction = "подтверждают" if supported else "опровергают"
        out.criteria.append(Criterion(
            "agreement", "Согласие источников", 20, 20,
            f"Все проверенные утверждения {direction} сообщение.",
        ))
    elif unknown:
        out.criteria.append(Criterion(
            "agreement", "Согласие источников", 0, 20,
            "Ни один источник событие не подтвердил и не опроверг.",
            "warn",
        ))
    else:
        out.criteria.append(Criterion(
            "agreement", "Согласие источников", 0, 20,
            "Проверять было нечего.",
            "warn",
        ))

    # 3. Надёжность доменов.
    if trusted:
        out.criteria.append(Criterion(
            "trust", "Надёжность источников", 15, 15,
            f"Доверенных источников: {trusted} из {sources}.",
        ))
    elif sources:
        out.criteria.append(Criterion(
            "trust", "Надёжность источников", 3, 15,
            "Источники есть, но ни один не входит в список доверенных.",
            "warn",
        ))
    else:
        out.criteria.append(Criterion(
            "trust", "Надёжность источников", 0, 15,
            "Проверять нечего: источников ноль.",
            "warn",
        ))

    # 4. Полнота проверки.
    if total and (supported or contradicted):
        share = (supported + contradicted) / total
        points = round(15 * share)
        detail = (
            f"Проверено {supported + contradicted} из {total} утверждений."
        )
        out.criteria.append(Criterion("coverage", "Полнота", points, 15, detail))
    else:
        out.criteria.append(Criterion(
            "coverage", "Полнота", 0, 15,
            f"Удалось проверить 0 из {total or 0} утверждений.",
            "warn",
        ))

    # 5. Противоречие внутри сообщения.
    if supported and contradicted:
        out.criteria.append(Criterion(
            "conflict", "Противоречие", 0, 10,
            "Внутри сообщения есть взаимоисключающие утверждения.",
            "bad",
        ))
    elif total:
        out.criteria.append(Criterion(
            "conflict", "Противоречие", 10, 10,
            "Взаимоисключающих утверждений не найдено.",
        ))
    else:
        out.criteria.append(Criterion(
            "conflict", "Противоречие", 0, 10,
            "Сравнивать нечего.",
            "warn",
        ))

    # 6. Готовое знание: банк или курируемая база.
    if bank_hit or kb_hit:
        what = "банк знаний" if bank_hit else "курируемая база"
        out.criteria.append(Criterion(
            "known", "Готовое знание", 10, 10,
            f"Утверждение уже проверено: {what}.",
        ))
    else:
        out.criteria.append(Criterion(
            "known", "Готовое знание", 0, 10,
            "Раньше не проверялось — искали заново.",
            "warn",
        ))

    points = sum(c.points for c in out.criteria)
    max_points = sum(c.max_points for c in out.criteria)
    confidence = round(100 * points / max_points) if max_points else 0

    # Тема не распознана — уверенность падает: правила проверки неизвестны.
    if tag is not None and not tag_confident:
        confidence = round(confidence * 0.85)

    # Прогноз: даже при полном согласии «правда» не ставится.
    is_future = "не наступило" in " ".join(out.rules_applied)

    # Вердикт выводится из рубрики, а не из таблицы.
    if contradicted and not supported:
        verdict = "фейк"
    elif supported and not contradicted:
        verdict = "правда" if not is_future else "непонятно"
    elif supported and contradicted:
        verdict = "непонятно"
    else:
        # Ни подтверждений, ни опровержений: «фейк» без основания нельзя.
        verdict = "непонятно"

    # Правила видят то, чего рубрика не видит: жанр вирусной выдумки,
    # стиль текста, запрет «фейка» для прогноза. Итог обязан совпадать
    # с вердиктом, который увидит человек: иначе в одном ответе стоят
    # «скорее фейк» и «это не значит, что новость ложная».
    rule_verdict = str(getattr(rule_result, "verdict", "") or "") if rule_result else ""
    rule_confidence = int(getattr(rule_result, "confidence", 0) or 0) if rule_result else 0
    rule_summary = str(getattr(rule_result, "summary", "") or "") if rule_result else ""
    if rule_verdict in ("фейк", "правда") and rule_confidence >= RULE_DECISION:
        verdict = rule_verdict
        confidence = max(confidence, rule_confidence)
    elif rule_verdict == "непонятно" and verdict == "фейк" and not strict_fake:
        # Правило сняло обвинение: прогноз, безымянное утверждение,
        # одиночный стилистический признак.
        verdict = "непонятно"

    # Строгий режим: поиск был, подтверждений нет — вердикт «фейк» принят
    # уровнем доверия, а рубрика сама по себе дала бы «непонятно». Итог
    # и объяснение обязаны совпадать с тем, что показано человеку.
    if strict_fake:
        verdict = "фейк"

    out.verdict = verdict
    out.confidence = max(0, min(100, confidence))

    if strict_fake:
        # Строгий режим: поиск был, подтверждений нет. Это и есть причина,
        # и она не должна читаться как «признаков недостаточно».
        out.summary = (
            "Фейк: источники по теме найдены, но ни один не подтверждает "
            "сообщение. Строгий режим трактует это как выдумку."
        )
    elif rule_summary and rule_verdict == verdict:
        # Объяснение правила точнее рубрики: оно называет причину словами.
        out.summary = rule_summary
    elif verdict == "фейк":
        out.summary = (
            f"Фейк: {contradicted} утверждений опровергнуты, "
            f"подтверждений нет. Уверенность {out.confidence}%."
        )
    elif verdict == "правда":
        out.summary = (
            f"Правда: подтверждено утверждений — {supported}. "
            f"Уверенность {out.confidence}%."
        )
    elif supported and contradicted:
        out.summary = (
            "Непонятно: источники противоречат друг другу — "
            f"подтверждено {supported}, опровергнуто {contradicted}."
        )
    elif rule_verdict == "непонятно" and rule_summary:
        # Правило объяснило причину лучше рубрики: например, «признаки
        # жанра вирусной выдумки есть, но одного стиля мало для приговора».
        out.summary = rule_summary
    elif not sources:
        out.summary = (
            "Непонятно: в доступных источниках ничего не нашлось. "
            "Это не значит, что новость ложная."
        )
    else:
        out.summary = (
            "Непонятно: источники тему знают, но самого события в них нет."
        )
    return out


def from_outcome(outcome: Any, tag_match: Any = None) -> VerdictReasoning:
    """Пересчитать рубрику для готового результата проверки."""
    tag = getattr(tag_match, "tag", None) if tag_match is not None else None
    confident = bool(getattr(tag_match, "is_confident", False))
    return judge(
        claims=getattr(outcome, "claims", []) or [],
        sources=getattr(outcome, "sources_used", 0) or 0,
        trusted=sum(
            1 for c in (getattr(outcome, "claims", []) or []) if getattr(c, "trusted", False)
        ),
        tag=tag,
        tag_confident=confident,
    )
