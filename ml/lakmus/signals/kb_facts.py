"""Проверка утверждений по курируемой базе фактов — без модели.

Сигнал T2 в офлайн-режиме: модель недоступна, поэтому утверждения ищутся
по ключевым темам в ``data/kb_facts.json``, а числа и даты сверяются
детерминированно. Это честная проверка на ограниченной базе, а не
имитация полноценного поиска: чего нет в базе — помечается как
непроверенное, а не додумывается.

Вердикты:
  supports_real   — утверждение совпало с записью базы;
  supports_fake   — база прямо опровергает (запись с verdict=false_claim);
  neutral         — тема найдена, но подтвердить или опровергнуть нельзя;
  unavailable     — база не загружена или тема не найдена.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Optional

from ..context import CheckContext
from ..schemas import InputType, Stage, Verdict
from .base import BaseSignal, make_result

KB_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "kb_facts.json"

RELIABILITY_MATCH = 0.9      # совпадение с курируемой записью и её первоисточником
RELIABILITY_REFUTE = 0.9     # прямое опровержение курируемой записью
RELIABILITY_TOPIC = 0.35     # тема найдена, но без сверки

#: Допуск при сравнении чисел: «21%» и «21,00%» — одно и то же.
NUMBER_TOLERANCE = 0.005


class KbError(RuntimeError):
    """База не читается — сигнал вернёт unavailable."""


#: Кэш разобранной базы: (mtime, размер) -> данные.
#: База весит 480 КБ и парсится заново на каждый вызов, а вызывается она
#: 3-4 раза за проверку. Читаем один раз и перечитываем только если файл
#: изменился — правка базы на диске по-прежнему подхватывается.
_KB_CACHE: dict[tuple[float, int], dict[str, Any]] = {}


def load_kb(path: Any = None) -> dict[str, Any]:
    """Читает базу фактов. Бросает KbError с понятной причиной."""
    target = pathlib.Path(path) if path else KB_PATH
    if not target.is_file():
        raise KbError(f"база фактов не найдена: {target}")
    try:
        stat = target.stat()
        key = (stat.st_mtime, stat.st_size)
        cached = _KB_CACHE.get(key)
        if cached is not None:
            return cached
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KbError(f"база фактов не читается: {type(exc).__name__}") from exc
    # Держим только актуальную версию: база одна.
    _KB_CACHE.clear()
    _KB_CACHE[key] = data
    return data


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower().replace("ё", "е")


def _numbers(text: str) -> set[float]:
    """Все числа из текста: «21%», «21,00», «5 000» → 21.0, 21.0, 5000.0."""
    out: set[float] = set()
    for raw in re.findall(r"\d[\d\s]*(?:[.,]\d+)?", text or ""):
        cleaned = raw.replace(" ", "").replace(",", ".")
        try:
            out.add(round(float(cleaned), 3))
        except ValueError:
            continue
    return out


def _numbers_agree(a: set[float], b: set[float]) -> Optional[bool]:
    """True — все числа утверждения есть в записи, False — хоть одно расходится.

    Сравниваем как подмножество, а не «любая пара совпала»: иначе «ставка
    15%» проходит проверку из-за совпавшего числа «25» (день месяца).
    """
    if not a or not b:
        return None
    for x in a:
        if not any(y and abs(x - y) / max(abs(y), 1e-9) <= NUMBER_TOLERANCE for y in b):
            return False
    return True


def _content_numbers(text: str) -> set[float]:
    """Числа, которые несут смысл: годы и дни месяца не в счёт.

    Год проверяется отдельно (_years), а день месяца совпадает почти
    всегда и только маскирует расхождение в главной цифре.
    """
    out: set[float] = set()
    for value in _numbers(text):
        if 1900 <= value <= 2100:
            continue  # год — отдельная проверка
        if value == int(value) and 1 <= value <= 31 and "." not in str(value):
            continue  # похоже на день месяца
        out.add(value)
    return out


def _years(text: str) -> set[int]:
    return {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")}


def find_topic(kb: dict[str, Any], text: str) -> Optional[dict[str, Any]]:
    """Тема по ключевым словам. Берём тему с самым длинным совпавшим словом."""
    topics = find_topics(kb, text)
    return topics[0] if topics else None


#: Окончания, которые срезаем перед сравнением. «Мобилизацию» и
#: «мобилизация» — одно слово, но подстрока разная: без стема тема
#: не находилась и фейк уходил в «непонятно».
_RU_ENDINGS = (
    "иями", "ями", "ами", "ов", "ев", "ий", "ия", "ию", "ии", "ей", "ой",
    "ый", "ая", "ое", "ые", "ую", "юю", "ах", "ях", "ом", "ем", "ам", "ям",
    "а", "я", "ы", "и", "у", "ю", "е", "о", "ь",
)


#: Кэш основ: слов в базе немного и они повторяются из проверки в проверку.
#: Без него стем считался 2190 раз за один вызов (перебор 31 окончания дважды).
_STEM_CACHE: dict[str, str] = {}


def _stem(word: str) -> str:
    """Грубая основа слова: срезаем до 2 окончаний подряд.

    Настоящей морфологии здесь нет и не нужно: цель — чтобы «мобилизацию»
    и «мобилизация» дали одну основу. Слишком короткие слова не трогаем.
    """
    cached = _STEM_CACHE.get(word)
    if cached is not None:
        return cached
    w = _norm(word)
    if len(w) > 4:
        for _ in range(2):
            for end in _RU_ENDINGS:
                if len(w) - len(end) >= 4 and w.endswith(end):
                    w = w[: -len(end)]
                    break
    if len(_STEM_CACHE) < 20_000:  # страховка от роста без границ
        _STEM_CACHE[word] = w
    return w


_WORD_RE = re.compile(r"[а-яa-z0-9]+")


def _stems_in(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(_norm(text)) if len(w) >= 4}


def _words_in(text: str) -> set[str]:
    return set(_WORD_RE.findall(_norm(text)))


def _keyword_matches(keyword: str, text: str, text_stems: set[str], text_words: set[str]) -> bool:
    """Все значимые слова ключа есть в тексте.

    Короткие слова («цб», «мрот») сравниваем точно, длинные — по основе.
    Раньше ключ «ставка цб» сводился к одной основе «ставк», и фраза
    «ставка выросла» находила тему ЦБ, хотя ЦБ в ней нет.
    """
    words = _WORD_RE.findall(_norm(keyword))
    if not words:
        return False
    for w in words:
        if len(w) >= 4:
            if _stem(w) not in text_stems:
                return False
        elif w not in text_words:
            return False
    return True


def find_topics(kb: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Все темы, чьи ключевые слова встретились, от лучшей к худшей.

    Сообщение может задевать несколько тем («индексация пенсий и тарифы
    ЖКХ»), поэтому сигнал перебирает их, а не хватается за первую.
    Сравнение идёт по основам слов: «мобилизацию» находит тему
    «мобилизация», иначе половина падежных форм терялась.
    """
    low = _norm(text)
    text_stems = _stems_in(text)
    text_words = _words_in(text)
    scored: list[tuple[int, dict[str, Any]]] = []
    for topic in kb.get("topics", []):
        best = 0
        for keyword in topic.get("keywords", []):
            k = _norm(keyword)
            if not k:
                continue
            # Прямое вхождение — самый надёжный сигнал, он весит больше.
            if k in low:
                best = max(best, len(k) * 2)
            elif _keyword_matches(keyword, text, text_stems, text_words):
                best = max(best, len(k))
        if best:
            scored.append((best, topic))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [topic for _, topic in scored]


def _claim_matches_fact(claim: str, fact: dict[str, Any]) -> bool:
    """О том же ли факте речь.

    Тема может быть общей, а утверждения — разными: «индексация пенсий»
    и «пенсионный возраст 65 лет» лежат в одной теме, но это разные факты.
    У каждой записи базы есть ``match_keywords`` — свои слова-маркеры.
    Если ни одного нет, сверять числа нельзя: получим ложное опровержение.
    """
    low = _norm(claim)
    markers = [str(k) for k in fact.get("match_keywords", [])]
    if not markers:
        return True  # запись без маркеров считаем универсальной для темы
    return any(_norm(m) in low for m in markers)


def _asserts_fact(text: str, fact: dict[str, Any]) -> bool:
    """Сообщает ли текст тот же факт, а не просто упоминает тему.

    «МРОТ с 1 января 2025 года — 22 440 рублей» сообщает факт: слова
    совпадают с записью базы почти целиком. «Что там по МРОТ?» — только
    называет тему, подтверждать нечего.
    """
    low = _norm(text)
    # Вопрос — это не утверждение.
    if low.rstrip().endswith("?") or "что скажешь" in low or "что там" in low:
        return False
    fact_claim = _norm(str(fact.get("claim", "")))
    if not fact_claim:
        return False
    text_stems = _stems_in(text)
    fact_stems = _stems_in(fact_claim)
    if not fact_stems:
        return False
    # Текст должен называть большую часть слов факта. Упоминание одной темы
    # даёт 10-20 %, пересказ того же факта — 60 % и выше.
    overlap = len(text_stems & fact_stems) / len(fact_stems)
    return overlap >= 0.5


#: Призывы и риторика, где «не» не отрицает факт: «не молчите»,
#: «не проходите мимо», «не оставайтесь в стороне». Если такое есть —
#: текст считается утвердительным независимо от «не».
_APPEALS = (
    "не молчите", "не молчать", "не проходите", "не проходи",
    "не оставайтесь", "не оставайся", "не игнорируйте", "не игнорируй",
    "не упустите", "не упусти", "не забывайте", "не забывай",
    "не верьте", "не верь",
)

#: Отрицательные маркеры именно для факта, а не для призыва.
_NEG_MARKERS = (
    "не принимал", "не принимали", "не было", "не будет", "не планир",
    "не запрещ", "не отмен", "не ввод", "не подпис", "не объяв",
    "никогда не", "не существует", "не является", "нет такого",
)


def _is_negated(text: str) -> bool:
    """Отрицает ли текст сам факт.

    Осторожно с призывами: «Власти скрывают: запретят наличные, не молчите!»
    содержит «не», но факт утвердительный. Раньше это давало ложное
    совпадение с записью «запрета не было» — и фейк получал «правда, 95».
    Поэтому сначала отсекаем призывы, потом ищем маркеры отрицания факта.
    """
    low = _norm(text)
    if any(appeal in low for appeal in _APPEALS):
        return False
    return any(marker in low for marker in _NEG_MARKERS)


def check_against_topic(topic: dict[str, Any], text: str) -> tuple[Verdict, float, str, dict[str, Any]]:
    """Сверка текста с записями темы.

    Возвращает (вердикт, сила, объяснение, детали).
    """
    facts = topic.get("facts", [])
    refuted = [f for f in facts if f.get("verdict") == "false_claim"]
    confirmed = [f for f in facts if f.get("verdict") == "true"]

    # 0. Речь вообще о другом факте внутри той же темы — не сверяем.
    #    «Индексация пенсий с 1 июля» не должна опровергаться записью
    #    про пенсионный возраст только потому, что тема общая.
    relevant_refuted = [f for f in refuted if _claim_matches_fact(text, f)]
    relevant_confirmed = [f for f in confirmed if _claim_matches_fact(text, f)]

    # 0a. Смысл отрицания: «ЦБ не запрещал наличные» — это правда, хотя
    #     тема та же, что у фейка про запрет. Смотрим только те записи,
    #     которые прямо называют действие (запрет, отмена, штраф):
    #     сравнение знака по всему тексту даёт ложные срабатывания.
    negated = _is_negated(text)
    if negated:
        for fact in relevant_refuted:
            if _is_negated(str(fact.get("claim", ""))):
                return (
                    Verdict.supports_real,
                    RELIABILITY_MATCH,
                    f"Совпадает с базой: {fact['claim']}.",
                    {
                        "topic": topic["topic"],
                        "matched_fact": fact,
                        "reason": "matched_negation",
                        "source": fact.get("source"),
                        "url": fact.get("url"),
                    },
                )

    if facts and not relevant_refuted and not relevant_confirmed:
        return (
            Verdict.neutral,
            RELIABILITY_TOPIC,
            f"Тема найдена ({topic['topic']}), но утверждение не про тот факт, "
            f"что в базе — нужна ручная проверка.",
            {"topic": topic["topic"], "reason": "different_fact"},
        )
    refuted = relevant_refuted
    confirmed = relevant_confirmed

    # 1. Прямое опровержение: база говорит, что такого нет.
    if refuted:
        fact = refuted[0]
        return (
            Verdict.supports_fake,
            RELIABILITY_REFUTE,
            f"База фактов опровергает: {fact['claim']}.",
            {
                "topic": topic["topic"],
                "matched_fact": fact,
                "reason": "refuted",
                "source": fact.get("source"),
                "url": fact.get("url"),
            },
        )

    # 2. Подтверждение с сверкой чисел и годов.
    if confirmed:
        fact = confirmed[0]
        claim_numbers = _content_numbers(text)
        fact_numbers = _content_numbers(str(fact.get("value", "")) + " " + fact["claim"])
        agree = _numbers_agree(claim_numbers, fact_numbers)

        claim_years = _years(text)
        fact_years = _years(fact.get("date", "") + " " + fact["claim"])
        year_ok = not claim_years or not fact_years or bool(claim_years & fact_years)

        if agree is False:
            # Лишнее число в тексте — не всегда расхождение. «Повысил на
            # 200 б.п., до 21,00%» — это тот же факт с пояснением, а не
            # другая ставка. Если главное число записи в тексте есть,
            # подтверждаем, но честно помечаем, что остальные числа не
            # сверялись. Опровергаем только когда главного числа нет:
            # тогда сообщение называет другое значение.
            main = _content_numbers(str(fact.get("value", "")))
            main_agrees = bool(main) and bool(claim_numbers & main)
            if main_agrees:
                return (
                    Verdict.supports_real,
                    RELIABILITY_MATCH,
                    f"Главное число совпадает с базой: {fact['claim']}.",
                    {
                        "topic": topic["topic"],
                        "matched_fact": fact,
                        "reason": "extra_numbers",
                        "claim_numbers": sorted(claim_numbers),
                        "fact_numbers": sorted(fact_numbers),
                        "source": fact.get("source"),
                        "url": fact.get("url"),
                        "limitation": "Лишние числа в сообщении не сверялись.",
                    },
                )
            return (
                Verdict.supports_fake,
                RELIABILITY_REFUTE,
                f"Число в сообщении не совпадает с базой: {fact['claim']}.",
                {
                    "topic": topic["topic"],
                    "matched_fact": fact,
                    "reason": "number_mismatch",
                    "claim_numbers": sorted(claim_numbers),
                    "fact_numbers": sorted(fact_numbers),
                    "source": fact.get("source"),
                    "url": fact.get("url"),
                },
            )

        if agree and year_ok:
            return (
                Verdict.supports_real,
                RELIABILITY_MATCH,
                f"Совпадает с базой: {fact['claim']}.",
                {
                    "topic": topic["topic"],
                    "matched_fact": fact,
                    "reason": "matched",
                    "source": fact.get("source"),
                    "url": fact.get("url"),
                },
            )

        # Чисел в утверждении нет — сверять нечего. Раньше здесь стояло
        # «Совпадает с базой», и любое упоминание темы («Что там по МРОТ?»)
        # получало вердикт «правда». Теперь проверяем, что в тексте вообще
        # утверждается тот же факт, а не просто названа тема.
        if not claim_numbers and not claim_years:
            if _asserts_fact(text, fact):
                return (
                    Verdict.supports_real,
                    RELIABILITY_MATCH,
                    f"Совпадает с базой: {fact['claim']}.",
                    {
                        "topic": topic["topic"],
                        "matched_fact": fact,
                        "reason": "matched",
                        "source": fact.get("source"),
                        "url": fact.get("url"),
                    },
                )
            return (
                Verdict.neutral,
                RELIABILITY_TOPIC,
                f"Тема найдена ({topic['topic']}), но утверждения о факте нет: "
                f"{fact['claim']}.",
                {
                    "topic": topic["topic"],
                    "matched_fact": fact,
                    "reason": "topic_only",
                    "source": fact.get("source"),
                    "url": fact.get("url"),
                },
            )

        return (
            Verdict.neutral,
            RELIABILITY_TOPIC,
            f"Тема найдена ({topic['topic']}), но числа или даты не сверены: {fact['claim']}.",
            {
                "topic": topic["topic"],
                "matched_fact": fact,
                "reason": "partial",
                "source": fact.get("source"),
                "url": fact.get("url"),
            },
        )

    return (
        Verdict.neutral,
        RELIABILITY_TOPIC,
        f"Тема найдена ({topic['topic']}), записей для сверки нет.",
        {"topic": topic["topic"], "reason": "topic_only"},
    )


class KbFactsSignal(BaseSignal):
    """Сверка утверждений с курируемой базой фактов (офлайн, без модели)."""

    id = "kb_facts"
    name = "Сверка с базой фактов"
    stage = Stage.sources
    timeout_s = 5.0

    async def run(self, ctx: CheckContext) -> Any:
        if ctx.input.type == InputType.image:
            return self.unavailable("сигнал работает с текстом, а не с изображением")

        text = (ctx.text or ctx.input.extracted_text or "").strip()
        if len(text) < 12:
            return self.unavailable("текст слишком короткий для сверки")

        try:
            kb = load_kb(ctx.options.get("kb_path"))
        except KbError as exc:
            return self.unavailable(str(exc))

        topic = find_topic(kb, text)
        if topic is None:
            return make_result(
                self, Verdict.unavailable, 0.0, 0.0,
                "Темы нет в базе фактов — утверждение не проверено. "
                "Это не значит, что оно ложное.",
                {"reason": "topic_not_found", "topics_in_kb": len(kb.get("topics", []))},
            )

        # Перебираем все задевающие темы: первая может не содержать нужного
        # факта («пенсии» без «индексации»), а вторая — содержать.
        verdict, reliability, summary, details = check_against_topic(topic, text)
        if details.get("reason") == "different_fact":
            for other in find_topics(kb, text)[1:]:
                alt = check_against_topic(other, text)
                if alt[3].get("reason") != "different_fact":
                    verdict, reliability, summary, details = alt
                    break

        strength = reliability  # сила равна надёжности: совпадение — единственный довод
        return make_result(self, verdict, strength, reliability, summary, details)
