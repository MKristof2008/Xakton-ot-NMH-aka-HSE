"""Полный пример вызова пайплайна на fake-клиенте (без API-ключа).

Запуск: py -3 example.py
Для живой модели замените FakeLLMClient на LLMClient
(переменные окружения LLM_API_KEY, LLM_BASE_URL, LLM_MODEL).
"""
import asyncio
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from doreposta import (
    ContentKind, FakeLLMClient, Source, compose_reply, evaluate_claims,
    extract_claims,
)

MESSAGE = "Всем студентам повысят стипендию с 1 сентября! Срочно репост!"

# Ответы fake-модели: 1-й вызов — извлечение, 2-й — оценка
FAKE_EXTRACT = json.dumps({"claims": [{
    "id": "c1",
    "original_span": "Всем студентам повысят стипендию с 1 сентября",
    "text": "Всем студентам повысят стипендию с 1 сентября",
    "queries": ["повышение стипендии сентябрь"],
    "missing_context": ["год не указан"]}],
    "uncheckable_reason": None}, ensure_ascii=False)

FAKE_EVALUATE = json.dumps({"claims": [{
    "id": "c1",
    "text": "Всем студентам повысят стипендию с 1 сентября",
    "status": "context",
    "explanation": "Повышение реально, но только для второго курса",
    "evidence": [{"source_id": "s1",
                  "quote": "стипендию повысят студентам второго курса",
                  "relation": "adds_context",
                  "explanation": "Утверждение говорит «всем», источник — только второй курс"}],
    "limitations": ["год повышения не указан в материале"],
    "next_action": "Проверить приказ на сайте учебного заведения"}]},
    ensure_ascii=False)

SOURCES = [Source(
    id="s1", title="Новости учебного заведения", url="https://example.org/news/123",
    content="С сентября стипендию повысят студентам второго курса очной формы.",
    content_kind=ContentKind.page,
)]


async def main():
    client = FakeLLMClient([FAKE_EXTRACT, FAKE_EVALUATE])

    extraction = await extract_claims(MESSAGE, {"message_date": None}, client)
    print("УТВЕРЖДЕНИЯ:", extraction.model_dump_json(indent=2))

    evaluation = await evaluate_claims(extraction.claims, SOURCES, {}, client)
    print("\nОЦЕНКА:", evaluation.model_dump_json(indent=2))

    print("\nДЛЯ ПЕРЕСЫЛКИ В ЧАТ:\n" + compose_reply(evaluation, SOURCES))


if __name__ == "__main__":
    asyncio.run(main())
