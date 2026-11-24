"""Числа в сообщении против базы фактов: главное число решает.

«Повысил на 200 б.п., до 21,00 %» — это тот же факт, что в базе, только
с пояснением. Раньше лишнее число считалось расхождением, и верное
сообщение получало «опровергнуто». Теперь опровергаем только тогда,
когда главного числа записи в тексте нет: тогда сообщение называет
другое значение, и это настоящая ошибка.
"""

from __future__ import annotations

from lakmus.signals import kb_facts


def kb() -> dict:
    return kb_facts.load_kb()


def check(text: str):
    topic = kb_facts.find_topic(kb(), text)
    assert topic is not None, f"тема не найдена для: {text}"
    return kb_facts.check_against_topic(topic, text)


def test_main_number_with_explanation_is_confirmed():
    verdict, _, explanation, details = check(
        "Совет директоров Банка России 25 октября 2024 года принял решение "
        "повысить ключевую ставку на 200 б.п., до 21,00% годовых."
    )
    assert verdict == kb_facts.Verdict.supports_real, explanation
    assert details["reason"] == "extra_numbers"
    assert "не сверялись" in details["limitation"]


def test_wrong_main_number_is_refuted():
    verdict, _, explanation, _ = check(
        "Банк России 25 октября 2024 года повысил ключевую ставку до 15% годовых."
    )
    assert verdict == kb_facts.Verdict.supports_fake, explanation
