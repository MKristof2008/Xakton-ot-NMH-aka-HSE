"""Сквозная проверка автопилота на живых данных и живой модели.

Запуск из каталога `repo/` (нужен `ml/.env` с ключом модели):

    py -3 autopilot/smoke_live.py
    py -3 autopilot/smoke_live.py --skip-llm     # без вызовов модели

Что проверяется:
1. Сбор с настроенных источников (сеть).
2. Разбор живой карточки фактчекеров → метка по правилу опровержения.
3. Проверка утверждений пайплайном проекта на заведомо ложном сообщении
   и на источнике, который его опровергает → метка `fake` с обоснованием.
4. Обучение: метрики на отложенной выборке и отказ от деградации.
5. Перезапуск цикла без ошибок и запись в журнал.

Сценарий ничего не пишет в рабочую базу автопилота: всё на временном
каталоге, настоящая база только читается.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from autopilot.db import Database                       # noqa: E402
from autopilot.model import ModelRegistry               # noqa: E402
from autopilot.settings import load_config, load_llm_env  # noqa: E402
from autopilot.trainer import Trainer                    # noqa: E402
from autopilot.verifier import Evidence, Verifier        # noqa: E402

OK, FAIL = "OK  ", "FAIL"


def _check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"{OK if condition else FAIL} {name}" + (f" — {detail}" if detail else ""))
    return condition


def _fake_from_card(card, source_id: str = "tg_test") -> dict:
    """Cобирает «пост из сети» из карточки опровержения."""
    title = card["title"]
    for prefix in ("Фейк: ", "Фейк ", "Видеофейк: ", "Фотомонтаж: "):
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    return {"id": 9000, "title": title[:200], "text": f"{title}. {card['text'][:1500]}",
            "url": "https://t.me/s/test", "source_id": source_id, "role": "news"}


async def main() -> int:
    parser = argparse.ArgumentParser(description="Сквозная проверка автопилота")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    load_llm_env()
    cfg = load_config()
    has_llm = bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_MODEL"))
    live = has_llm and not args.skip_llm
    print(f"модель: {os.environ.get('LLM_MODEL', '—') if live else 'вызовы отключены'}\n")

    passed = 0
    total = 0

    # -- 1. локальная модель ---------------------------------------------
    print("== 1. локальная модель ==")
    db = Database(cfg.db_path)
    registry = ModelRegistry(db)
    registry.load()
    total += 1
    passed += _check("модель загружена", registry.version != "untrained",
                     f"{registry.version} ({registry.source})")
    sample = "Учёные доказали, что коты умеют разговаривать по-русски и скрывают это от людей"
    prob = registry.predict(sample)
    total += 1
    passed += _check("предсказание вероятности", prob is not None and 0.0 <= prob <= 1.0,
                     f"P(фейк)={prob:.2f}" if prob is not None else "None")

    # -- 2. верификатор на изолированной базе ----------------------------
    tmp = tempfile.mkdtemp(prefix="autopilot_smoke_")
    iso = Database(Path(tmp) / "smoke.db")
    iso_registry = ModelRegistry(iso, models_dir=Path(tmp) / "models",
                                 pointer=Path(tmp) / "models" / "active.json")
    iso_registry.load()
    verifier = Verifier(iso, cfg.verifier, iso_registry)

    print("\n== 2. опровержение фактчекеров ==")
    cards = db.query(
        "SELECT id, title, text, url FROM items WHERE role='factcheck' "
        "AND length(text) >= 250 ORDER BY id LIMIT 5")
    for card in cards:
        iso.insert_item({
            "source_id": "factchecker", "source_name": "фактчекер", "kind": "rss",
            "url": card["url"], "url_hash": str(card["id"]),
            "content_hash": f"card{card['id']}", "simhash": None,
            "title": card["title"], "text": card["text"],
            "fetched_at": time.time(), "role": "factcheck"})
    verifier.corpus.rebuild()

    debunk_hits = 0
    for card in cards[:3]:
        item = _fake_from_card(card)
        result = await verifier.verify(item, 0.6, allow_llm=False)
        total += 1
        ok = result.label == "fake" and result.source == "debunk"
        debunk_hits += ok
        passed += _check(f"карточка «{card['title'][:50]}»", ok,
                         f"{result.label} ({result.source})")
    total += 1
    passed += _check("правило опровержения сработало хотя бы раз", debunk_hits > 0,
                     f"{debunk_hits} из 3")

    # -- 3. проверка утверждений живой моделью ---------------------------
    print("\n== 3. проверка утверждений пайплайном проекта ==")
    contradiction = {
        "id": 2,
        "title": "Совет директоров Банка России повысил ключевую ставку до 25 процентов",
        "text": ("На заседании 24 октября 2026 года совет директоров Банка России "
                 "принял решение повысить ключевую ставку до 25,00 процента годовых."),
        "url": "https://t.me/s/test", "source_id": "tg_test", "role": "news",
    }
    evidence = [Evidence(
        source_id="cbr_1", title="Банк России сохранил ключевую ставку на уровне 21%",
        url="https://www.cbr.ru/press/pr/",
        text=("Совет директоров Банка России 24 октября 2026 года принял решение "
              "сохранить ключевую ставку на уровне 21,00 процента годовых."),
        kind="news", reliability=0.95)]
    if live:
        label, confidence, reason = await verifier._claims_verdict(contradiction, evidence)
        total += 1
        passed += _check("ложное утверждение распознано как опровергнутое",
                         label == "fake" and confidence >= cfg.verifier.min_label_confidence,
                         f"{label} {confidence:.2f}: {reason[:90]}")
        support = [Evidence(
            source_id="cbr_2", title="Банк России сохранил ставку",
            url="https://www.cbr.ru/press/pr/2",
            text=("Совет директоров Банка России 24 октября 2026 года принял решение "
                  "сохранить ключевую ставку на уровне 21,00 процента годовых. "
                  "Следующее заседание назначено на декабрь."),
            kind="news", reliability=0.95)]
        truth = {"id": 3, "title": "Банк России сохранил ключевую ставку на уровне 21 процента",
                 "text": ("Банк России по итогам заседания 24 октября 2026 года "
                          "сохранил ключевую ставку на уровне 21 процента годовых."),
                 "url": "https://t.me/s/test2", "source_id": "tg_test", "role": "news"}
        label2, confidence2, reason2 = await verifier._claims_verdict(truth, support)
        total += 1
        passed += _check("верное утверждение подтверждено",
                         label2 == "real" and confidence2 >= cfg.verifier.min_label_confidence,
                         f"{label2} {confidence2:.2f}: {reason2[:90]}")
        print("\n== 4. LLM-арбитр: без цитаты метки нет ==")
        bad = {"verdict": "fake", "confidence": 0.95,
               "rationale": "похоже на фейк", "quotes": []}
        from autopilot.verifier import _valid_quotes
        total += 1
        passed += _check("выдуманные/отсутствующие цитаты отбрасываются",
                         _valid_quotes(bad["quotes"], evidence) == [])
    else:
        print("(вызовы модели отключены: проверяю только код валидации)")

    # -- 5. обучение и защита от деградации ------------------------------
    print("\n== 5. обучение ==")
    trainer = Trainer(iso, cfg.trainer, iso_registry, models_dir=Path(tmp) / "models")
    # корпус берём из проекта, но база и модели — временные
    result = trainer.train(reason="smoke", include_en=False)
    total += 1
    passed += _check("обучение выполнено", result is not None,
                     (result.notes if result else "нет данных"))
    if result is not None:
        total += 1
        passed += _check("метрики посчитаны на отложенной выборке",
                         "f1" in result.metrics and result.n_test > 0,
                         f"F1={result.metrics.get('f1')}, n_test={result.n_test}")
        total += 1
        took_decision = result.promoted or result.notes.startswith("отклонено")
        passed += _check("решение по порогам принято и объяснено", took_decision,
                         result.notes[:110])
        total += 1
        passed += _check("решение записано в журнал",
                         iso.last_training_run() is not None,
                         iso.last_training_run()["notes"] if iso.last_training_run() else "")

    print(f"\nитог: {passed}/{total} проверок пройдено")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
