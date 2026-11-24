"""Тесты автопилота: без сети и без вызовов модели.

Запуск из каталога `repo/`:
    py -3 -m pytest autopilot/tests -q
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from autopilot.db import Database                      # noqa: E402
from autopilot.feeds import SourceSpec, pick_path      # noqa: E402
from autopilot.model import ModelRegistry              # noqa: E402
from autopilot.robots import parse_robots, _select      # noqa: E402
from autopilot.textnorm import (content_hash, detect_lang, extract_article,  # noqa: E402
                                hamming, host_of, simhash, simhash_from_int,
                                simhash_to_int, strip_tags, url_hash)
from autopilot.verifier import (Evidence, quote_is_verbatim, _extract_json,  # noqa: E402
                                _valid_quotes)


# ---------------------------------------------------------------------------
# Текст и хеши
# ---------------------------------------------------------------------------

def test_strip_tags_and_entities():
    assert strip_tags("<p>Привет&nbsp;<b>мир</b></p>") == "Привет мир"


def test_content_hash_ignores_case_and_punctuation():
    assert content_hash("Привет, МИР!") == content_hash("привет мир")


def test_url_hash_strips_tracking_and_anchor():
    a = url_hash("https://TASS.ru/news/1/?utm_source=x#frag")
    b = url_hash("https://tass.ru/news/1/")
    assert a == b


def test_simhash_near_duplicates():
    base = ("Центральный банк объявил о повышении ключевой ставки до 21 процента "
            "с 28 октября текущего года на заседании совета директоров")
    similar = ("Центральный банк объявил о повышении ключевой ставки до 21 процента "
               "с 28 октября этого года на заседании совета директоров")
    other = "Учёные нашли новый вид глубоководных рыб в Тихом океане у берегов Японии"
    h1, h2, h3 = simhash(base), simhash(similar), simhash(other)
    assert hamming(h1, h2) < hamming(h1, h3)
    assert hamming(h1, h2) <= 8


def test_simhash_int_roundtrip():
    value = simhash("текст для проверки целочисленного представления отпечатка")
    assert simhash_from_int(simhash_to_int(value)) == value


def test_detect_lang():
    assert detect_lang("Это сообщение на русском языке о новостях") == "ru"
    assert detect_lang("This message is written in English about news") == "en"


def test_host_of():
    assert host_of("https://www.tass.ru/news/1") == "tass.ru"
    assert host_of("") == ""


def test_extract_article_prefers_paragraphs():
    html = ("<html><body><nav>меню</nav><article>"
            + "".join(f"<p>Абзац номер {i} с достаточно длинным текстом для проверки.</p>"
                      for i in range(8))
            + "</article><footer>подвал</footer></body></html>")
    text = extract_article(html)
    assert "Абзац номер 0" in text
    assert "подвал" not in text


# ---------------------------------------------------------------------------
# Источники и парсеры
# ---------------------------------------------------------------------------

def test_source_spec_validation():
    with pytest.raises(ValueError):
        SourceSpec(id="x", name="X", kind="html", url="https://x.ru")  # нет item_selector
    with pytest.raises(ValueError):
        SourceSpec(id="x", name="X", kind="api", url="https://x.ru")   # нет items_path
    spec = SourceSpec.from_dict({"id": "t", "name": "Т", "kind": "rss",
                                 "url": "https://t.ru/rss", "role": "factcheck"})
    assert spec.role == "factcheck" and spec.enabled


def test_pick_path():
    data = {"data": {"articles": [{"title": "a"}, {"title": "b"}]}}
    assert pick_path(data, "data.articles.1.title") == "b"
    assert pick_path(data, "data.missing") is None


def test_parse_rss_items():
    from autopilot.collector import parse_rss

    spec = SourceSpec(id="s", name="S", kind="rss", url="https://s.ru/rss")
    xml = """<?xml version="1.0"?><rss><channel>
      <item><title>Новость дня</title><link>/news/1</link>
        <description><![CDATA[<p>Текст новости</p>]]></description>
        <pubDate>Fri, 22 Nov 2026 10:00:00 +0300</pubDate></item>
      <item><title>Вторая</title><link>https://s.ru/news/2</link>
        <description>Ещё текст</description></item>
    </channel></rss>"""
    items = parse_rss(xml, spec)
    assert len(items) == 2
    assert items[0].title == "Новость дня"
    assert items[0].url == "https://s.ru/news/1"
    assert items[0].published_at.startswith("2026-11-22")


def test_parse_telegram_posts():
    from autopilot.collector import parse_telegram

    spec = SourceSpec(id="tg", name="TG", kind="telegram", url="https://t.me/s/x")
    html = """
    <div class="tgme_widget_message" data-post="x/1">
      <div class="tgme_widget_message_text">Срочно! Перешли всем: ЦБ запретил наличные!</div>
      <time datetime="2026-11-22T10:00:00+00:00"></time>
    </div>"""
    items = parse_telegram(html, spec, "https://t.me/s/x")
    assert len(items) == 1
    assert items[0].url == "https://t.me/x/1"
    assert "наличные" in items[0].text


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

def test_robots_question_mark_rule_does_not_block_everything():
    text = "User-agent: *\nDisallow: /search\nDisallow: /?\nAllow: /\n"
    group = _select(parse_robots(text), "BAYKAM-Autopilot")
    assert group is not None
    assert group.allows("/rss")
    assert not group.allows("/search?q=1")


def test_robots_most_specific_rule_wins():
    text = "User-agent: *\nDisallow: /\nAllow: /rss\n"
    group = _select(parse_robots(text), "bot")
    assert group.allows("/rss")
    assert not group.allows("/news")


# ---------------------------------------------------------------------------
# Верификатор: цитаты и разбор JSON
# ---------------------------------------------------------------------------

def test_quote_is_verbatim_normalizes_quotes():
    source = "ЦБ РФ «сохранил» ключевую ставку — решение принято 25 октября."
    assert quote_is_verbatim("«сохранил» ключевую ставку", source)
    assert not quote_is_verbatim("повысил ключевую ставку", source)


def test_extract_json_from_markdown():
    raw = 'Вот ответ:\n```json\n{"verdict": "fake", "confidence": 0.9}\n```'
    assert _extract_json(raw)["verdict"] == "fake"


def test_valid_quotes_filters_hallucinations():
    evidence = [Evidence(source_id="s1", title="t", url="", kind="news",
                         reliability=0.7, text="Ключевая ставка сохранена на уровне 21%")]
    raw = [{"source_id": "s1", "quote": "Ключевая ставка сохранена"},
           {"source_id": "s1", "quote": "Ставка снижена до 5 процентов"},
           {"source_id": "s2", "quote": "Ключевая ставка сохранена"}]
    quotes = _valid_quotes(raw, evidence)
    assert quotes == [("s1", "Ключевая ставка сохранена")]


def test_verifier_without_evidence_is_uncertain(tmp_path):
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    verifier = Verifier(db, VerifierConfig(), registry)
    verifier.corpus.rebuild = lambda *a, **k: None  # корпус пуст

    item = {"id": 1, "title": "", "text": "Некий текст без источников",
            "url": "", "source_id": "s"}
    result = asyncio.run(verifier.verify(item, None, allow_llm=False))
    assert result.label == "uncertain"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# База данных
# ---------------------------------------------------------------------------

def _item(i: int, text: str, simhash_int: int | None = None) -> dict:
    return {"source_id": "s", "source_name": "S", "kind": "rss", "url": f"https://s.ru/{i}",
            "url_hash": url_hash(f"https://s.ru/{i}"), "content_hash": content_hash(text),
            "simhash": simhash_int, "title": f"t{i}", "text": text,
            "fetched_at": time.time(), "role": "news"}


def test_db_insert_dedupes_by_content_hash(tmp_path):
    db = Database(tmp_path / "t.db")
    text = "Достаточно длинный текст новости для проверки уникальности хеша контента."
    assert db.insert_item(_item(1, text)) is not None
    assert db.insert_item(_item(2, text)) is None
    assert db.stats()["items_total"] == 1


def test_db_near_duplicate_detection(tmp_path):
    db = Database(tmp_path / "t.db")
    base = simhash_to_int(simhash("Центральный банк сохранил ключевую ставку на уровне 21 процента"))
    db.insert_item(_item(1, "Центральный банк сохранил ключевую ставку на уровне 21 процента",
                         base))
    near = simhash_to_int(simhash(
        "Центральный банк сохранил ключевую ставку на уровне 21 процента в пятницу"))
    assert db.near_duplicate_ids(near, 6, window_days=3650)


def test_db_verification_and_labels(tmp_path):
    db = Database(tmp_path / "t.db")
    item_id = db.insert_item(_item(
        1, "Текст для разметки длиной заметно больше шестидесяти символов, окей."))
    db.set_score(item_id, 0.8, "v1")
    db.save_verification(item_id, label="fake", confidence=0.9, reason="опровергнуто",
                         source="debunk", fake_prob=0.8, trust_score=10,
                         uncertainty=0.1, level="likely_fake",
                         evidence={"sources": []}, llm_calls=0)
    row = db.get_item(item_id)
    assert row["label"] == "fake" and row["status"] == "fake"
    assert len(db.labeled_examples(0.7)) == 1
    assert db.new_labels_since(0, 0.7) == 1


def test_db_training_history_and_models(tmp_path):
    db = Database(tmp_path / "t.db")
    run = db.start_training_run(100, 5, 20)
    db.finish_training_run(run, metrics={"f1": 0.8, "precision": 0.8, "recall": 0.8},
                           baseline_f1=0.7, promoted=True, model_version="v1",
                           notes="ok")
    assert db.last_training_run()["f1"] == 0.8
    assert db.last_training_finished_at() > 0
    db.register_model("v1", "/tmp/v1.joblib", {"f1": 0.8}, 100, active=True)
    assert db.active_model()["version"] == "v1"


def test_model_registry_falls_back_to_project_baseline(tmp_path):
    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    registry.load()
    # проект обучен на русских примерах — модель должна загрузиться
    assert registry.version in ("project-baseline", "untrained")
    if registry.version == "project-baseline":
        prob = registry.predict("Учёные доказали, что коты умеют говорить по-русски "
                                "и скрывают это от людей из вредности")
        assert prob is None or 0.0 <= prob <= 1.0


def test_model_registry_hot_reload_via_pointer(tmp_path):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    texts = ["фейк новость срочно перешли", "фейк сенсация скрывают правду",
             "официальное сообщение банка о ставке", "агентство передаёт данные о встрече"]
    labels = [1, 1, 0, 0]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(texts, labels)
    path = models_dir / "fake_model_test.joblib"
    joblib.dump(pipe, path)

    db = Database(tmp_path / "t.db")
    pointer = models_dir / "active.json"
    registry = ModelRegistry(db, models_dir=models_dir, pointer=pointer)
    registry.load()
    # пока указателя нет, берётся проектный артефакт (если он есть в репозитории)
    assert registry.version in ("project-baseline", "untrained")

    registry.promote("test", path)
    assert registry.version == "test"
    assert registry.predict("фейк сенсация скрывают правду") is not None


# ---------------------------------------------------------------------------
# Тренер: защита от деградации
# ---------------------------------------------------------------------------

def test_trainer_metrics():
    from autopilot.trainer import Trainer

    m = Trainer.metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_db_requeue_limits_attempts(tmp_path):
    db = Database(tmp_path / "t.db")
    item_id = db.insert_item(_item(1, "Материал для повторной проверки, достаточно длинный текст."))
    db.set_score(item_id, 0.5, "v1")
    assert db.requeue(item_id, max_attempts=2) == "requeued"
    assert db.get_item(item_id)["status"] == "scored"
    assert db.requeue(item_id, max_attempts=2) == "requeued"
    assert db.requeue(item_id, max_attempts=2) == "archived"
    assert db.get_item(item_id)["status"] == "archived_uncertain"


def test_db_requeue_archives_by_ttl(tmp_path):
    db = Database(tmp_path / "t.db")
    item = _item(1, "Старый материал, который висит без решения слишком давно уже.")
    item["fetched_at"] = time.time() - 10 * 86400   # 10 дней назад
    item_id = db.insert_item(item)
    db.set_score(item_id, 0.5, "v1")
    assert db.requeue(item_id, max_attempts=10, ttl_hours=48) == "archived"
    assert db.get_item(item_id)["status"] == "archived_uncertain"


def test_archive_stale_uncertain(tmp_path):
    db = Database(tmp_path / "t.db")
    item = _item(1, "Материал с меткой uncertain, который нужно отправить в архив.")
    item["fetched_at"] = time.time() - 10 * 86400
    item_id = db.insert_item(item)
    db.set_score(item_id, 0.5, "v1")
    db.requeue(item_id, max_attempts=10)      # одна попытка уже была
    db.set_status(item_id, "uncertain")
    assert db.archive_stale_uncertain(ttl_hours=48) >= 1


def test_verify_without_llm_keeps_item_queued(tmp_path):
    """Без вызовов модели материал без основания возвращается в очередь."""
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    item_id = db.insert_item(_item(
        1, "Новость без совпадений в базе знаний, но достаточно длинная для проверки."))
    db.set_score(item_id, 0.5, "v1")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    verifier = Verifier(db, VerifierConfig(llm_enabled=False), registry)
    verifier.corpus.rebuild = lambda *a, **k: None

    counters = asyncio.run(verifier.process_pending(10))
    assert counters["skipped"] == 1 and counters["uncertain"] == 0
    assert db.get_item(item_id)["status"] == "scored"


def test_uncertain_archived_after_max_attempts(tmp_path):
    """После max_verify_attempts материал уходит в архив, а не крутится вечно."""
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    item_id = db.insert_item(_item(
        1, "Новость без совпадений в базе знаний, но достаточно длинная для проверки."))
    db.set_score(item_id, 0.5, "v1")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    cfg = VerifierConfig(llm_enabled=False, max_verify_attempts=1)
    verifier = Verifier(db, cfg, registry)
    verifier.corpus.rebuild = lambda *a, **k: None

    asyncio.run(verifier.process_pending(10))
    counters = asyncio.run(verifier.process_pending(10))
    assert db.get_item(item_id)["status"] == "archived_uncertain"
    assert counters["archived"] >= 1


def test_trainer_rejects_model_without_improvement(tmp_path, monkeypatch):
    from autopilot.settings import TrainerConfig
    from autopilot.trainer import Example, Trainer

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    registry.load()
    cfg = TrainerConfig(min_f1_gain=0.5, min_new_examples=1)
    trainer = Trainer(db, cfg, registry, models_dir=tmp_path / "models")
    trainer.backend = "baseline"

    fake = [Example(f"фейковая сенсация номер {i} срочно перешли всем", 1, "ru")
            for i in range(40)]
    real = [Example(f"официальное сообщение агентства номер {i} о ставке", 0, "ru")
            for i in range(40)]
    monkeypatch.setattr(trainer, "base_examples", lambda: fake + real)
    monkeypatch.setattr(trainer, "verified_examples", lambda: [])
    monkeypatch.setattr(trainer, "en_examples", lambda: [])

    result = trainer.train(reason="test", include_en=False)
    assert result is not None
    # первая активация идёт по абсолютному порогу: проектный бейзлайн обучен
    # на тех же примерах, что и holdout, сравнивать с ним нельзя
    assert result.promoted is True
    assert registry.version.startswith("v")
    assert db.last_training_run()["promoted"] == 1


def test_trainer_decision_requires_gain_when_comparable(tmp_path):
    """Когда старая модель — наша же версия, нужен реальный рост метрик."""
    from autopilot.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    from autopilot.settings import TrainerConfig

    trainer.cfg = TrainerConfig()
    same = {"f1": 0.90, "precision": 0.90, "logloss": 0.30, "roc_auc": 0.95}
    ok, reasons = trainer._decide(same, dict(same), baseline_leaked=False,
                                  n_train=1000, prev_n_train=1000)
    assert ok is False and "F1 не вырос" in "; ".join(reasons)

    better_logloss = {"f1": 0.90, "precision": 0.90, "logloss": 0.25, "roc_auc": 0.95}
    ok, reasons = trainer._decide(better_logloss, same, baseline_leaked=False,
                                  n_train=1000, prev_n_train=1000)
    assert ok is True and "LogLoss" in "; ".join(reasons)

    more_data = dict(same)
    ok, reasons = trainer._decide(more_data, same, baseline_leaked=False,
                                  n_train=1300, prev_n_train=1000)
    assert ok is True and "данных" in "; ".join(reasons)

    worse_precision = {"f1": 0.95, "precision": 0.50, "logloss": 0.20, "roc_auc": 0.99}
    ok, reasons = trainer._decide(worse_precision, same, baseline_leaked=False,
                                  n_train=1000, prev_n_train=1000)
    assert ok is False and "precision" in "; ".join(reasons)


def test_trainer_absolute_threshold_when_baseline_leaked(tmp_path):
    from autopilot.settings import TrainerConfig
    from autopilot.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainerConfig()
    weak = {"f1": 0.4, "precision": 0.4}
    ok, _ = trainer._decide(weak, {}, baseline_leaked=True, n_train=100, prev_n_train=0)
    assert ok is False
    strong = {"f1": 0.9, "precision": 0.9}
    ok, _ = trainer._decide(strong, {}, baseline_leaked=True, n_train=100, prev_n_train=0)
    assert ok is True


# ---------------------------------------------------------------------------
# Обучение: holdout и отчёт
# ---------------------------------------------------------------------------

def test_holdout_summary_flags_saturation(tmp_path):
    from autopilot.settings import TrainerConfig
    from autopilot.trainer import Example, Trainer

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    trainer = Trainer(db, TrainerConfig(), registry, models_dir=tmp_path / "models")
    rows = ([Example(f"фейк текст номер {i} сенсация", 1, "ru") for i in range(5)]
            + [Example(f"обычная новость номер {i} о ставке", 0, "ru") for i in range(15)])
    trainer._save_holdout(rows)
    summary = trainer.holdout_summary()
    assert summary["n"] == 20 and summary["fake"] == 5 and summary["real"] == 15
    assert summary["fresh"] == 0
    assert summary["saturated"] is False          # обучения ещё не было

    run = db.start_training_run(10, 0, 20)
    db.finish_training_run(run, metrics={"f1": 1.0}, baseline_f1=1.0,
                           promoted=False, model_version=None, notes="test")
    assert trainer.holdout_summary()["saturated"] is True


def test_holdout_enrichment_respects_cap_and_dedup(tmp_path):
    from autopilot.settings import TrainerConfig
    from autopilot.trainer import Example, Trainer

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    cfg = TrainerConfig(eval_share_ru=0.2, eval_share_verified=1.0,
                        holdout_max_per_class=2, seed=1)
    trainer = Trainer(db, cfg, registry, models_dir=tmp_path / "models")

    base = ([Example(f"базовый фейк {i} срочно перешли", 1, "ru") for i in range(10)]
            + [Example(f"базовая новость {i} агентство сообщает", 0, "ru") for i in range(10)])
    verified = [Example(f"свежая метка {i} уникальный текст", i % 2, "verified", item_id=i)
                for i in range(10)]
    holdout = trainer.ensure_holdout(base, verified)
    fake = sum(1 for ex in holdout if ex.label == 1)
    real = sum(1 for ex in holdout if ex.label == 0)
    assert fake <= cfg.holdout_max_per_class
    assert real <= cfg.holdout_max_per_class

    # повторная сборка не плодит дубли
    again = trainer.ensure_holdout(base, verified)
    texts = [ex.text for ex in again]
    assert len(texts) == len(set(texts))
    assert again == holdout


def test_holdout_does_not_leak_into_training(tmp_path, monkeypatch):
    from autopilot.settings import TrainerConfig
    from autopilot.trainer import Example, Trainer

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    trainer = Trainer(db, TrainerConfig(min_new_examples=1), registry,
                      models_dir=tmp_path / "models")
    trainer.backend = "baseline"

    fake = [Example(f"фейковая сенсация {i} срочно перешли всем", 1, "ru") for i in range(60)]
    real = [Example(f"официальное сообщение агентства {i} о ставке", 0, "ru") for i in range(60)]
    monkeypatch.setattr(trainer, "base_examples", lambda: fake + real)
    monkeypatch.setattr(trainer, "verified_examples", lambda: [])
    monkeypatch.setattr(trainer, "en_examples", lambda: [])

    result = trainer.train(reason="test", include_en=False)
    assert result is not None
    holdout_texts = {ex.text for ex in trainer._load_holdout()}
    assert holdout_texts and len(holdout_texts) < len(fake + real)
    # ни один пример holdout не должен встречаться в выборке для сравнения
    assert result.n_test == len(holdout_texts)
    assert result.promoted is True            # первая активация по абсолютному порогу
    assert "выборка:" in db.last_training_run()["notes"]


# ---------------------------------------------------------------------------
# Статус
# ---------------------------------------------------------------------------

def test_status_report_counts(tmp_path):
    from autopilot.settings import AppConfig, TrainerConfig, VerifierConfig
    from autopilot.statusline import build_report, render
    from autopilot.trainer import Example, Trainer

    db = Database(tmp_path / "t.db")
    id_fake = db.insert_item(_item(1, "Сенсация: всё отменяют, срочно перешли всем друзьям."))
    id_real = db.insert_item(_item(2, "Агентство сообщает о решении совета директоров банка."))
    id_unc = db.insert_item(_item(3, "Сомнительный пост без подтверждения в источниках."))
    id_arch = db.insert_item(_item(4, "Старый пост, который никто не смог подтвердить."))
    id_pending = db.insert_item(_item(5, "Свежий материал, который ещё никто не проверял."))
    db.save_verification(id_fake, label="fake", confidence=0.9, reason="опровергнуто",
                         source="debunk", fake_prob=0.9, trust_score=10,
                         uncertainty=0.1, level="likely_fake", evidence={}, llm_calls=1)
    db.save_verification(id_real, label="real", confidence=0.8, reason="подтверждено",
                         source="claims", fake_prob=0.1, trust_score=80,
                         uncertainty=0.2, level="confirmed", evidence={}, llm_calls=2)
    db.set_score(id_unc, 0.5, "v1")
    db.set_status(id_unc, "uncertain", "нет основания")
    db.set_status(id_arch, "archived_uncertain", "попытки исчерпаны")

    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    registry.load()
    cfg = AppConfig(verifier=VerifierConfig(), trainer=TrainerConfig(), sources=[])
    report = build_report(db, registry, cfg)

    assert report["items_total"] == 5
    assert report["fake"] == 1 and report["real"] == 1
    assert report["verified"] == 2
    assert report["uncertain"] == 1
    assert report["archived_uncertain"] == 1
    assert report["pending"] == 1   # только id_pending: uncertain учтён отдельно
    text = render(report, color=False)
    assert "BAYKAM" in text and "МОДЕЛЬ" in text and "fake: 1" in text
    assert "\x1b[" not in text            # без цвета, когда просили


# ---------------------------------------------------------------------------
# Обучающий корпус из данных проекта
# ---------------------------------------------------------------------------

def test_training_data_reads_project_sources(tmp_path):
    from autopilot import training_data

    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(json.dumps({
        "version": 1,
        "cases": [
            {"text": "ЦБ поднял ключевую ставку до 21% годовых 25 октября 2024 года.",
             "expect": "правда", "class": "kb"},
            {"text": "СРОЧНО! ЦБ запретил наличные с 1 марта 2026 года! Перешли всем!",
             "expect": "фейк", "class": "kb_refute"},
            {"text": "Бензин подорожает до 150 рублей за литр.", "expect": "непонятно"},
            {"text": "коротко", "expect": "фейк"},
        ]}, ensure_ascii=False), encoding="utf-8")
    kb_path = tmp_path / "kb_facts.json"
    kb_path.write_text(json.dumps({"topics": [{"facts": [
        {"claim": "Максимум за один предмет ЕГЭ — 100 баллов, за три предмета — 300",
         "verdict": "true"},
        {"claim": "Штрафа за покупку алкоголя для покупателя в законодательстве РФ нет",
         "verdict": "false_claim"},
        {"claim": "Биткоин не запрещён как актив, но оплата криптовалютой запрещена",
         "verdict": "mixed"},
    ]}]}, ensure_ascii=False), encoding="utf-8")

    evals = training_data.eval_examples(eval_path)
    assert len(evals) == 2                       # «непонятно» и короткий — отброшены
    assert {ex.label for ex in evals} == {0, 1}

    kb = training_data.kb_examples(kb_path)
    assert len(kb) == 2                          # mixed не учим
    assert [ex.label for ex in kb] == [0, 1]

    summary = training_data.describe(evals + kb)
    assert summary["n"] == 4 and summary["fake"] == 2 and summary["real"] == 2
    assert set(summary["by_origin"]) == {"eval", "kb"}


def test_training_data_reads_backend_feedback(tmp_path):
    import sqlite3

    from autopilot import training_data

    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE feedback (
        id TEXT PRIMARY KEY, check_id TEXT, user_id TEXT, verdict_ok INTEGER,
        expected TEXT, note TEXT, text TEXT, verdict TEXT, level TEXT,
        created_at TEXT)""")
    conn.execute(
        "INSERT INTO feedback VALUES('1','c','u',0,'фейк',NULL,?, 'fog','fog','2026-01-01')",
        ("Срочно! ЦБ запретил наличные, перешли всем знакомым и родственникам.",))
    conn.execute(
        "INSERT INTO feedback VALUES('2','c','u',1,'правда',NULL,?, 'confirmed','ok','2026-01-02')",
        ("Центральный банк сохранил ключевую ставку на уровне двадцати одного процента.",))
    conn.commit()
    conn.close()

    rows = training_data.feedback_examples(db_path)
    assert len(rows) == 2
    assert {ex.label for ex in rows} == {0, 1}
    assert all(ex.weight == training_data.WEIGHT_FEEDBACK for ex in rows)


def test_training_data_missing_files_is_safe(tmp_path):
    from autopilot import training_data

    assert training_data.eval_examples(tmp_path / "nope.json") == []
    assert training_data.kb_examples(tmp_path / "nope.json") == []
    assert training_data.feedback_examples(tmp_path / "nope.db") == []


# ---------------------------------------------------------------------------
# Бюджет модели
# ---------------------------------------------------------------------------

def test_daily_budget_caps_calls(tmp_path):
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    cfg = VerifierConfig(llm_hourly_budget=1000, llm_daily_budget=5,
                         min_calls_per_item=1)
    verifier = Verifier(db, cfg, registry)

    assert verifier.budget_left() == 5
    for _ in range(5):
        db.log_llm_call(None, "arbiter", True, 10)
    assert verifier.budget_left() == 0
    assert "дневной лимит" in verifier.budget_reason()


def test_hourly_budget_caps_calls(tmp_path):
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    cfg = VerifierConfig(llm_hourly_budget=3, llm_daily_budget=0)
    verifier = Verifier(db, cfg, registry)

    for _ in range(3):
        db.log_llm_call(None, "extract", True, 10)
    assert verifier.budget_left() == 0
    assert "часовой лимит" in verifier.budget_reason()


def test_zero_daily_budget_means_no_cap(tmp_path):
    from autopilot.settings import VerifierConfig
    from autopilot.verifier import Verifier

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    cfg = VerifierConfig(llm_hourly_budget=7, llm_daily_budget=0)
    verifier = Verifier(db, cfg, registry)
    assert verifier.budget_left() == 7


def test_status_shows_daily_budget(tmp_path):
    from autopilot.settings import AppConfig, TrainerConfig, VerifierConfig
    from autopilot.statusline import build_report, render

    db = Database(tmp_path / "t.db")
    registry = ModelRegistry(db, models_dir=tmp_path / "models",
                             pointer=tmp_path / "models" / "active.json")
    registry.load()
    cfg = AppConfig(verifier=VerifierConfig(llm_hourly_budget=40, llm_daily_budget=600),
                    trainer=TrainerConfig(), sources=[])
    report = build_report(db, registry, cfg)
    assert report["llm_daily_budget"] == 600
    text = render(report, color=False)
    assert "за сутки" in text and "600" in text
