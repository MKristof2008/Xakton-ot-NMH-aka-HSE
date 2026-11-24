"""Три входа проверки: текст, ссылка, скриншот — один контракт ответа.

Эксперт на встрече спрашивал именно это: «работает ли со ссылками из
мессенджеров» и «можно ли закинуть скриншот». Тесты держат эти входы
живыми, чтобы они снова не остались только в ML.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("VERDICT_DB", ":memory:")

from backend.main import app  # noqa: E402

client = TestClient(app)
# Проверка доступна только с аккаунтом. Токен берём на каждый тест:
# in-memory база общая для процесса, и токен, созданный при импорте,
# к моменту теста может относиться к уже пересозданной базе.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _fresh_auth(monkeypatch):
    global AUTH
    AUTH = {"X-Token": client.post("/api/users").json()["token"]}

TEXT = "СРОЧНО! ЦБ запретил наличные с 1 марта 2026 года!"

# Минимальный валидный PNG 1×1 — достаточно, чтобы сигналы изображения
# отработали и вернули вердикт, а не упали на битом файле.
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001"
)


def test_text_check_returns_contract(auth_headers):
    r = client.post("/api/check", json={"text": TEXT}, headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in ("правда", "фейк", "непонятно")
    assert body["passport"].startswith("/p/")
    assert "level" in body and "score" in body


def test_check_without_account_rejected():
    """Без токена проверок нет: иначе лимиты free обходились бы перезагрузкой."""
    r = client.post("/api/check", json={"text": TEXT})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


def test_short_text_rejected(auth_headers):
    r = client.post("/api/check", json={"text": "ой"}, headers=auth_headers)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "BAD_INPUT"


def test_url_check_accepted(auth_headers, monkeypatch):
    """Ссылка без текста — валидный вход: текст берётся со страницы."""
    import lakmus.sources as sources_mod
    from lakmus.article import Article

    monkeypatch.setattr(
        sources_mod,
        "fetch_article",
        lambda url, timeout=12: Article(
            title="МРОТ вырос с января",
            text="МРОТ с 1 января 2025 года составляет 22 440 рубля в месяц.",
            domain="example.invalid",
            extractor="fallback",
        ),
    )
    r = client.post(
        "/api/check",
        json={"url": "https://example.invalid/news/1"},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in ("правда", "фейк", "непонятно")
    assert body["passport"].startswith("/p/")


def test_url_unreachable_asks_for_text(auth_headers, monkeypatch):
    """Страница не скачалась — честно просим текст, а не выдумываем вердикт."""
    import lakmus.sources as sources_mod

    monkeypatch.setattr(sources_mod, "fetch_article", lambda url, timeout=12: None)
    r = client.post(
        "/api/check",
        json={"url": "https://example.invalid/news/1"},
        headers=auth_headers,
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "BAD_INPUT"


def test_url_without_scheme_rejected(auth_headers):
    r = client.post(
        "/api/check", json={"url": "example.com/news"}, headers=auth_headers
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "BAD_INPUT"


def test_image_check_via_multipart(auth_headers):
    r = client.post(
        "/api/check",
        files={"image": ("shot.png", PNG_1PX, "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in ("правда", "фейк", "непонятно")
    assert body["passport"].startswith("/p/")


def test_image_wrong_type_rejected(auth_headers):
    r = client.post(
        "/api/check",
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        headers=auth_headers,
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "BAD_INPUT"


def test_text_with_image_together(auth_headers):
    """Скриншот плюс подпись: текст идёт в проверку, картинка — в сигналы."""
    r = client.post(
        "/api/check",
        data={"text": TEXT},
        files={"image": ("shot.png", PNG_1PX, "image/png")},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] in ("правда", "фейк", "непонятно")
