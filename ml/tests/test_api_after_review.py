"""API-проверки после доработок: ссылка вместо текста, история, миграции."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app  # noqa: E402
from backend import store  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VERDICT_DB", str(tmp_path / "test.db"))
    return TestClient(app)


@pytest.fixture()
def auth(client):
    res = client.post("/api/users")
    assert res.status_code == 200
    return {"X-Token": res.json()["token"]}


def test_url_only_check_accepts_link(client, auth, monkeypatch):
    """Ссылка вместо текста — штатный сценарий «поделиться» из мессенджера.

    Сеть в тестах не нужна: подменяем извлечение статьи со страницы.
    """
    import lakmus.sources as sources_mod
    from lakmus.article import Article

    monkeypatch.setattr(
        sources_mod,
        "fetch_article",
        lambda url, timeout=12: Article(
            title="МРОТ вырос с января",
            text="МРОТ с 1 января 2025 года составляет 22 440 рубля в месяц.",
            domain="example.com",
            extractor="fallback",
        ),
    )
    res = client.post(
        "/api/check",
        json={"url": "https://example.com/news/1"},
        headers=auth,
    )
    assert res.status_code == 200, res.text
    assert res.json()["verdict"] in ("правда", "фейк", "непонятно")


def test_bad_scheme_rejected(client, auth):
    res = client.post("/api/check", json={"url": "ftp://example.com/x"}, headers=auth)
    assert res.status_code == 422


def test_private_url_not_fetched(client, auth):
    """SSRF: ссылка внутрь сети не должна давать вердикт по внутренней странице."""
    res = client.post(
        "/api/check",
        json={"url": "http://127.0.0.1:8000/api/health"},
        headers=auth,
    )
    # Страница не скачалась — текста нет, значит просим ввести сообщение.
    assert res.status_code == 422


def test_text_still_works(client, auth):
    res = client.post(
        "/api/check",
        json={"text": "МРОТ с 1 января 2025 года — 22 440 рублей."},
        headers=auth,
    )
    assert res.status_code == 200
    assert res.json()["verdict"] == "правда"


def test_history_returns_user_checks(client, auth):
    client.post(
        "/api/check",
        json={"text": "МРОТ с 1 января 2025 года — 22 440 рублей."},
        headers=auth,
    )
    res = client.get("/api/history", headers=auth)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) >= 1
    assert items[0]["passport"].startswith("/p/")


def test_history_requires_auth(client):
    assert client.get("/api/history").status_code == 401


def test_old_database_is_migrated(tmp_path, monkeypatch):
    """База без колонки email раньше падала на старте: no such column."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, plan TEXT NOT NULL,"
        " token TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO users(id, plan, token, created_at) VALUES('u1','free','tok','2026-01-01')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("VERDICT_DB", str(db))
    user = store.user_by_token("tok")
    assert user is not None
    assert "email" in user, "колонка email должна появиться после миграции"


def test_expired_subscription_downgrades(tmp_path, monkeypatch):
    """Оплаченный месяц должен заканчиваться: срок раньше не проверялся."""
    monkeypatch.setenv("VERDICT_DB", str(tmp_path / "exp.db"))
    u = store.create_user()
    payment = store.create_payment(u["id"], "pro", "card")
    store.confirm_payment(payment["id"])
    assert store.user_by_token(u["token"])["plan"] == "pro"

    with store.connect() as conn:
        conn.execute(
            "UPDATE users SET expires_at = datetime('now','-1 day') WHERE id = ?",
            (u["id"],),
        )
    assert store.user_by_token(u["token"])["plan"] == "free"
