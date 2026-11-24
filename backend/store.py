"""Хранилище проверок, топов и подписок. SQLite на stdlib.

Три сущности:
  checks   — каждая проверка, включая анонимные;
  top      — агрегат по тексту: сколько раз проверяли, какой вердикт;
  users    — тариф (free / pro) и ключ доступа.

Топы считаются из checks, а не ведутся отдельно: это защита от
рассинхрона между счётчиком и историей.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "verdict.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    title       TEXT,
    text_hash   TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    score       INTEGER NOT NULL,
    uncertainty REAL NOT NULL,
    level       TEXT NOT NULL,
    full_json   TEXT NOT NULL,
    user_id     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checks_hash ON checks(text_hash);
CREATE INDEX IF NOT EXISTS idx_checks_created ON checks(created_at);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    plan        TEXT NOT NULL DEFAULT 'free',
    token       TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT,
    email       TEXT,
    pass_hash   TEXT,
    name        TEXT
);
-- Индекс по email создаётся отдельно, ПОСЛЕ миграции: на старой базе
-- колонки email нет, и CREATE INDEX падал с «no such column».


CREATE TABLE IF NOT EXISTS payments (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    plan          TEXT NOT NULL,
    amount_rub    INTEGER NOT NULL,
    status        TEXT NOT NULL,
    method        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    confirmed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_payments_user ON payments(user_id);

CREATE TABLE IF NOT EXISTS top_pins (
    text_hash   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    pinned_at   TEXT NOT NULL
);

-- Корпоративный доступ: ключ вместо токена пользователя.
-- Храним только хеш: утечка базы не отдаёт рабочие ключи.
CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    label        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

-- Обратная связь по вердикту: основа дообучения на реальных ошибках.
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    check_id    TEXT NOT NULL,
    user_id     TEXT,
    verdict_ok  INTEGER NOT NULL,
    expected    TEXT,
    note        TEXT,
    text        TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    level       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_check ON feedback(check_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
"""

FREE_CHECKS_PER_DAY = 10


#: Колонки, добавленные после первого релиза. CREATE TABLE IF NOT EXISTS
#: не расширяет уже созданную таблицу, поэтому старые базы (например, у
#: того, кто обновил проект) падали на «no such column: email».
_MIGRATIONS: dict[str, dict[str, str]] = {
    "users": {
        "email": "TEXT",
        "pass_hash": "TEXT",
        "name": "TEXT",
        "expires_at": "TEXT",
    },
    "checks": {
        "user_id": "TEXT",
        # Заголовок статьи, если проверяли ссылку: в истории он читается
        # лучше простыни текста и не тащит меню и подвал страницы.
        "title": "TEXT",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Добавить колонки, которых нет в старой базе. Безопасно к повтору."""
    for table, columns in _MIGRATIONS.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not existing:
            continue  # таблицы ещё нет — её создаст SCHEMA
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    # Уникальный индекс по email создаём только после миграции: на старой
    # таблице без этой колонки SCHEMA падала ещё до первой строки.
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)"
            " WHERE email IS NOT NULL"
        )
    except sqlite3.OperationalError:
        pass


def db_path() -> Path:
    return Path(os.environ.get("VERDICT_DB") or DEFAULT_DB)


#: Колонки, добавленные после первого релиза. `CREATE TABLE IF NOT EXISTS`
#: не меняет уже существующую таблицу, поэтому на старой базе приложение
#: падало с «no such column: email». Досоздаём недостающие колонки здесь.
_MIGRATIONS: dict[str, dict[str, str]] = {
    "users": {
        "expires_at": "TEXT",
        "email": "TEXT",
        "pass_hash": "TEXT",
        "name": "TEXT",
    },
    "checks": {
        "user_id": "TEXT",
        # Заголовок статьи, если проверяли ссылку: в истории он читается
        # лучше простыни текста и не тащит меню и подвал страницы.
        "title": "TEXT",
    },
    "payments": {
        "confirmed_at": "TEXT",
    },
}


#: Индексы, которые зависят от колонок из миграций.
_POST_MIGRATION_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)
    WHERE email IS NOT NULL;
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Довести схему существующей базы до текущей, не теряя данные."""
    for table, columns in _MIGRATIONS.items():
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            continue  # таблицы ещё нет — её создаст SCHEMA
        have = {row[1] for row in rows}
        for name, kind in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    # Индексы по новым колонкам — только теперь, когда они точно есть.
    conn.executescript(_POST_MIGRATION_SQL)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    # «:memory:» — не файл: каждое соединение создавало бы пустую базу,
    # и пользователь исчезал между запросами. Держим одно общее соединение
    # с общим кэшем, чтобы in-memory база жила весь процесс.
    if str(path) == ":memory:":
        global _MEMORY_CONN
        try:
            _MEMORY_CONN  # type: ignore[name-defined]
        except NameError:
            _MEMORY_CONN = sqlite3.connect(
                "file:witness_mem?mode=memory&cache=shared",
                uri=True, timeout=10, check_same_thread=False,
            )
            _MEMORY_CONN.row_factory = sqlite3.Row
        _MEMORY_CONN.executescript(SCHEMA)
        try:
            yield _MEMORY_CONN
            _MEMORY_CONN.commit()
        finally:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        # Порядок важен: SCHEMA создаёт таблицы, потом добираем колонки,
        # которых не было в старой базе (например, email у пользователей).
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def text_hash(text: str) -> str:
    """Одинаковый текст → одинаковый хеш. Основа топов."""
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------- пользователи и тарифы ----------

def create_user(plan: str = "free") -> dict[str, Any]:
    """Гостевой пользователь без пароля. Нужен для внутренних сценариев."""
    uid = f"usr_{secrets.token_hex(4)}"
    token = secrets.token_urlsafe(24)
    with connect() as conn:
        conn.execute(
            "INSERT INTO users(id, plan, token, created_at) VALUES(?,?,?,?)",
            (uid, plan, token, now_iso()),
        )
    return {"id": uid, "plan": plan, "token": token}


# ---------- регистрация и вход ----------

#: Параметры хеширования пароля. pbkdf2 на stdlib: без внешних зависимостей.
_PBKDF2_ROUNDS = 200_000


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Пароль → «соль$хеш». Соль случайная, если не передана."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ROUNDS
    )
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Сравнение без утечки по времени."""
    if not stored or "$" not in stored:
        return False
    salt, expected = stored.split("$", 1)
    actual = hash_password(password, salt).split("$", 1)[1]
    return hmac.compare_digest(actual, expected)


def register_user(email: str, password: str, name: str = "") -> Optional[dict[str, Any]]:
    """Создать аккаунт. None — email уже занят."""
    email = (email or "").strip().lower()
    uid = f"usr_{secrets.token_hex(4)}"
    token = secrets.token_urlsafe(24)
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users(id, plan, token, created_at, email, pass_hash, name)"
                " VALUES(?,?,?,?,?,?,?)",
                (uid, "free", token, now_iso(), email, hash_password(password), name.strip()),
            )
    except sqlite3.IntegrityError:
        return None
    return {
        "id": uid, "plan": "free", "token": token, "email": email,
        "name": name.strip(), "expires_at": None,
    }


def authenticate(email: str, password: str) -> Optional[dict[str, Any]]:
    """Проверить email и пароль. None — неверные данные."""
    email = (email or "").strip().lower()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    if not row or not verify_password(password, row["pass_hash"] or ""):
        return None
    user = dict(row)
    # Новый токен на каждый вход: старые сессии не остаются валидными
    # навсегда, если токен утёк.
    token = secrets.token_urlsafe(24)
    with connect() as conn:
        conn.execute("UPDATE users SET token = ? WHERE id = ?", (token, user["id"]))
    user["token"] = token
    return user


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    """То, что можно отдавать наружу: без хеша пароля."""
    return {
        "id": user["id"],
        "plan": user["plan"],
        "email": user.get("email"),
        "name": user.get("name") or "",
        "expires_at": user.get("expires_at"),
        "created_at": user.get("created_at"),
    }


def user_by_token(token: str) -> Optional[dict[str, Any]]:
    """Пользователь по токену. Истёкшая подписка понижается до free.

    Раньше срок писался при оплате и нигде не проверялся: оплаченный
    месяц не заканчивался никогда. Проверка здесь, а не в обработчике,
    чтобы её видели все пути — и /api/check, и /api/me, и платежи.
    """
    if not token:
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None
        user = dict(row)
        if _subscription_expired(user):
            conn.execute(
                "UPDATE users SET plan = 'free', expires_at = NULL WHERE id = ?",
                (user["id"],),
            )
            user["plan"] = "free"
            user["expires_at"] = None
    return user


def _subscription_expired(user: dict[str, Any]) -> bool:
    """True, если платный тариф истёк.

    expires_at пишется в формате SQLite `YYYY-MM-DD HH:MM:SS` (UTC).
    Сравниваем в SQL, чтобы не зависеть от формата строки в Python.
    """
    if user.get("plan", "free") == "free":
        return False
    expires = user.get("expires_at")
    if not expires:
        return False
    with connect() as conn:
        row = conn.execute(
            "SELECT datetime('now') > datetime(?) AS expired", (expires,)
        ).fetchone()
    return bool(row and row["expired"])


PLAN_PRICES: dict[str, int] = {"free": 0, "pro": 300, "team": 5000}
PLAN_DAYS: dict[str, int] = {"free": 0, "pro": 30, "team": 30}


def upgrade_user(token: str, plan: str = "pro") -> bool:
    """Прямой апгрейд без оплаты. Только для тестов и админки.

    Пользовательский путь идёт через платежи: см. create_payment /
    confirm_payment. Здесь оставлено для внутренних сценариев.
    """
    if plan not in PLAN_PRICES:
        return False
    with connect() as conn:
        cur = conn.execute("UPDATE users SET plan = ? WHERE token = ?", (plan, token))
    return cur.rowcount > 0


# ---------- платежи ----------

def create_payment(user_id: str, plan: str, method: str = "card") -> Optional[dict[str, Any]]:
    """Создать платёж в статусе pending. None — неизвестный тариф."""
    if plan not in PLAN_PRICES or plan == "free":
        return None
    payment_id = f"pay_{secrets.token_hex(6)}"
    amount = PLAN_PRICES[plan]
    with connect() as conn:
        conn.execute(
            "INSERT INTO payments(id, user_id, plan, amount_rub, status, method,"
            " created_at) VALUES(?,?,?,?,?,?,?)",
            (payment_id, user_id, plan, amount, "pending", method, now_iso()),
        )
    return {
        "id": payment_id,
        "plan": plan,
        "amount_rub": amount,
        "status": "pending",
        "method": method,
        "created_at": now_iso(),
    }


def get_payment(payment_id: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM payments WHERE id = ?", (payment_id,)
        ).fetchone()
    return dict(row) if row else None


def confirm_payment(payment_id: str) -> Optional[dict[str, Any]]:
    """Провести платёж: тариф пользователя повышается, срок продлевается.

    Идемпотентно: повторное подтверждение не продлевает подписку дважды.
    """
    payment = get_payment(payment_id)
    if payment is None:
        return None
    if payment["status"] == "paid":
        return {**payment, "already_paid": True}

    stamp = now_iso()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE payments SET status = 'paid', confirmed_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (stamp, payment_id),
        )
        if cur.rowcount == 0:
            return get_payment(payment_id)

        days = PLAN_DAYS.get(payment["plan"], 30)
        conn.execute(
            "UPDATE users SET plan = ?, expires_at = datetime('now', ?) "
            "WHERE id = ?",
            (payment["plan"], f"+{days} days", payment["user_id"]),
        )
    result = get_payment(payment_id) or payment
    return result


def payment_history(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM payments WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def checks_today(user_id: str) -> int:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE user_id = ? AND created_at LIKE ?",
            (user_id, f"{day}%"),
        ).fetchone()
    return int(row["n"]) if row else 0


def free_quota_left(user_id: str) -> int:
    return max(0, FREE_CHECKS_PER_DAY - checks_today(user_id))


# ---------- проверки ----------

def save_check(
    *,
    text: str,
    verdict: str,
    score: int,
    uncertainty: float,
    level: str,
    full: dict[str, Any],
    user_id: Optional[str] = None,
    check_id: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Сохранить проверку. check_id можно передать — тогда ответ API и база
    указывают на одну и ту же запись (иначе фидбек по id не находился)."""
    check_id = check_id or f"chk_{secrets.token_hex(5)}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO checks(id, text, title, text_hash, verdict, score,"
            " uncertainty, level, full_json, user_id, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                check_id,
                text,
                title,
                text_hash(text),
                verdict,
                int(score),
                float(uncertainty),
                level,
                json.dumps(full, ensure_ascii=False),
                user_id,
                now_iso(),
            ),
        )
    return check_id


def get_check(check_id: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM checks WHERE id = ?", (check_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["full"] = json.loads(data.pop("full_json"))
    return data


def user_checks(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """История проверок пользователя — свежие сверху.

    Главред назвал историю «базовой» вещью: человек проверял новость
    вчера и хочет вернуться к вердикту, а не искать сообщение заново.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, title, verdict, score, uncertainty, level,"
            " text_hash, created_at"
            " FROM checks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_by_hash(text_hash_value: str) -> Optional[dict[str, Any]]:
    """Последняя проверка этого утверждения — основа «паспорта».

    Паспорт один на утверждение, а не на каждую проверку: сколько бы раз
    текст ни проверяли, ссылка ведёт на актуальный результат.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM checks WHERE text_hash = ? ORDER BY created_at DESC LIMIT 1",
            (text_hash_value,),
        ).fetchone()
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE text_hash = ?",
            (text_hash_value,),
        ).fetchone()["n"]
    if not row:
        return None
    data = dict(row)
    data["full"] = json.loads(data.pop("full_json"))
    data["checks_count"] = int(count)
    return data


def check_count_by_hash(text_hash_value: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE text_hash = ?",
            (text_hash_value,),
        ).fetchone()
    return int(row["n"]) if row else 0


# ---------- топы ----------

def top_items(limit: int = 20, window_hours: int = 48) -> list[dict[str, Any]]:
    """Топ проверяемых утверждений за окно.

    Считается по text_hash: одна и та же новость от ста человек — одна
    строка с счётчиком 100. Показываем и вердикт, и разброс мнений.

    Плюс «живость»: сколько проверок за последний час и минуту. Это не
    декорация — по этим числам видно, что утверждение сейчас на волне,
    а не просто накопило счётчик за неделю.
    """
    now = time.time()
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - window_hours * 3600))
    hour_ago = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 3600))
    minute_ago = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60))

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT text_hash,
                   MIN(text)                         AS sample_text,
                   COUNT(*)                          AS checks_count,
                   MAX(created_at)                   AS last_at,
                   SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_hour,
                   SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_minute,
                   (SELECT verdict FROM checks c2
                     WHERE c2.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS verdict,
                   (SELECT score FROM checks c3
                     WHERE c3.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS score,
                   (SELECT level FROM checks c4
                     WHERE c4.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS level,
                   COUNT(DISTINCT verdict)           AS verdict_variants
              FROM checks c
             WHERE created_at >= ?
             GROUP BY text_hash
             ORDER BY last_hour DESC, checks_count DESC, last_at DESC
             LIMIT ?
            """,
            (hour_ago, minute_ago, since, int(limit)),
        ).fetchall()
        pinned = {r["text_hash"] for r in conn.execute("SELECT text_hash FROM top_pins")}
    out = []
    for row in rows:
        item = dict(row)
        item["pinned"] = item["text_hash"] in pinned
        item["last_hour"] = int(item.get("last_hour") or 0)
        item["last_minute"] = int(item.get("last_minute") or 0)
        out.append(item)
    return out


def pin_top(text_hash_value: str, title: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO top_pins(text_hash, title, pinned_at) VALUES(?,?,?)",
            (text_hash_value, title[:200], now_iso()),
        )


def stats() -> dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM checks").fetchone()["n"]
        unique = conn.execute("SELECT COUNT(DISTINCT text_hash) AS n FROM checks").fetchone()["n"]
        fake = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE level = 'likely_fake'"
        ).fetchone()["n"]
        real = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE level IN ('confirmed','likely_real')"
        ).fetchone()["n"]
        fog = conn.execute("SELECT COUNT(*) AS n FROM checks WHERE level = 'fog'").fetchone()["n"]
        pro = conn.execute("SELECT COUNT(*) AS n FROM users WHERE plan = 'pro'").fetchone()["n"]
    return {
        "checks": int(total),
        "unique_claims": int(unique),
        "fake": int(fake),
        "real": int(real),
        "fog": int(fog),
        "pro_users": int(pro),
    }


# ---------- API-ключи для корпоративного доступа ----------

def _key_hash(raw: str) -> str:
    """Ключ храним только хешем: из базы его не восстановить."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_api_key(user_id: str, label: str = "") -> dict[str, Any]:
    """Выпустить ключ. Сырое значение возвращается один раз и нигде не хранится."""
    raw = "rvz_" + secrets.token_urlsafe(32)
    key_id = f"key_{secrets.token_hex(5)}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO api_keys(id, user_id, key_hash, prefix, label, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (key_id, user_id, _key_hash(raw), raw[:12], label[:120], now_iso()),
        )
    return {
        "id": key_id,
        "key": raw,
        "prefix": raw[:12],
        "label": label[:120],
        "created_at": now_iso(),
    }


def list_api_keys(user_id: str) -> list[dict[str, Any]]:
    """Список ключей без самих ключей — только префикс и даты."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, prefix, label, created_at, last_used_at, revoked_at"
            " FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(user_id: str, key_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET revoked_at = ?"
            " WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
            (now_iso(), key_id, user_id),
        )
    return cur.rowcount > 0


def user_by_api_key(raw: str) -> Optional[dict[str, Any]]:
    """Пользователь по ключу. Отозванный ключ не работает."""
    if not raw:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM api_keys k JOIN users u ON u.id = k.user_id"
            " WHERE k.key_hash = ? AND k.revoked_at IS NULL",
            (_key_hash(raw),),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (now_iso(), _key_hash(raw)),
            )
    return dict(row) if row else None


def api_key_usage(user_id: str, days: int = 30) -> dict[str, Any]:
    """Сколько проверок сделано по ключам за период — для отчёта корпорации."""
    since = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400)
    )
    with connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE user_id = ? AND created_at >= ?",
            (user_id, since),
        ).fetchone()["n"]
    return {"days": days, "checks": int(total)}


# ---------- обратная связь ----------

def save_feedback(
    *,
    check_id: str,
    text: str,
    verdict: str,
    level: str,
    verdict_ok: bool,
    expected: Optional[str] = None,
    note: str = "",
    user_id: Optional[str] = None,
) -> str:
    feedback_id = f"fb_{secrets.token_hex(5)}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback(id, check_id, user_id, verdict_ok, expected, note,"
            " text, verdict, level, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                feedback_id,
                check_id,
                user_id,
                1 if verdict_ok else 0,
                expected,
                note[:1000],
                text,
                verdict,
                level,
                now_iso(),
            ),
        )
    return feedback_id


def feedback_stats(days: int = 30) -> dict[str, Any]:
    since = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400)
    )
    with connect() as conn:
        rows = conn.execute(
            "SELECT level, verdict_ok, COUNT(*) AS n FROM feedback"
            " WHERE created_at >= ? GROUP BY level, verdict_ok",
            (since,),
        ).fetchall()
    by_level: dict[str, dict[str, int]] = {}
    total = 0
    ok = 0
    for row in rows:
        level = row["level"] or "unknown"
        bucket = by_level.setdefault(level, {"total": 0, "ok": 0, "wrong": 0})
        bucket["total"] += int(row["n"])
        total += int(row["n"])
        if row["verdict_ok"]:
            bucket["ok"] += int(row["n"])
            ok += int(row["n"])
        else:
            bucket["wrong"] += int(row["n"])
    return {
        "days": days,
        "total": total,
        "ok": ok,
        "wrong": total - ok,
        "accuracy": round(ok / total, 4) if total else None,
        "by_level": by_level,
    }


def feedback_examples(limit: int = 200, only_wrong: bool = True) -> list[dict[str, Any]]:
    """Примеры для дообучения: текст, что ответили, что ждали.

    Это не «обучение модели» само по себе — это корпус. Обучение идёт
    отдельным шагом, чтобы ошибка в живом сервисе не меняла модель молча.
    """
    query = "SELECT * FROM feedback"
    if only_wrong:
        query += " WHERE verdict_ok = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    with connect() as conn:
        rows = conn.execute(query, (int(limit),)).fetchall()
    return [dict(r) for r in rows]


# ---------- лента проверенных новостей ----------

def feed_items(limit: int = 30, hours: int = 168) -> list[dict[str, Any]]:
    """Лента: что уже проверили и с каким итогом.

    Отличие от топов: топ — рейтинг сомнений (что проверяют чаще),
    лента — витрина результата (что проверено и признано правдой).
    """
    since = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600)
    )
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT text_hash,
                   MIN(text)       AS sample_text,
                   COUNT(*)        AS checks_count,
                   MAX(created_at) AS last_at,
                   (SELECT verdict FROM checks c2
                     WHERE c2.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS verdict,
                   (SELECT level FROM checks c3
                     WHERE c3.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS level,
                   (SELECT score FROM checks c4
                     WHERE c4.text_hash = c.text_hash
                     ORDER BY created_at DESC LIMIT 1) AS score
              FROM checks c
             WHERE created_at >= ?
             GROUP BY text_hash
             ORDER BY last_at DESC
             LIMIT ?
            """,
            (since, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]
