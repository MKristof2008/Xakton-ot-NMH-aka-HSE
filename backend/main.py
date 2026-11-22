"""HTTP API «Ревизор» — фактчекинг по замыслу эксперта NMH.

Две ступени по подписке:
  free — вердикт и уровень доверия;
  pro  — плюс разбор: сигналы, доказательства, источники, что проверить дальше.

Плюс вкладка топов: что проверяют сейчас, по счётчику одинаковых запросов.

Запуск: py -3 -m uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(BACKEND_DIR))

# .env ищется рядом с backend и в корне проекта: так удобнее и для
# локального запуска, и для Docker.
for candidate in (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    ML_DIR / ".env",
):
    if candidate.exists():
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from lakmus.context import CheckContext  # noqa: E402
from lakmus.pipeline import check as run_check  # noqa: E402
from lakmus.full_pipeline import run_check as run_full_check  # noqa: E402
from lakmus.schemas import CheckInput, InputType  # noqa: E402
from lakmus.sources import fetch_page  # noqa: E402
from lakmus.llm_pipeline import explain_media  # noqa: E402
from lakmus.tariffs import for_plan  # noqa: E402

import store  # noqa: E402
import autopilot_feed  # noqa: E402
from passport import passport_html  # noqa: E402
from quote_page import build_page as build_quote_page  # noqa: E402
from sources_page import load_topics, page_html as sources_page_html  # noqa: E402

MAX_TEXT = 4000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_VIDEO_BYTES = 25 * 1024 * 1024
VIDEO_TYPES = ("video/mp4", "video/quicktime")

app = FastAPI(title="Ревизор — фактчекинг API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Token", "X-Api-Key"],
)


# ---------- ошибки в одном формате ----------

def fail(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(404)
async def not_found(request: Request, exc: Any) -> JSONResponse:
    return fail("NOT_FOUND", "такого адреса нет", 404)


# ---------- модели ----------

class CheckRequest(BaseModel):
    # Текст необязателен: можно прислать только ссылку из мессенджера
    # или только скриншот — оба сценария штатные.
    text: str = Field(default="", max_length=MAX_TEXT)
    #: Ссылка на проверяемую страницу: нужна, чтобы распознать
    #: сатирический или недостоверный сайт до всякой модели, и как
    #: замена тексту.
    url: Optional[str] = None
    url: Optional[str] = None


class UpgradeRequest(BaseModel):
    plan: str = "pro"


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class PaymentRequest(BaseModel):
    plan: str = "pro"
    method: str = "card"


class PaymentConfirm(BaseModel):
    payment_id: str


class PinRequest(BaseModel):
    text_hash: str
    title: str = ""


class ApiKeyRequest(BaseModel):
    label: str = ""


class FeedbackRequest(BaseModel):
    check_id: str
    verdict_ok: bool
    expected: Optional[str] = None
    note: str = ""


# ---------- сборка ответа ----------

def level_label(level: str) -> str:
    return {
        "confirmed": "Подтверждено",
        "likely_real": "Скорее правда",
        "fog": "Туман",
        "doubtful": "Сомнительно",
        "likely_fake": "Скорее фейк",
    }.get(level, level)


def verdict_of(result: Any) -> str:
    """Короткий вердикт для бесплатного тарифа — одно слово."""
    level = result.trust.level.value
    if level in ("confirmed", "likely_real"):
        return "правда"
    if level in ("likely_fake", "doubtful"):
        return "фейк"
    return "непонятно"


def _detector_explanation(result: Any) -> str:
    """Объяснение вердикта по медиа без модели: собираем из сигналов.

    Модель переводит отчёт детекторов на человеческий язык, но её может не
    быть. Тогда объяснение всё равно нужно: вердикт без обоснования — это
    ровно то, за что ругают «нейросеть сказала».
    """
    parts: list[str] = []
    for signal in result.signals or []:
        verdict = getattr(getattr(signal, "verdict", None), "value", "")
        if verdict in ("supports_fake", "supports_real"):
            parts.append(str(signal.summary or "").strip())
    if not parts:
        parts = [
            str(s.summary or "").strip()
            for s in (result.signals or [])
            if getattr(s, "summary", "")
        ][:2]
    if not parts:
        return "Признаков для вывода не нашлось: данных слишком мало."
    return " ".join(p for p in parts[:3] if p)[:600]


def _clean_url(value: Optional[str]) -> Optional[str]:
    """Проверить, что ссылка выглядит как http(s), иначе не использовать."""
    if not value:
        return None
    candidate = value.strip()
    if not re.match(r"^https?://[^\s]+$", candidate, re.I):
        return None
    return candidate[:2000]


def outcome_payload(outcome: Any, check_id: str, plan: str) -> dict[str, Any]:
    """Ответ по тарифу из результата полного пайплайна."""
    base: dict[str, Any] = {
        "id": check_id,
        "verdict": outcome.verdict,
        "level": outcome.level,
        "level_label": level_label(outcome.level),
        "score": outcome.score,
        "uncertainty": round(outcome.uncertainty, 4),
        "plan": plan,
        "mode": outcome.mode,
    }
    # Тема видна всем: она объясняет, по каким правилам шла проверка,
    # и не раскрывает разбор.
    if getattr(outcome, "tag", None):
        base["tag"] = outcome.tag
    # Границы проверки отдаются всем тарифам. Это не разбор, а честность:
    # человек должен видеть, чего система не знает, до покупки подписки.
    # Жюри спрашивает ровно об этом.
    base["limits"] = outcome.limitations
    if plan not in ("pro", "team"):
        base["upgrade_hint"] = (
            "Разбор — почему такой вердикт, источники и что проверить дальше — "
            "доступен по подписке."
        )
        return base

    base["claims"] = [c.as_dict() for c in outcome.claims]
    base["sources_used"] = outcome.sources_used
    base["duration_ms"] = outcome.duration_ms
    # Рубрика: из чего сложился балл и вердикт. Это и есть платный разбор.
    if getattr(outcome, "reasoning", None):
        base["reasoning"] = outcome.reasoning
    return base


def media_payload(
    result: Any, check_id: str, plan: str, kind: str
) -> dict[str, Any]:
    """Ответ по фото и видео: правила вывода + чек-лист проверки.

    Медиа проверяется не источниками, а отчётом детекторов. Поэтому здесь
    тот же принцип, что у текста: вердикт собирается по явным правилам
    (``verdict_rules``), а не берётся из «мнения» модели. Модель только
    объясняет отчёт детекторов словами.
    """
    from lakmus import playbook, verdict_rules

    base = public_payload(result, check_id, plan)
    rules = verdict_rules.rules_for_media(result.signals)

    # Тема для медиа — тип материала: фото или видео.
    base["tag"] = {
        "key": kind,
        "title": "Фото и изображение" if kind == "image" else "Видео",
        "hits": [],
        "risk": 2,
    }
    # Чек-лист виден всем: он объясняет, что вообще проверялось.
    base["playbook"] = playbook.for_material(kind).as_dict()

    if plan not in ("pro", "team"):
        return base

    base["verdict_rules"] = rules.as_dict()
    base["reasoning"] = {
        "confidence": rules.confidence,
        "verdict": rules.verdict,
        "summary": rules.summary,
        "tag": {"key": kind, "title": base["tag"]["title"]},
        "rules_applied": [h.reason for h in rules.hits],
        "criteria": [
            {
                "key": h.key,
                "title": h.title,
                "points": round(h.weight * 100),
                "max_points": 100,
                "reason": h.reason,
                "tone": (
                    "bad" if h.direction == "fake"
                    else "ok" if h.direction == "real"
                    else "warn"
                ),
            }
            for h in rules.hits
        ],
    }
    return base


def public_payload(result: Any, check_id: str, plan: str) -> dict[str, Any]:
    """Собирает ответ по тарифу. Free не видит разбор — только вердикт."""
    base: dict[str, Any] = {
        "id": check_id,
        "verdict": verdict_of(result),
        "level": result.trust.level.value,
        "level_label": level_label(result.trust.level.value),
        "score": result.trust.score,
        "uncertainty": round(result.trust.uncertainty, 4),
        "plan": plan,
    }
    if plan not in ("pro", "team"):
        base["upgrade_hint"] = (
            "Разбор — почему такой вердикт, источники и что проверить дальше — "
            "доступен по подписке."
        )
        return base

    base["signals"] = [
        {
            "id": s.id,
            "name": s.name,
            "stage": s.stage.value,
            "verdict": s.verdict.value,
            "strength": round(s.strength, 3),
            "reliability": round(s.reliability, 3),
            "summary": s.summary,
            "details": s.details,
        }
        for s in result.signals
    ]
    base["debate"] = {
        "for_real": [{"text": t.text, "signal_ids": t.signal_ids} for t in result.debate.for_real],
        "for_fake": [{"text": t.text, "signal_ids": t.signal_ids} for t in result.debate.for_fake],
    }
    base["provenance"] = [
        {"date": p.date, "where": p.where, "note": p.note} for p in result.provenance
    ]
    base["next_steps"] = [
        {"kind": n.kind.value, "title": n.title, "why": n.why} for n in result.next_steps
    ]
    base["limits"] = result.limits
    base["duration_ms"] = result.duration_ms
    return base


def resolve_user(token: Optional[str]) -> Optional[dict[str, Any]]:
    """Пользователь по X-Token или по корпоративному ключу X-Api-Key."""
    return store.user_by_token(token or "")


def resolve_any_user(
    token: Optional[str], api_key: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Токен браузера или API-ключ корпорации — оба дают доступ к тарифу."""
    user = resolve_user(token)
    if user:
        return user
    return store.user_by_api_key(api_key or "")


# ---------- эндпоинты ----------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": bool(os.environ.get("LLM_API_KEY")),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.post("/api/check-media")
async def check_media(
    file: UploadFile = File(...),
    caption: str = Form(default=""),
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Проверить изображение: метаданные, следы генерации, монтаж, дубликаты.

    Только изображения. Видео пока не поддерживается — честно говорим об
    этом, а не делаем вид, что проверили.
    """
    user = resolve_user(x_token)
    if user is None:
        return fail("UNAUTHORIZED", "нужно войти или зарегистрироваться", 401)

    mime = (file.content_type or "").lower()
    if mime.startswith("video/"):
        return fail(
            "UNSUPPORTED_TYPE",
            "Видео пока не проверяем: поддерживаются изображения PNG, JPEG, WebP",
            415,
        )
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        return fail("UNSUPPORTED_TYPE", "подойдёт PNG, JPEG или WebP", 415)

    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        return fail("TOO_LARGE", "файл больше 15 МБ", 413)
    if not data:
        return fail("BAD_INPUT", "пустой файл", 422)

    # Проверяем настоящий формат, а не только заявленный MIME.
    try:
        from PIL import Image
        import io as _io

        with Image.open(_io.BytesIO(data)) as probe:
            probe.verify()
    except Exception:
        return fail("BAD_INPUT", "файл не открывается как изображение", 415)

    plan = user["plan"]
    if plan == "free" and store.free_quota_left(user["id"]) <= 0:
        return fail(
            "QUOTA_EXCEEDED",
            "бесплатные проверки на сегодня закончились. Подписка снимает лимит.",
            429,
        )

    check_id = f"med_{int(time.time() * 1000) % 10**10:010d}"
    ctx = CheckContext(
        check_id=check_id,
        input=CheckInput(type=InputType.image),
        data=data,
        text=(caption or "").strip(),
    )
    try:
        result = await run_check(ctx, llm=None)
    except Exception as exc:
        return fail("CHECK_FAILED", f"проверка не удалась: {type(exc).__name__}", 502)

    payload = {
        "id": check_id,
        "kind": "image",
        "captcha": None,
        "trust": {
            "score": result.trust.score,
            "uncertainty": round(result.trust.uncertainty, 4),
            "coverage": round(result.trust.coverage, 4),
            "level": result.trust.level.value,
            "level_label": level_label(result.trust.level.value),
            "conflict": result.trust.conflict,
        },
        "signals": [
            {
                "id": s.id,
                "name": s.name,
                "stage": s.stage.value,
                "verdict": s.verdict.value,
                "strength": round(s.strength, 3),
                "reliability": round(s.reliability, 3),
                "summary": s.summary,
                "details": s.details,
            }
            for s in result.signals
        ],
        "limits": result.limits,
        "duration_ms": result.duration_ms,
        "plan": plan,
    }
    payload["quota_left"] = store.free_quota_left(user["id"]) if plan == "free" else None
    return payload


@app.post("/api/check")
async def check_text(
    request: Request,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
) -> Any:
    """Проверить утверждение. Free — вердикт, pro — полный разбор.

    Принимает JSON (текст и/или ссылка) и multipart с картинкой или видео:
    скриншот из мессенджера — самый частый материал, который присылают
    на проверку.
    """
    image: Optional[bytes] = None
    image_name = ""
    video: Optional[bytes] = None
    video_name = ""
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        text = str(form.get("text") or "")
        url = str(form.get("url") or "") or None
        upload = form.get("image")
        if upload is not None and not isinstance(upload, str):
            if (upload.content_type or "") not in IMAGE_TYPES:
                return fail(
                    "BAD_INPUT",
                    "картинка должна быть PNG, JPEG, WebP или GIF",
                    422,
                )
            image = await upload.read()
            image_name = upload.filename or "screenshot"
            if len(image) > MAX_IMAGE_BYTES:
                return fail("BAD_INPUT", "картинка больше 8 МБ", 422)
            if not image:
                return fail("BAD_INPUT", "пустой файл", 422)
        clip = form.get("video")
        if clip is not None and not isinstance(clip, str):
            if (clip.content_type or "") not in VIDEO_TYPES:
                return fail("BAD_INPUT", "видео должно быть MP4 или MOV", 422)
            video = await clip.read()
            video_name = clip.filename or "clip.mp4"
            if len(video) > MAX_VIDEO_BYTES:
                return fail("BAD_INPUT", "видео больше 25 МБ", 422)
            if not video:
                return fail("BAD_INPUT", "пустой файл", 422)
    else:
        try:
            req = CheckRequest(**(await request.json()))
        except Exception:
            return fail("BAD_INPUT", "не разобрал тело запроса", 422)
        text = req.text
        url = req.url

    text = (text or "").strip()
    url = (url or "").strip() or None
    if url and not url.startswith(("http://", "https://")):
        return fail("BAD_INPUT", "ссылка должна начинаться с http:// или https://", 422)

    if not image and not video:
        # Ссылка вместо текста — штатный сценарий: пользователь жмёт
        # «поделиться» в мессенджере и присылает адрес.
        if url and not text:
            # Импорт внутри функции: тесты подменяют lakmus.sources.fetch_page,
            # и при импорте на уровне модуля подмена не срабатывала.
            from lakmus.sources import fetch_page as _fetch

            # В потоке: синхронное скачивание иначе держит event loop до 12 с.
            page = await asyncio.to_thread(_fetch, url) or ""
            text = page.strip()[:MAX_TEXT]
        if len(text) < 8:
            return fail(
                "BAD_INPUT", "слишком короткий текст: нужно хотя бы 8 символов", 422
            )
    if len(text) > MAX_TEXT:
        return fail("BAD_INPUT", f"текст длиннее {MAX_TEXT} символов", 422)

    # Без аккаунта проверок нет: лимиты, тариф и история платежей живут
    # в профиле. Вход — по токену браузера или по API-ключу корпорации;
    # оба привязаны к аккаунту, поэтому лимиты и тариф работают одинаково.
    user = resolve_any_user(x_token, x_api_key)
    if user is None:
        return fail(
            "UNAUTHORIZED",
            "нужно войти или зарегистрироваться — проверка доступна по аккаунту",
            401,
        )
    plan = user["plan"]

    if plan == "free" and store.free_quota_left(user["id"]) <= 0:
        return fail(
            "QUOTA_EXCEEDED",
            "бесплатные проверки на сегодня закончились. Подписка снимает лимит.",
            429,
        )

    check_id = f"chk_{int(time.time() * 1000) % 10**10:010d}"
    try:
        if image or video:
            # Файл: сигналы изображения или видео. Картинка — метаданные,
            # следы генерации, монтаж, контекст публикации. Видео — метаданные
            # контейнера и проверка, не является ли ролик статичной картинкой.
            # Текст подписи идёт в OCR для картинки и остаётся в паспорте.
            kind = InputType.video if video else InputType.image
            ctx = CheckContext(
                check_id=check_id,
                input=CheckInput(
                    type=kind,
                    filename=video_name if video else image_name,
                ),
                data=video or image,
                text=text,
            )
            result = await run_check(ctx)
            payload = media_payload(result, check_id, plan, "video" if video else "image")
            kind_label = "видео" if video else "изображение"
            subject = text or f"[{kind_label}] {video_name or image_name}"

            # Модель переводит отчёт детекторов на человеческий язык.
            # Детекторы решают, что нашли; модель объясняет, что это значит.
            # Без ключа шаг просто пропускается — вердикт остаётся прежним.
            media_note = await explain_media(
                result.signals,
                caption=text,
                kind=kind_label,
                tier=for_plan(plan),
            )
            if media_note:
                payload["explanation"] = media_note["explanation"]
                payload["explanation_source"] = "model"
                if media_note.get("action"):
                    payload["action"] = media_note["action"]
                if plan in ("pro", "team"):
                    payload["model_verdict"] = media_note["verdict"]
            else:
                # Модели нет или она промолчала — объясняем детерминированно.
                # Пустой explanation ломает и интерфейс, и паспорт: человек
                # видит вердикт без единого слова обоснования.
                payload["explanation"] = _detector_explanation(result)
                payload["explanation_source"] = "detectors"

            store.save_check(
                text=subject,
                verdict=payload["verdict"],
                score=result.trust.score,
                uncertainty=result.trust.uncertainty,
                level=result.trust.level.value,
                full=public_payload(result, check_id, "pro"),
                user_id=user["id"],
                check_id=check_id,
            )
        else:
            # Полный пайплайн: извлечение → источники → оценка → валидация цитат.
            # Без LLM_API_KEY работает детерминированно, с ключом включает модель.
            outcome = await run_full_check(
                text, url=_clean_url(url), tier=for_plan(plan)
            )
            payload = outcome_payload(outcome, check_id, plan)
            subject = text or (url or "")
            # В базу всегда кладём полный разбор, независимо от тарифа: он нужен
            # публичному паспорту и повторному открытию по подписке. Тариф
            # ограничивает только ответ API, а не хранение.
            store.save_check(
                text=subject,
                verdict=payload["verdict"],
                score=outcome.score,
                uncertainty=outcome.uncertainty,
                level=outcome.level,
                full=outcome_payload(outcome, check_id, "pro"),
                user_id=user["id"],
                check_id=check_id,
            )
    except Exception as exc:  # пайплайн упал — это не 500 без объяснения
        return fail("CHECK_FAILED", f"проверка не удалась: {type(exc).__name__}", 502)

    # Постоянная ссылка на паспорт: одна на утверждение, живёт после закрытия.
    # Для медиа subject задаётся выше (имя файла или подпись), для текста —
    # сам текст или ссылка. Берём уже вычисленное, а не пересчитываем.
    subject = locals().get("subject") or text
    payload["passport"] = f"/p/{store.text_hash(subject)}"
    payload["quota_left"] = (
        store.free_quota_left(user["id"]) if plan == "free" else None
    )
    return payload


@app.get("/api/check/{check_id}")
async def get_check(
    check_id: str,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Повторно открыть проверку. Разбор отдаётся только по подписке."""
    row = store.get_check(check_id)
    if not row:
        return fail("NOT_FOUND", "проверка не найдена", 404)
    user = resolve_user(x_token)
    plan = user["plan"] if user else "free"
    full = row["full"]
    if plan not in ("pro", "team"):
        return {k: v for k, v in full.items() if k not in (
            "signals", "debate", "provenance", "next_steps", "limits", "duration_ms"
        )} | {"plan": plan, "upgrade_hint": "Разбор доступен по подписке."}
    return full


@app.get("/api/top")
async def top(limit: int = 20, hours: int = 48) -> dict[str, Any]:
    """Топы проверяемых утверждений — та самая вкладка из совета эксперта."""
    limit = max(1, min(50, limit))
    hours = max(1, min(24 * 30, hours))
    items = store.top_items(limit=limit, window_hours=hours)
    for item in items:
        item["level_label"] = level_label(item.get("level") or "fog")
    return {"window_hours": hours, "items": items}


@app.post("/api/top/pin")
async def pin(req: PinRequest) -> Any:
    if not req.text_hash:
        return fail("BAD_INPUT", "нужен text_hash", 422)
    store.pin_top(req.text_hash, req.title)
    return {"ok": True}


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    return store.stats()


@app.get("/api/feed")
async def feed(limit: int = 30, hours: int = 168) -> dict[str, Any]:
    """Лента проверенного: что уже разобрали и с каким итогом.

    Эксперт на встрече: «дайте проверенные новости сегодня, или топ-новостей,
    или топ-запросов». Топы — это рейтинг сомнений, а лента — витрина
    результата: сюда попадает то, что проверили, с вердиктом и источником.
    """
    limit = max(1, min(100, limit))
    hours = max(1, min(24 * 90, hours))
    items = store.feed_items(limit=limit, hours=hours)
    for item in items:
        item["level_label"] = level_label(item.get("level") or "fog")
        item["passport"] = f"/p/{item['text_hash']}"
    verified = [i for i in items if i["level"] in ("confirmed", "likely_real")]
    refuted = [i for i in items if i["level"] == "likely_fake"]
    return {
        "window_hours": hours,
        "items": items,
        "verified": len(verified),
        "refuted": len(refuted),
        "unclear": len(items) - len(verified) - len(refuted),
    }


# ---------- автопилот: что собрал и проверил фоновый контур ----------

@app.get("/api/autopilot/stats")
async def autopilot_stats() -> dict[str, Any]:
    """Сводка автопилота: собрано, размечено, очередь, модель, расход модели.

    Читается только на чтение из базы автопилота (WAL), поэтому не мешает
    его записи. Если автопилот ещё не запускался — `available: false`.
    """
    return autopilot_feed.summary()


@app.get("/api/autopilot/recent")
async def autopilot_recent(limit: int = 30, label: str | None = None) -> dict[str, Any]:
    """Последние материалы, которым автопилот поставил метку.

    `label` — fake | real | uncertain; без него отдаются все размеченные.
    """
    return autopilot_feed.recent(limit=limit, label=label)

# ---------- обратная связь: основа дообучения ----------

@app.post("/api/feedback")
async def create_feedback(
    req: FeedbackRequest,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
) -> Any:
    """Человек сказал, верный ли вердикт. Это корпус для дообучения.

    Модель по одному отзыву не меняется: собираем корпус, разбираем его
    отдельным шагом. Иначе один случайный клик перекосил бы весь сервис.
    """
    row = store.get_check(req.check_id)
    if not row:
        return fail("NOT_FOUND", "проверка не найдена", 404)
    if req.expected and req.expected not in ("правда", "фейк", "непонятно"):
        return fail("BAD_INPUT", "expected: правда, фейк или непонятно", 422)

    user = resolve_any_user(x_token, x_api_key)
    feedback_id = store.save_feedback(
        check_id=req.check_id,
        text=row["text"],
        verdict=row["verdict"],
        level=row["level"],
        verdict_ok=req.verdict_ok,
        expected=req.expected,
        note=req.note,
        user_id=user["id"] if user else None,
    )

    # Ошибка человека — это правило, а не просто запись в журнале.
    # Если он сказал «должно быть фейк/правда», знание уходит в банк с
    # origin=human: следующий такой же текст получит верный ответ сразу,
    # и модель не сможет перетереть эту правку.
    learned = False
    if not req.verdict_ok and req.expected in ("правда", "фейк"):
        learned = _learn_from_feedback(row, req.expected, req.note or "")

    return {
        "id": feedback_id,
        "ok": True,
        "learned": learned,
        "note": (
            "Спасибо — поправка записана в банк знаний, следующая такая "
            "проверка ответит верно."
            if learned else
            "Спасибо — это попадёт в разбор ошибок."
        ),
    }


def _learn_from_feedback(row: Any, expected: str, note: str) -> bool:
    """Записать поправку человека в банк знаний.

    Учимся только тому, что можно проверить: у проверки должен быть
    источник. Без него поправка останется в журнале обратной связи, но в
    банк не пойдёт — иначе банк наполнится неподтверждёнными мнениями.
    """
    try:
        from lakmus import bank as bank_module
        from lakmus import tags as tag_module

        full = row["full"] or {}
        claims = full.get("claims") or []
        # Берём основание из первого утверждения: цитата и ссылка оттуда.
        first = claims[0] if claims else {}
        source_url = str(first.get("source_url") or "")
        if not source_url:
            return False

        text = str(row["text"] or "")
        tag_match = tag_module.classify(text)
        status = "supported" if expected == "правда" else "contradicted"
        bank = bank_module.shared()
        entry = bank.learn(
            text,
            status,
            tag=tag_match.tag.key,
            explanation=(note or "Поправка редакции: вердикт должен быть другим.")[:800],
            quote=str(first.get("quote") or ""),
            source_title=str(first.get("source_title") or ""),
            source_url=source_url,
            origin="human",
        )
        if entry is None:
            return False
        bank.save()
        return True
    except Exception:
        # Банк не критичен: проверка продолжает работать и без него.
        return False


@app.get("/api/feedback/stats")
async def feedback_stats(days: int = 30) -> dict[str, Any]:
    """Точность по отзывам людей: сколько вердиктов признали верными."""
    return store.feedback_stats(days=max(1, min(365, days)))


# ---------- корпоративный доступ по API-ключу ----------

@app.post("/api/keys")
async def create_key(
    req: ApiKeyRequest,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Выпустить API-ключ. Доступно тарифу «Редакция».

    Ключ показывается один раз: в базе лежит только его хеш.
    """
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    if user["plan"] != "team":
        return fail(
            "PLAN_REQUIRED",
            "API-ключи доступны на тарифе «Редакция»",
            402,
        )
    return store.create_api_key(user["id"], req.label)


@app.get("/api/keys")
async def list_keys(
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    return {
        "items": store.list_api_keys(user["id"]),
        "usage": store.api_key_usage(user["id"]),
    }


@app.delete("/api/keys/{key_id}")
async def delete_key(
    key_id: str,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    if not store.revoke_api_key(user["id"], key_id):
        return fail("NOT_FOUND", "ключ не найден или уже отозван", 404)
    return {"ok": True}


@app.post("/api/users")
async def create_user() -> dict[str, Any]:
    """Гостевой пользователь с бесплатным тарифом и своим токеном."""
    return store.create_user("free")


@app.post("/api/auth/register")
async def register(req: RegisterRequest) -> Any:
    """Создать аккаунт. Без аккаунта проверки недоступны."""
    email = (req.email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", email):
        return fail("BAD_INPUT", "нужен корректный email", 422)
    if len(req.password or "") < 8:
        return fail("BAD_INPUT", "пароль короче 8 символов", 422)

    user = store.register_user(email, req.password, req.name)
    if user is None:
        return fail("EMAIL_TAKEN", "аккаунт с таким email уже есть", 409)

    out = store.public_user(user)
    out["token"] = user["token"]
    out["quota_left"] = store.FREE_CHECKS_PER_DAY
    out["quota_per_day"] = store.FREE_CHECKS_PER_DAY
    return out


@app.post("/api/auth/login")
async def login(req: LoginRequest) -> Any:
    user = store.authenticate(req.email, req.password)
    if user is None:
        return fail("BAD_CREDENTIALS", "неверный email или пароль", 401)

    out = store.public_user(user)
    out["token"] = user["token"]
    if user["plan"] == "free":
        out["quota_left"] = store.free_quota_left(user["id"])
        out["quota_per_day"] = store.FREE_CHECKS_PER_DAY
    return out


@app.post("/api/auth/logout")
async def logout(
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Выход: токен отзывается, сессия больше не действует."""
    user = resolve_user(x_token)
    if user is not None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE users SET token = ? WHERE id = ?",
                (f"revoked_{store.secrets.token_hex(8)}", user["id"]),
            )
    return {"ok": True}


@app.get("/api/me")
async def me(x_token: Optional[str] = Header(default=None, alias="X-Token")) -> Any:
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    out = store.public_user(user)
    if user["plan"] == "free":
        out["quota_left"] = store.free_quota_left(user["id"])
        out["quota_per_day"] = store.FREE_CHECKS_PER_DAY
    return out


@app.get("/api/history")
async def history(
    limit: int = 50,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """История проверок пользователя. Главред назвал её базовой вещью."""
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    limit = max(1, min(100, limit))
    items = store.user_checks(user["id"], limit=limit)
    for item in items:
        item["level_label"] = level_label(item.get("level") or "fog")
        item["passport"] = f"/p/{item['text_hash']}"
    return {"items": items}


@app.post("/api/upgrade")
async def upgrade(
    req: UpgradeRequest,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Демо-апгрейд тарифа. В продукте здесь платёжный провайдер."""
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    if not store.upgrade_user(x_token or "", req.plan):
        return fail("BAD_INPUT", "неизвестный тариф", 422)
    return {"ok": True, "plan": req.plan}


@app.get("/p/{hash_value}", response_class=HTMLResponse)
async def passport(hash_value: str, request: Request) -> Any:
    """Публичный «паспорт утверждения» — постоянная ссылка на проверку.

    Отдаётся серверным HTML, чтобы Telegram и WhatsApp показывали превью
    при пересылке. Вход не нужен: карточка должна открываться у любого.
    """
    if not re.fullmatch(r"[0-9a-f]{6,32}", hash_value or ""):
        return fail("NOT_FOUND", "паспорт не найден", 404)

    row = store.get_by_hash(hash_value)
    if not row:
        return fail("NOT_FOUND", "паспорт не найден", 404)

    full = row.get("full") or {}
    canonical = str(request.url).split("?")[0]
    page = passport_html(
        text=row["text"],
        verdict=full.get("verdict", "непонятно"),
        level=full.get("level", row.get("level", "fog")),
        score=int(full.get("score", row.get("score", 50))),
        claims=full.get("claims", []),
        checks_count=int(row.get("checks_count", 1)),
        checked_at=str(row.get("created_at", ""))[:10],
        canonical_url=canonical,
        hash_value=hash_value,
    )
    return HTMLResponse(page)


def _has_quote_material(source_text: str, claim: dict[str, Any]) -> bool:
    """Есть ли в тексте хоть какое-то пересечение с цитатой утверждения."""
    quote = str(claim.get("quote", ""))
    if not quote:
        return True
    words = {w for w in re.findall(r"[а-яa-z]{5,}", quote.lower().replace("ё", "е"))}
    if not words:
        return True
    text = source_text.lower().replace("ё", "е")
    return sum(1 for w in words if w in text) >= max(1, len(words) // 3)


@app.get("/p/{hash_value}/quote/{claim_index}", response_class=HTMLResponse)
async def quote_in_source(hash_value: str, claim_index: int, request: Request) -> Any:
    """Где именно в источнике сказано то, на чём основан вердикт."""
    if not re.fullmatch(r"[0-9a-f]{6,32}", hash_value or ""):
        return fail("NOT_FOUND", "страница не найдена", 404)
    row = store.get_by_hash(hash_value)
    if not row:
        return fail("NOT_FOUND", "проверка не найдена", 404)

    full = row.get("full") or {}
    claims = full.get("claims") or []
    if claim_index < 0 or claim_index >= len(claims):
        return fail("NOT_FOUND", "утверждение не найдено", 404)

    claim = claims[claim_index]
    quote = str(claim.get("quote", ""))
    source_url = str(claim.get("source_url", ""))
    source_title = str(claim.get("source_title", "")) or source_url

    # Источник для показа: сначала пробуем скачать страницу первоисточника.
    # Если там нет нужного места (частая история: ссылка ведёт на раздел,
    # а не на конкретный документ), показываем запись базы фактов — это
    # тоже реальный источник, просто локальный.
    source_text = ""
    if source_url:
        source_text = fetch_page(source_url) or ""
    # Запасной текст — запись базы фактов. Если скачанная страница не
    # содержит нужного места, покажем её вместо свалки навигации.
    fallback_text = (
        str(claim.get("quote", "")) + ". " + str(claim.get("explanation", ""))
    ).strip()
    fallback_title = source_title or "База фактов"

    canonical = str(request.url).split("?")[0]
    page = build_quote_page(
        claim_text=str(claim.get("text", "")),
        quote=quote,
        source_title=source_title,
        source_url=source_url,
        source_text=source_text,
        fallback_text=fallback_text,
        fallback_title=fallback_title,
        verdict_label=level_label(str(full.get("level", "fog"))),
        level=str(full.get("level", "fog")),
        canonical_url=canonical,
    )
    return HTMLResponse(page)


@app.get("/sources", response_class=HTMLResponse)
async def sources(request: Request) -> Any:
    """Открытая витрина базы фактов: что проверяем и по каким источникам."""
    topics = load_topics()
    canonical = str(request.url).split("?")[0]
    return HTMLResponse(sources_page_html(topics, canonical))


@app.get("/api/sources")
async def sources_json() -> dict[str, Any]:
    """Тот же список в JSON — для интерфейса и для проверки."""
    topics = load_topics()
    return {
        "topics": len(topics),
        "facts": sum(len(t.get("facts", [])) for t in topics),
        "items": [
            {
                "topic": t.get("topic", ""),
                "facts": [
                    {
                        "claim": f.get("claim", ""),
                        "verdict": f.get("verdict", ""),
                        "source": f.get("source", ""),
                        "url": f.get("url", ""),
                    }
                    for f in t.get("facts", [])
                ],
            }
            for t in topics
        ],
    }


@app.post("/api/payments")
async def create_payment(
    req: PaymentRequest,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Создать платёж. Возвращает платёж в статусе pending.

    Реального эквайринга в демо нет: оплата подтверждается отдельным
    вызовом. В продукте здесь создаётся платёж у провайдера и клиент
    уходит на его страницу.
    """
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    if req.plan not in store.PLAN_PRICES or req.plan == "free":
        return fail("BAD_INPUT", "неизвестный тариф", 422)
    if req.method not in ("card", "sbp", "invoice"):
        return fail("BAD_INPUT", "способ оплаты: card, sbp или invoice", 422)

    payment = store.create_payment(user["id"], req.plan, req.method)
    if payment is None:
        return fail("BAD_INPUT", "не удалось создать платёж", 422)
    payment["plan_name"] = {"pro": "Разбор", "team": "Редакция"}.get(req.plan, req.plan)
    payment["days"] = store.PLAN_DAYS.get(req.plan, 30)
    return payment


@app.post("/api/payments/confirm")
async def confirm_payment(
    req: PaymentConfirm,
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    """Подтвердить платёж. В продукте вызывается вебхуком провайдера."""
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)

    payment = store.get_payment(req.payment_id)
    if payment is None:
        return fail("NOT_FOUND", "платёж не найден", 404)
    if payment["user_id"] != user["id"]:
        return fail("FORBIDDEN", "чужой платёж", 403)

    done = store.confirm_payment(req.payment_id)
    fresh = store.user_by_token(x_token or "")
    return {
        "payment": done,
        "plan": fresh["plan"] if fresh else None,
        "expires_at": fresh.get("expires_at") if fresh else None,
    }


@app.get("/api/payments")
async def payments(
    x_token: Optional[str] = Header(default=None, alias="X-Token"),
) -> Any:
    user = resolve_user(x_token)
    if not user:
        return fail("UNAUTHORIZED", "нужен токен пользователя", 401)
    return {"items": store.payment_history(user["id"])}


@app.get("/api/pricing")
async def pricing() -> dict[str, Any]:
    """Тарифы. Эксперт просила продумать это подробно — здесь срок,
    скидка за год и честная пометка, что эквайринг в демо не подключён."""
    return {
        "plans": [
            {
                "id": "free",
                "name": "База",
                "price_rub": 0,
                "price_year_rub": 0,
                "limits": f"{store.FREE_CHECKS_PER_DAY} проверок в день",
                "features": [
                    "Вердикт: правда, фейк или непонятно",
                    "Тема, по которой шла проверка",
                    "Паспорт проверки — ссылка для пересылки",
                ],
            },
            {
                "id": "pro",
                "name": "Плюс",
                "price_rub": 300,
                "price_year_rub": 2880,
                "limits": "Без ограничений",
                "features": [
                    "Всё из «Базы»",
                    "Разбор по рубрике: из чего сложился балл",
                    "Цитата из первоисточника с подсветкой",
                    "Что проверить дальше",
                    "Без рекламы",
                ],
                "recommended": True,
            },
            {
                "id": "team",
                "name": "Макс",
                "price_rub": 5000,
                "price_year_rub": 48000,
                "limits": "До 10 сотрудников",
                "features": [
                    "Всё из «Плюса»",
                    "Полный анализ: таблица утверждений и вывод по каждому",
                    "Готовая формулировка для пересылки",
                    "API-ключи для редакции",
                    "Приоритетная поддержка",
                ],
            },
        ],
        "trial": {
            "days": 30,
            "note": "Первый месяц бесплатно, карта не нужна",
        },
        "note": "Сначала репутация, потом подписка: месяц бесплатно — без карты.",
        "demo_warning": (
            "Демонстрационный режим: платёж проходит без реального списания. "
            "В продукте здесь эквайринг."
        ),
    }

# ---------- раздача собранного интерфейса ----------
# В проде nginx обычно отдаёт статику сам, но если фронт собран в
# frontend/dist, FastAPI может отдать его напрямую — удобно для одного
# контейнера без отдельного веб-сервера.

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")
