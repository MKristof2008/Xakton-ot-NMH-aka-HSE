"""Нормализация текста, хеши, simhash, извлечение статьи из HTML."""
from __future__ import annotations

import hashlib
import html as html_lib
import re
import unicodedata
from typing import Iterable
from urllib.parse import urlsplit

_TAG_RE = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(
    r"(?is)<(script|style|noscript|template|svg|iframe|form|nav|footer|header)[^>]*>.*?</\1>")
_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁ]+")
_CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)

_BOILERPLATE = (
    "подписывайтесь", "читайте также", "реклама", "cookie", "куки",
    "все права защищены", "при использовании материалов", "источник:",
    "фото:", "видео:", "теги:", "поделиться", "комментарии",
)

MIN_ARTICLE_CHARS = 200


def strip_tags(raw: str) -> str:
    """HTML -> текст (без bs4, быстро и предсказуемо)."""
    if not raw:
        return ""
    text = _COMMENT_RE.sub(" ", raw)
    text = _DROP_RE.sub(" ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6])>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return normalize_ws(text)


def normalize_ws(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def normalize_for_hash(text: str) -> str:
    text = normalize_ws(text).lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return _WS_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()[:32]


_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
             "utm_content", "from", "rss", "ref", "src", "yclid", "gclid"}


def canonical_url(url: str) -> str:
    """Без якоря, трекинг-параметров и хвостового слэша."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    query = "&".join(
        chunk for chunk in parts.query.split("&")
        if chunk and chunk.split("=", 1)[0].lower() not in _TRACKING)
    path = parts.path.rstrip("/") or "/"
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}" + \
        (f"?{query}" if query else "")


def url_hash(url: str) -> str:
    """Хеш канонического URL: якорь и трекинг-параметры не влияют."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    query = "&".join(
        chunk for chunk in parts.query.split("&")
        if chunk and chunk.split("=", 1)[0].lower() not in _TRACKING)
    path = parts.path.rstrip("/") or "/"
    canonical = f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"
    if query:
        canonical += f"?{query}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2]


def simhash_features(text: str) -> list[str]:
    """Признаки для SimHash: слова (вес 3) и пары слов (вес 1).

    Слова дают устойчивость к правкам и перестановкам, пары добавляют
    чувствительность к порядку и связкам. Проверено на перепечатках:
    одна правка в тексте — 8 бит из 64, другой текст — 36.
    """
    toks = tokens(text)
    if not toks:
        return []
    pairs = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks * 3 + pairs


def simhash(text: str, bits: int = 64) -> str:
    """64-битный SimHash в hex. Пустая строка -> "0"*16."""
    features = simhash_features(text)
    if not features:
        return "0" * (bits // 4)
    weights = [0] * bits
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for i in range(bits):
            weights[i] += 1 if (value >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if weights[i] > 0:
            out |= 1 << i
    return f"{out:0{bits // 4}x}"


def simhash_to_int(hex_value: str) -> int | None:
    """Hex -> знаковое 64-битное число (SQLite INTEGER знаковый)."""
    if not hex_value:
        return None
    value = int(hex_value, 16)
    return value - (1 << 64) if value >= (1 << 63) else value


def simhash_from_int(value: int | None) -> str:
    if value is None:
        return ""
    return f"{int(value) & 0xFFFFFFFFFFFFFFFF:016x}"


def hamming(a: str, b: str) -> int:
    try:
        return bin((int(a, 16) ^ int(b, 16)) & 0xFFFFFFFFFFFFFFFF).count("1")
    except (ValueError, TypeError):
        return 64


def detect_lang(text: str) -> str:
    letters = [ch for ch in text[:3000] if ch.isalpha()]
    if not letters:
        return "other"
    ratio = len(_CYR_RE.findall(text[:3000])) / len(letters)
    if ratio >= 0.45:
        return "ru"
    if ratio <= 0.05:
        return "en"
    return "other"


def looks_like_boilerplate(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _BOILERPLATE)


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "\n"):
        pos = cut.rfind(sep)
        if pos > limit * 0.6:
            return cut[:pos + 1].strip()
    return cut.strip()


def host_of(url: str) -> str:
    if not url:
        return ""
    try:
        return urlsplit(url).netloc.lower().lstrip("www.")
    except ValueError:
        return ""


def domain_trusted(url: str, trusted_domains: Iterable[str]) -> bool:
    host = host_of(url)
    if not host:
        return False
    for trusted in trusted_domains:
        trusted = trusted.lower().lstrip("www.")
        if host == trusted or host.endswith("." + trusted):
            return True
    return False


def extract_article(html_text: str) -> str:
    """Достаёт основной текст статьи из HTML.

    Делегирует общему извлекателю `lakmus.article`: один код на backend,
    ML-пайплайн и автопилот. Раньше здесь был свой разбор, и в текст
    попадали меню, подвал и чужая лента новостей из боковой колонки.
    """
    try:
        from lakmus.article import extract_article as _extract

        return _extract(html_text).text
    except Exception:  # lakmus недоступен — грубая очистка, но не падение
        return strip_tags(html_text)


def extract_article_full(html_text: str, url: str = ""):
    """Статья целиком: заголовок, чистый текст, признаки сатиры."""
    try:
        from lakmus.article import extract_article as _extract

        return _extract(html_text, url)
    except Exception:
        return None
