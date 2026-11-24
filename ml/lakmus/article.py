"""Извлечение статьи из HTML: чистый текст, заголовок и метаданные.

Одна реализация для backend, ML-пайплайна и автопилота. Раньше каждый
модуль вырезал теги по-своему, и в текст попадало всё подряд: меню
(«Открыть меню / Закрыть меню / Разделы»), подвал («В соцсетях»),
боковая колонка «Лента новостей» с чужими заголовками. Такой мусор
тратил токены модели и отравлял обучающий корпус.

Порядок работы:
1. **trafilatura** — основной извлекатель: сам находит основной блок,
   отбрасывает навигацию и комментарии. Флаги подобраны под фактчекинг:
   без комментариев и таблиц, с уклоном в точность.
2. **BeautifulSoup** — запасной путь, если trafilatura недоступна или
   вернула слишком мало. Перед разбором удаляются служебные теги и блоки
   с «говорящими» классами (menu, sidebar, related, comments, share…),
   затем текст берётся из `<article>` / `<main>` / `[itemprop=articleBody]`.
3. **OpenGraph** — последний шанс для мессенджеров и соцсетей: там весь
   смысл в `og:description`.

Заголовок и признаки сатиры выносятся отдельно и **не** попадают в текст
статьи: модель должна видеть только то, что утверждает материал.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from typing import Optional

#: Домены, которые публикуют сатиру осознанно. Совпадает с реестром
#: репутации, но продублирован: извлекатель не должен зависеть от него.
SATIRE_DOMAINS: tuple[str, ...] = (
    "panorama.pub",
    "panorama.ru",
    "fognews.ru",
    "the-onion.com",
    "theonion.com",
    "clickhole.com",
    "babylonbee.com",
    "waterfordwhispersnews.com",
    "irk.ru",
)

#: Фразы на странице, по которым видно, что материал — пародия.
SATIRE_MARKERS: tuple[str, ...] = (
    "опережая события",
    "сатирическ",
    "пародийн",
    "шутлив",
    "не является реальной новостью",
    "вымышленн",
    "выдуманн",
    "ирония",
    "юмористическ",
    "fake news",
    "satire",
    "parody",
    "not real news",
)

#: Теги, которые не несут содержания статьи.
DROP_TAGS: tuple[str, ...] = (
    "script", "style", "noscript", "template", "svg", "iframe", "object",
    "embed", "form", "nav", "header", "footer", "aside", "menu", "dialog",
)

#: Фрагменты имён классов и id служебных блоков.
BOILERPLATE_HINTS: tuple[str, ...] = (
    "menu", "sidebar", "related", "news-feed", "newsfeed", "news-list",
    "comments", "comment", "footer", "header", "share", "social",
    "subscribe", "subscription", "banner", "advert", "promo", "cookie",
    "breadcrumb", "pagination", "pager", "tags", "tag-list", "widget",
    "popup", "modal", "overlay", "toolbar", "navbar", "nav-", "-nav",
    "teaser", "recommend", "trending", "popular",
)

#: Контейнеры, где обычно лежит именно статья — в порядке приоритета.
ARTICLE_SELECTORS: tuple[str, ...] = (
    "article",
    "[itemprop='articleBody']",
    "[itemprop=articleBody]",
    "div.article__body",
    "div.article__text",
    "div.article-body",
    "div.article-text",
    "div[class*='article-body']",
    "div[class*='article__body']",
    "div[class*='article-text']",
    "div[class*='article__text']",
    "div[class*='post-content']",
    "div[class*='entry-content']",
    "div[class*='story-body']",
    "main",
    "[role='main']",
)

MIN_ARTICLE_CHARS = 200
MIN_PARAGRAPH_CHARS = 30


@dataclass
class Article:
    """Результат извлечения: то, что можно отдать модели и показать человеку."""

    title: str = ""
    text: str = ""
    domain: str = ""
    is_satire: bool = False
    satirical_markers: list[str] = field(default_factory=list)
    extractor: str = "none"   # trafilatura | fallback | og | none
    published_at: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def domain_of(url: str) -> str:
    """Хост без префикса www (lstrip тут не годится: съедает буквы)."""
    match = re.match(r"https?://(\[[^\]]+\]|[^/:?#]+)", url or "")
    if not match:
        return ""
    host = match.group(1).lower()
    return host[4:] if host.startswith("www.") else host


def _normalize_ws(text: str) -> str:
    text = html_lib.unescape(text or "")
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ---------------------------------------------------------------------------
# Заголовок и признаки сатиры
# ---------------------------------------------------------------------------

def _meta_content(soup, *, prop: str | None = None, name: str | None = None) -> str:
    if prop:
        node = soup.find("meta", attrs={"property": prop})
    elif name:
        node = soup.find("meta", attrs={"name": name})
    else:
        return ""
    if node is None:
        return ""
    return _normalize_ws(node.get("content") or "")


def extract_title(soup, url: str = "") -> str:
    """Заголовок по приоритету: h1 статьи, og:title, twitter:title, <title>."""
    for selector in ("article h1", "main h1", "h1"):
        node = soup.select_one(selector)
        if node is not None:
            text = _normalize_ws(node.get_text(" ", strip=True))
            if len(text) >= 5:
                return text
    for value in (_meta_content(soup, prop="og:title"),
                  _meta_content(soup, name="twitter:title")):
        if len(value) >= 5:
            return value
    if soup.title and soup.title.string:
        title = _normalize_ws(soup.title.string)
        # «Заголовок — Название сайта»: хвост после тире обычно имя издания
        for sep in (" | ", " — ", " - ", " :: "):
            if sep in title:
                head = title.split(sep)[0].strip()
                if len(head) >= 5:
                    return head
        return title
    return ""


def detect_satire(text: str, title: str, domain: str) -> tuple[bool, list[str]]:
    """Пародия ли это: домен из списка или явный дисклеймер на странице.

    Проверяем только заголовок и метаописание — в самом тексте слово
    «сатира» может встретиться в цитате и дать ложное срабатывание.
    """
    host = domain.lower()
    for known in SATIRE_DOMAINS:
        if host == known or host.endswith("." + known):
            return True, [f"домен {known}"]

    haystack = f"{title} {text}".lower().replace("ё", "е")
    found = [marker for marker in SATIRE_MARKERS if marker in haystack]
    # одного совпадения мало: «ирония» встречается в обычных статьях.
    strong = [m for m in found if m not in ("ирония", "юмористическ", "satire", "parody")]
    if strong or len(found) >= 2:
        return True, found
    return False, []


def _published_at(soup) -> Optional[str]:
    for prop in ("article:published_time", "og:published_time"):
        value = _meta_content(soup, prop=prop)
        if value:
            return value
    for name in ("date", "publish-date", "pubdate", "article:published_time"):
        value = _meta_content(soup, name=name)
        if value:
            return value
    node = soup.find("time")
    if node is not None and node.get("datetime"):
        return str(node.get("datetime")).strip()
    return None


# ---------------------------------------------------------------------------
# Основной текст
# ---------------------------------------------------------------------------

def _looks_boilerplate(value: str) -> bool:
    low = (value or "").lower()
    return any(hint in low for hint in BOILERPLATE_HINTS)


def _drop_boilerplate(soup) -> None:
    """Удаляет служебные теги и блоки по «говорящим» классам и id."""
    for tag in soup(list(DROP_TAGS)):
        tag.decompose()
    for tag in soup.find_all(True):
        # тег мог исчезнуть вместе с родителем
        if tag.parent is None and tag.name != "[document]":
            continue
        classes = " ".join(tag.get("class") or []).lower()
        node_id = (tag.get("id") or "").lower()
        role = (tag.get("role") or "").lower()
        if role in ("navigation", "banner", "contentinfo", "complementary"):
            tag.decompose()
            continue
        if _looks_boilerplate(classes) or _looks_boilerplate(node_id):
            tag.decompose()


def _paragraphs(node) -> list[str]:
    out: list[str] = []
    for p in node.find_all(["p", "li"]):
        text = _normalize_ws(p.get_text(" ", strip=True))
        if len(text) >= MIN_PARAGRAPH_CHARS:
            out.append(text)
    return out


def _fallback_text(page: str) -> str:
    """BeautifulSoup-извлечение: статья целиком, без меню и ленты новостей."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    soup = BeautifulSoup(page, "html.parser")
    _drop_boilerplate(soup)

    best = ""
    for selector in ARTICLE_SELECTORS:
        try:
            node = soup.select_one(selector)
        except Exception:  # кривой селектор не должен ломать извлечение
            continue
        if node is None:
            continue
        text = "\n".join(_paragraphs(node))
        if len(text) > len(best):
            best = text

    if len(best) < MIN_ARTICLE_CHARS:
        candidate = "\n".join(_paragraphs(soup))
        if len(candidate) > len(best):
            best = candidate
    return best


def _trafilatura_text(page: str, url: str) -> tuple[str, str]:
    """(текст, заголовок) от trafilatura. Пустые строки, если недоступна."""
    try:
        import trafilatura
    except ImportError:
        return "", ""
    try:
        text = trafilatura.extract(
            page,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        ) or ""
    except Exception:
        return "", ""
    return _normalize_ws(text), ""


def _og_text(page: str) -> str:
    """Описание из OpenGraph — для страниц, где текста нет вовсе."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    soup = BeautifulSoup(page, "html.parser")
    for value in (_meta_content(soup, prop="og:description"),
                  _meta_content(soup, name="description"),
                  _meta_content(soup, name="twitter:description")):
        if len(value) >= 80:
            return value
    return ""


def extract_article(page: str, url: str = "", *, limit: int = 0) -> Article:
    """Извлечь статью из HTML. Никогда не бросает.

    `limit` обрезает текст по границе предложения (0 — без обрезки).
    """
    if not page or not page.strip():
        return Article(domain=domain_of(url))

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page, "html.parser")
    except ImportError:
        soup = None

    title = extract_title(soup, url) if soup is not None else ""
    published = _published_at(soup) if soup is not None else None

    text, extractor = "", "none"
    candidate, _ = _trafilatura_text(page, url)
    if len(candidate) >= MIN_ARTICLE_CHARS:
        text, extractor = candidate, "trafilatura"

    if len(text) < MIN_ARTICLE_CHARS:
        fallback = _fallback_text(page)
        if len(fallback) > len(text):
            text, extractor = fallback, "fallback"

    if len(text) < MIN_ARTICLE_CHARS:
        og = _og_text(page)
        if len(og) > len(text):
            text, extractor = og, "og"

    text = _normalize_ws(text)
    if not title and text:
        # без заголовка страница бесполезна в истории: берём первое предложение
        title = text.split(". ")[0][:160]

    is_satire, markers = detect_satire(title, text, domain_of(url))

    if limit and len(text) > limit:
        cut = text[:limit]
        for sep in (". ", "! ", "? ", "\n"):
            pos = cut.rfind(sep)
            if pos > limit * 0.6:
                cut = cut[:pos + 1]
                break
        text = cut.strip()

    return Article(
        title=title,
        text=text,
        domain=domain_of(url),
        is_satire=is_satire,
        satirical_markers=markers,
        extractor=extractor,
        published_at=published,
    )
