"""Компонент 1 — Collector: фоновый обход источников.

Возможности:
- RSS/Atom, HTML-ленты по CSS-селекторам, JSON API, Telegram-веб (t.me/s/…);
- Playwright для страниц, которые рендерятся JavaScript'ом (`requires_js`);
- очистка текста, дедупликация (точная по content_hash + near-dup по SimHash);
- rate-limit на домен, ретраи с экспоненциальной паузой, robots.txt;
- «предохранитель»: источник с повторяющимися ошибками временно пропускается;
- сохранение сырых данных в SQLite.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .db import Database
from .feeds import RateLimiter, SourceSpec, pick_path, resolve_link
from .logsetup import get_logger
from .robots import RobotsCache
from .settings import CollectorConfig
from .textnorm import (content_hash, detect_lang, domain_trusted, extract_article,
                       normalize_ws, simhash, simhash_to_int, strip_tags, trim,
                       url_hash)

log = get_logger("collector")


@dataclass
class RawItem:
    """Нормализованный элемент до записи в БД."""
    source_id: str
    kind: str
    title: str
    text: str
    url: str = ""
    published_at: str | None = None
    lang: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceReport:
    source_id: str
    ok: bool
    seen: int = 0
    inserted: int = 0
    duplicates: int = 0
    skipped: int = 0
    http_status: int | None = None
    error: str | None = None
    duration_ms: int = 0


class Collector:
    def __init__(self, db: Database, cfg: CollectorConfig, sources):
        self.db = db
        self.cfg = cfg
        self.sources = [s for s in sources if s.enabled]
        self.limiter = RateLimiter()
        self.robots = RobotsCache(user_agent=cfg.user_agent)
        self._client: httpx.AsyncClient | None = None
        self._playwright = None
        self._browser = None

    # -- HTTP ------------------------------------------------------------
    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.cfg.request_timeout_s),
                headers={
                    "User-Agent": self.cfg.user_agent,
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.6",
                },
                follow_redirects=True,
                limits=httpx.Limits(max_connections=max(4, self.cfg.max_concurrency * 2),
                                    max_keepalive_connections=4),
            )
        return self._client

    async def _fetch_text(self, url: str, spec: SourceSpec,
                          headers: dict[str, str] | None = None) -> tuple[int, str]:
        await self.limiter.wait(spec.domain, spec.rate_limit_s)
        if self.cfg.respect_robots and not self.robots.allowed(url):
            raise PermissionError(f"robots.txt запрещает {url}")

        client = await self._http()
        merged = {**spec.headers, **(headers or {})}
        last_exc: Exception | None = None
        for attempt in range(self.cfg.retries):
            try:
                resp = await client.get(url, headers=merged or None)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.status_code, resp.text
            except (httpx.HTTPError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and \
                        400 <= exc.response.status_code < 500 and \
                        exc.response.status_code != 429:
                    break
                if attempt < self.cfg.retries - 1:
                    delay = self.cfg.backoff_s * (2 ** attempt) + random.uniform(0, 0.5)
                    log.debug("retry %s через %.1f с: %s", url, delay, exc)
                    await asyncio.sleep(delay)
        raise last_exc or RuntimeError(f"не удалось получить {url}")

    async def _playwright_html(self, url: str) -> str:
        from playwright.async_api import async_playwright

        if self.cfg.respect_robots and not self.robots.allowed(url):
            raise PermissionError(f"robots.txt запрещает {url}")
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        page = await self._browser.new_page(user_agent=self.cfg.user_agent)
        try:
            await page.goto(url, timeout=int(self.cfg.request_timeout_s * 1000),
                            wait_until="domcontentloaded")
            await page.wait_for_timeout(1200)
            return await page.content()
        finally:
            await page.close()

    # -- Парсеры источников ---------------------------------------------
    async def _fetch_rss(self, spec: SourceSpec) -> list[RawItem]:
        _, text = await self._fetch_text(spec.url, spec)
        return parse_rss(text, spec)

    async def _fetch_html(self, spec: SourceSpec) -> list[RawItem]:
        if spec.requires_js:
            html_text = await self._playwright_html(spec.url)
        else:
            _, html_text = await self._fetch_text(spec.url, spec)
        return parse_html_listing(html_text, spec, spec.url)

    async def _fetch_telegram(self, spec: SourceSpec) -> list[RawItem]:
        url = spec.url if "/s/" in spec.url else \
            f"https://t.me/s/{spec.url.rstrip('/').split('/')[-1]}"
        _, html_text = await self._fetch_text(url, spec)
        return parse_telegram(html_text, spec, url)

    async def _fetch_api(self, spec: SourceSpec) -> list[RawItem]:
        _, text = await self._fetch_text(spec.url, spec,
                                         headers={"Accept": "application/json"})
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ответ не JSON: {exc}") from exc
        return parse_api(data, spec)

    # -- Дочитывание статьи ---------------------------------------------
    async def _enrich(self, item: RawItem, spec: SourceSpec) -> RawItem:
        """Если текста мало — дочитываем страницу и достаём основной текст.

        Часть площадок отдаёт ленту, но блокирует статьи для простых
        HTTP-клиентов (403). Для них поднимаем Playwright — он выглядит
        как обычный браузер.
        """
        if not self.cfg.fetch_article_text or not item.url or len(item.text) >= 600:
            return item
        if not domain_trusted(item.url, [spec.domain]):
            return item
        try:
            if spec.requires_js:
                html_text = await self._playwright_html(item.url)
            else:
                try:
                    _, html_text = await self._fetch_text(item.url, spec)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in (401, 403, 429):
                        raise
                    item.meta["article_via_js"] = True
                    html_text = await self._playwright_html(item.url)
            article = extract_article(html_text)
            if len(article) > len(item.text):
                item.text = normalize_ws(f"{item.title}. {article}" if item.title else article)
                item.meta["article_fetched"] = True
        except (httpx.HTTPError, PermissionError, OSError, asyncio.TimeoutError,
                RuntimeError, ImportError) as exc:
            item.meta["article_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return item

    # -- Запись в БД -----------------------------------------------------
    def _store(self, item: RawItem, spec: SourceSpec) -> str:
        text = normalize_ws(item.text)
        if len(text) < self.cfg.min_text_chars:
            return "skipped"

        c_hash = content_hash(text)
        if self.db.content_hash_exists(c_hash):
            return "duplicate"

        sh_int = simhash_to_int(simhash(text))
        if self.db.near_duplicate_ids(sh_int, self.cfg.near_dup_hamming,
                                      self.cfg.near_dup_window_days):
            return "duplicate"

        item_id = self.db.insert_item({
            "source_id": spec.id,
            "source_name": spec.name,
            "kind": item.kind,
            "url": item.url,
            "url_hash": url_hash(item.url) if item.url else c_hash,
            "content_hash": c_hash,
            "simhash": sh_int,
            "title": trim(normalize_ws(item.title), 300),
            "text": text,
            "published_at": item.published_at,
            "fetched_at": time.time(),
            "lang": item.lang or detect_lang(text),
            "role": spec.role,
            "meta": {**item.meta, "source_reliability": spec.reliability,
                     "role": spec.role},
        })
        return "inserted" if item_id else "duplicate"

    # -- Основной цикл по одному источнику -------------------------------
    async def collect_source(self, spec: SourceSpec) -> SourceReport:
        started = time.monotonic()
        state = self.db.source_state(spec.id)
        if state and int(state["failures"] or 0) >= 3 and state["last_error"]:
            if time.time() - float(state["last_ok_at"] or 0) < self.cfg.failure_cooldown_s:
                log.warning("источник %s на паузе после ошибок (%s)",
                            spec.id, str(state["last_error"])[:80])
                return SourceReport(spec.id, ok=False, error="cooldown")

        parser = {
            "rss": self._fetch_rss,
            "html": self._fetch_html,
            "telegram": self._fetch_telegram,
            "api": self._fetch_api,
        }[spec.kind]

        report = SourceReport(spec.id, ok=False)
        try:
            items = await parser(spec)
            report.seen = len(items)
            limit = spec.max_items or self.cfg.max_items_per_source
            enriched = 0
            for item in items[:limit]:
                if (self.cfg.fetch_article_text and len(item.text) < 600
                        and enriched < self.cfg.article_fetch_limit_per_source):
                    await self._enrich(item, spec)
                    enriched += 1
                outcome = self._store(item, spec)
                if outcome == "inserted":
                    report.inserted += 1
                elif outcome == "duplicate":
                    report.duplicates += 1
                else:
                    report.skipped += 1
            report.ok = True
        except Exception as exc:  # источник не должен ронять цикл
            report.error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("источник %s: ошибка — %s", spec.id, report.error)

        report.duration_ms = int((time.monotonic() - started) * 1000)
        self.db.record_fetch(spec.id, report.ok, report.http_status, report.seen,
                             report.inserted, report.error, report.duration_ms)
        if report.ok:
            log.info("источник %-14s ok: новых %d, дублей %d, пропущено %d (%.0f мс)",
                     spec.id, report.inserted, report.duplicates, report.skipped,
                     report.duration_ms)
        return report

    async def collect_once(self) -> list[SourceReport]:
        sem = asyncio.Semaphore(max(1, self.cfg.max_concurrency))

        async def guarded(spec: SourceSpec) -> SourceReport:
            async with sem:
                return await self.collect_source(spec)

        if not self.sources:
            log.warning("нет включённых источников — добавьте их в config.json")
            return []
        return await asyncio.gather(*(guarded(s) for s in self.sources))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


# ---------------------------------------------------------------------------
# Чистые парсеры (тестируются без сети)
# ---------------------------------------------------------------------------

def parse_rss(xml_text: str, spec: SourceSpec) -> list[RawItem]:
    """RSS/Atom -> элементы. Разбирает и битые ленты."""
    import xml.etree.ElementTree as ET

    items: list[RawItem] = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        for block in re.findall(r"(?is)<item>(.*?)</item>", xml_text):
            items.append(RawItem(
                spec.id, "rss", strip_tags(_tag_value(block, "title")),
                strip_tags(_tag_value(block, "description")),
                resolve_link(spec.url, _tag_value(block, "link")), None, spec.lang))
        return items

    ns = {"atom": "http://www.w3.org/2005/Atom",
          "dc": "http://purl.org/dc/elements/1.1/",
          "content": "http://purl.org/rss/1.0/modules/content/"}
    entries = root.findall(".//item")
    is_atom = False
    if not entries:
        entries = root.findall(".//atom:entry", ns)
        is_atom = True

    for entry in entries:
        if is_atom:
            title = entry.findtext("atom:title", default="", namespaces=ns)
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            desc = (entry.findtext("atom:summary", default="", namespaces=ns) or
                    entry.findtext("atom:content", default="", namespaces=ns))
            date = entry.findtext("atom:updated", default="", namespaces=ns)
        else:
            title = entry.findtext("title", default="")
            link = entry.findtext("link", default="")
            desc = (entry.findtext("content:encoded", default="", namespaces=ns) or
                    entry.findtext("description", default=""))
            date = (entry.findtext("pubDate", default="") or
                    entry.findtext("dc:date", default="", namespaces=ns))

        title_clean = strip_tags(title)
        text_clean = strip_tags(desc)
        if len(text_clean) < 60 and title_clean:
            text_clean = title_clean
        if len(title_clean) < 5 and len(text_clean) < 20:
            continue
        items.append(RawItem(
            source_id=spec.id, kind="rss", title=title_clean, text=text_clean,
            url=resolve_link(spec.url, link or ""),
            published_at=normalize_date(date), lang=spec.lang))
    return items


def _tag_value(block: str, tag: str) -> str:
    m = re.search(rf"(?is)<{tag}[^>]*>(.*?)</{tag}>", block)
    if not m:
        return ""
    value = m.group(1)
    cdata = re.search(r"(?s)^\s*<!\[CDATA\[(.*?)\]\]>\s*$", value)
    return cdata.group(1) if cdata else value


def normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt.isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            from datetime import datetime

            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return raw[:40]


def parse_html_listing(html_text: str, spec: SourceSpec, base_url: str) -> list[RawItem]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("для kind=html нужен beautifulsoup4") from exc

    soup = BeautifulSoup(html_text, "html.parser")
    items: list[RawItem] = []
    for node in soup.select(spec.item_selector or ""):
        title = _select_text(node, spec.title_selector) or node.get_text(" ", strip=True)
        link_node = node.select_one(spec.link_selector) if spec.link_selector else node
        href = ""
        if link_node is not None:
            href = link_node.get("href", "") if link_node.name == "a" else ""
            if not href:
                anchor = link_node.find("a") if hasattr(link_node, "find") else None
                href = anchor.get("href", "") if anchor else ""
        text = _select_text(node, spec.text_selector)
        date = _select_text(node, spec.date_selector)
        if not title and not text:
            continue
        items.append(RawItem(spec.id, "html", strip_tags(title),
                             strip_tags(text or title),
                             resolve_link(base_url, href),
                             normalize_date(date), spec.lang))
    return items


def _select_text(node, selector: str | None) -> str:
    if not selector:
        return ""
    found = node.select_one(selector)
    return found.get_text(" ", strip=True) if found else ""


def parse_telegram(html_text: str, spec: SourceSpec, base_url: str) -> list[RawItem]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("для kind=telegram нужен beautifulsoup4") from exc

    soup = BeautifulSoup(html_text, "html.parser")
    channel = base_url.rstrip("/").split("/")[-1]
    items: list[RawItem] = []
    for msg in soup.select("div.tgme_widget_message"):
        body = msg.select_one("div.tgme_widget_message_text")
        if body is None:
            continue
        text = normalize_ws(body.get_text("\n", strip=True))
        if len(text) < 40:
            continue
        time_el = msg.select_one("time")
        published = time_el.get("datetime") if time_el else None
        msg_id = msg.get("data-post", "")
        url = f"https://t.me/{msg_id}" if msg_id else base_url
        items.append(RawItem(spec.id, "telegram", "", text, url,
                             normalize_date(published), spec.lang or "ru",
                             meta={"channel": channel}))
    return items


def parse_api(data: Any, spec: SourceSpec) -> list[RawItem]:
    rows = pick_path(data, spec.items_path or "")
    if not isinstance(rows, list):
        return []
    mapping = {"title": "title", "text": "text", "url": "url",
               "published_at": "published_at", **spec.field_map}
    items: list[RawItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = strip_tags(str(pick_path(row, mapping["title"]) or ""))
        text = strip_tags(str(pick_path(row, mapping["text"]) or ""))
        url = resolve_link(spec.url, str(pick_path(row, mapping["url"]) or ""))
        published = normalize_date(str(pick_path(row, mapping["published_at"]) or ""))
        if not title and not text:
            continue
        items.append(RawItem(spec.id, "api", title, text or title, url,
                             published, spec.lang))
    return items
