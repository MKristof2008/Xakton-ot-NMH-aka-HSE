"""Источники контента: RSS/Atom, страницы-ленты (CSS-селекторы), Telegram-веб.

Каждый источник описывается словарём в `autopilot/config.json`:

    {"id": "tass", "kind": "rss", "name": "ТАСС",
     "url": "https://tass.ru/rss/v2.xml", "reliability": 0.9}

Виды (`kind`):
- `rss`    — обычная RSS/Atom-лента (feedparser);
- `html`   — страница со списком материалов; нужны `item_selector` и,
             опционально, `title_selector`, `link_selector`, `date_selector`,
             `text_selector` (селекторы CSS для BeautifulSoup);
- `telegram` — публичный веб-превью канала `https://t.me/s/<channel>`
             (без токена бота, читаем HTML, Playwright не нужен);
- `api`    — JSON API; нужны `items_path` (точечный путь до списка, например
             `data.articles`) и `field_map` вида
             {"title": "title", "text": "description", "url": "url",
              "published_at": "publishedAt"}.

Общие поля: `id`, `name`, `kind`, `url`, `enabled`, `reliability` (0–1),
`lang`, `headers` (dict), `rate_limit_s` (минимальный интервал между запросами
к домену), `requires_js` (true -> тянем страницу через Playwright).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

KNOWN_KINDS = {"rss", "html", "telegram", "api"}


@dataclass
class SourceSpec:
    id: str
    name: str
    kind: str
    url: str
    enabled: bool = True
    reliability: float = 0.5
    role: str = "news"           # news | factcheck (карточки опровержений)
    lang: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    rate_limit_s: float = 1.0
    requires_js: bool = False
    # html / api
    item_selector: str | None = None
    title_selector: str | None = None
    link_selector: str | None = None
    date_selector: str | None = None
    text_selector: str | None = None
    items_path: str | None = None
    field_map: dict[str, str] = field(default_factory=dict)
    max_items: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_KINDS:
            raise ValueError(f"неизвестный kind={self.kind!r} (ожидается {sorted(KNOWN_KINDS)})")
        if not self.id or not self.url:
            raise ValueError("у источника обязательны id и url")
        if self.role not in ("news", "factcheck"):
            raise ValueError(f"role должен быть news или factcheck, получено {self.role!r}")
        if self.kind == "html" and not self.item_selector:
            raise ValueError(f"{self.id}: для kind=html нужен item_selector")
        if self.kind == "api" and not self.items_path:
            raise ValueError(f"{self.id}: для kind=api нужен items_path")
        self.reliability = min(max(float(self.reliability), 0.0), 1.0)

    @property
    def domain(self) -> str:
        return urlsplit(self.url).netloc.lower()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def resolve_link(base: str, href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        return f"https:{href}"
    return urljoin(base, href.strip())


def pick_path(data: Any, dotted: str) -> Any:
    """Достаёт значение по пути `a.b.0.c`; None, если путь не существует."""
    cur = data
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class RateLimiter:
    """Не чаще одного запроса к домену за `interval` секунд."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    async def wait(self, domain: str, interval: float) -> None:
        import asyncio

        now = time.monotonic()
        last = self._last.get(domain)
        if last is not None:
            delay = interval - (now - last)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last[domain] = time.monotonic()
