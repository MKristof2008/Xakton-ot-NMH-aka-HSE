"""Провайдеры источников для проверки утверждений.

Пайплайн умеет оценивать утверждения только по текстам источников.
Здесь — откуда эти тексты берутся. Три провайдера, все с общим
интерфейсом:

  LocalKbProvider    — курируемая база фактов (офлайн, всегда доступна);
  CatalogProvider    — каталог доверенных сайтов, страницы скачиваются;
  NullProvider       — заглушка, когда источников нет вовсе.

Провайдер не решает, правда это или нет. Он только приносит материал.
Решение принимает пайплайн, а цитаты проверяет детерминированный код.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

if TYPE_CHECKING:  # только для аннотаций, без импорта на старте
    from .article import Article

#: Домены, которым доверяем как первоисточникам. Порядок важен: сверху те,
#: чьё слово весомее.
TRUSTED_DOMAINS: tuple[str, ...] = (
    "cbr.ru",
    "government.ru",
    "publication.pravo.gov.ru",
    "consultant.ru",
    "pravo.gov.ru",
    "nalog.gov.ru",
    "rosstat.gov.ru",
    "obrnadzor.gov.ru",
    "minzdrav.gov.ru",
    "mintrud.gov.ru",
    "tass.ru",
    "rbc.ru",
    "interfax.ru",
    "ria.ru",
    "kommersant.ru",
    "vedomosti.ru",
    "forbes.ru",
)

MAX_PAGE_BYTES = 2_000_000
MAX_CONTENT_CHARS = 4000
FETCH_TIMEOUT = 12

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


@dataclass
class FetchedSource:
    """Источник, который провайдер отдал пайплайну."""

    id: str
    title: str
    url: str
    content: str
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    domain: str = ""
    #: Репутация домена, заполняется пайплайном (trusted/satire/untrusted/unknown)
    reputation: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "domain": self.domain,
            "reputation": self.reputation,
        }


class SourceProvider(Protocol):
    """Интерфейс провайдера источников."""

    name: str

    async def find(self, query: str, limit: int = 3) -> Sequence[FetchedSource]:
        ...


# ---------------------------------------------------------------- утилиты

def _strip_html(raw: str) -> str:
    """Грубое извлечение текста: без зависимостей и без исполнения скриптов."""
    text = re.sub(r"(?is)<(script|style|noscript|svg|iframe)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def domain_of(url: str) -> str:
    """Хост без префикса www.

    Раньше стоял lstrip("www.") — это набор символов, а не префикс:
    «w3.org» превращался в «3.org», а «wow.ru» в «ow.ru». Порог доверия
    от этого не ломался, но сравнение доменов врало.
    """
    # Скобки IPv6 берём в группу отдельно: [::1] иначе не разберётся.
    match = re.match(r"https?://(\[[^\]]+\]|[^/:?#]+)", url or "")
    if not match:
        return ""
    host = match.group(1).lower()
    return host[4:] if host.startswith("www.") else host


def is_trusted(url: str) -> bool:
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def _is_private_host(url: str) -> bool:
    """Защита от SSRF: не даём скачивать локальные адреса.

    Пользователь может прислать ссылку, и без этой проверки бэкенд пойдёт
    по ней сам — в том числе на localhost, в облачные метаданные или в
    админ-панель роутера.

    Проверяем не только dotted-запись: 127.0.0.1 записывают ещё как
    0x7f000001, 2130706433 и [::1]. Раньше такие адреса проходили насквозь.
    """
    host = domain_of(url)
    if not host:
        return True
    # IPv6 в скобках: [::1]:8000
    bare = host.strip("[]")
    if bare in ("localhost", "::1", "0.0.0.0", "::", "metadata.google.internal"):
        return True
    if bare.startswith("localhost.") or bare.endswith(".localhost"):
        return True

    ip = _parse_ipv4(bare)
    if ip is None:
        return False  # обычный домен — решает DNS, см. _resolve_is_private
    a, b, c, _ = ip
    if a == 10 or a == 127 or a == 0:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 169 and b == 254:  # link-local, облачные метаданные
        return True
    if a == 100 and 64 <= b <= 127:  # CGNAT
        return True
    return False


def _parse_ipv4(host: str) -> Optional[tuple[int, int, int, int]]:
    """Разобрать IPv4 в любой записи: 127.0.0.1, 0x7f000001, 2130706433.

    None — если это не IPv4-адрес (тогда это домен).
    """
    try:
        if re.fullmatch(r"\d{1,10}", host):
            n = int(host)
            if n > 0xFFFFFFFF:
                return None
            return (n >> 24, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)
        if re.fullmatch(r"0[xX][0-9a-fA-F]{1,8}", host):
            n = int(host, 16)
            return (n >> 24, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)
        parts = host.split(".")
        if len(parts) != 4:
            return None
        nums: list[int] = []
        for part in parts:
            if re.fullmatch(r"0[xX][0-9a-fA-F]{1,2}", part):
                nums.append(int(part, 16))
            elif re.fullmatch(r"0[0-7]{1,3}", part):
                nums.append(int(part, 8))
            elif re.fullmatch(r"\d{1,3}", part):
                nums.append(int(part))
            else:
                return None
        if any(n > 255 for n in nums):
            return None
        return (nums[0], nums[1], nums[2], nums[3])
    except ValueError:
        return None


#: Домены мессенджеров: страница там не отдаёт текст без JS, но OpenGraph
#: содержит заголовок и описание — этого хватает, чтобы понять, о чём новость.
_MESSENGER_HOSTS = ("t.me", "telegram.me", "max.ru", "vk.me", "wa.me", "whatsapp.com")


def _og_preview(page: str) -> Optional[str]:
    """Заголовок и описание из OpenGraph — для страниц без читаемого текста."""
    def pick(prop: str) -> str:
        m = re.search(
            rf'<meta[^>]+(?:property|name)=["\']{prop}["\'][^>]+content=["\']([^"\']+)',
            page, re.I,
        )
        return html.unescape(m.group(1)).strip() if m else ""

    parts = [pick("og:title"), pick("og:description")]
    text = " — ".join(p for p in parts if p)
    return text if len(text) > 20 else None


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Редирект проверяем так же, как исходную ссылку.

    Иначе SSRF обходится в один шаг: публичный URL отвечает 302 на
    127.0.0.1, и urllib идёт туда сам, уже без проверки.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        if _is_private_host(newurl):
            return None  # отказ от редиректа — urllib вернёт исходный ответ
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_GuardedRedirects)


def fetch_article(url: str, timeout: int = FETCH_TIMEOUT) -> Optional["Article"]:
    """Скачать страницу и извлечь из неё статью целиком.

    Возвращает заголовок, чистый текст и признаки сатиры — без меню,
    подвала и боковой ленты. None, если страницу не удалось получить.
    """
    page = _download(url, timeout=timeout)
    if page is None:
        return None
    from .article import extract_article

    return extract_article(page, url, limit=MAX_CONTENT_CHARS)


def fetch_page(url: str, timeout: int = FETCH_TIMEOUT) -> Optional[str]:
    """Скачать страницу и вернуть чистый текст статьи. None — если не вышло.

    Никогда не бросает: недоступный источник — штатная ситуация, пайплайн
    должен продолжить без него.
    """
    article = fetch_article(url, timeout=timeout)
    return article.text if article is not None and article.ok else None


def _download(url: str, timeout: int = FETCH_TIMEOUT) -> Optional[str]:
    """Скачать HTML страницы. None, если не получилось."""
    if not url.startswith(("http://", "https://")) or _is_private_host(url):
        return None
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(MAX_PAGE_BYTES)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


# -------------------------------------------------------------- провайдеры

class LocalKbProvider:
    """Источники из курируемой базы фактов.

    Работает всегда и без сети. Это же и есть «эталон»: записи базы
    содержат первоисточник и ссылку на него.
    """

    name = "local_kb"

    def __init__(self, kb_path: Any = None):
        self._path = pathlib.Path(kb_path) if kb_path else (
            pathlib.Path(__file__).resolve().parent / "data" / "kb_facts.json"
        )

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"topics": []}

    async def find(self, query: str, limit: int = 3) -> Sequence[FetchedSource]:
        low = (query or "").lower().replace("ё", "е")
        kb = self._load()
        out: list[FetchedSource] = []
        for topic in kb.get("topics", []):
            hit = any(
                (k.lower().replace("ё", "е") in low)
                for k in topic.get("keywords", [])
            )
            if not hit:
                continue
            for fact in topic.get("facts", []):
                out.append(
                    FetchedSource(
                        id=f"kb_{fact.get('id', topic['id'])}",
                        title=f"{fact.get('source', 'База фактов')}: {fact['claim'][:80]}",
                        url=str(fact.get("url", "")),
                        content=f"{fact['claim']}. Источник: {fact.get('source', '')}",
                        retrieved_at=None,
                        domain=domain_of(str(fact.get("url", ""))),
                    )
                )
            if len(out) >= limit:
                break
        return out[:limit]


class CatalogProvider:
    """Источники из каталога доверенных сайтов.

    Провайдер не делает поиск по интернету (для этого нужен внешний API),
    но умеет принести страницу по конкретному адресу — например, если
    пользователь прислал ссылку или пайплайн знает адрес первоисточника.
    """

    name = "catalog"

    def __init__(self, catalog: Optional[Sequence[dict[str, str]]] = None):
        self._catalog = list(catalog or [])

    def add(self, url: str, title: str = "") -> None:
        self._catalog.append({"url": url, "title": title})

    async def find(self, query: str, limit: int = 3) -> Sequence[FetchedSource]:
        # Страницы качаются параллельно и в потоке: fetch_page синхронный,
        # а внутри asyncio он блокировал бы весь event loop — остальные
        # провайдеры и оценка ждали бы его скачивания.
        items = [i for i in self._catalog[:limit] if i.get("url")]
        if not items:
            return []
        pages = await asyncio.gather(
            *(asyncio.to_thread(fetch_page, i["url"]) for i in items),
            return_exceptions=True,
        )
        out: list[FetchedSource] = []
        for item, content in zip(items, pages):
            if not isinstance(content, str) or not content:
                continue
            url = item["url"]
            out.append(
                FetchedSource(
                    id=f"web_{domain_of(url)}_{len(out)}",
                    title=item.get("title") or url,
                    url=url,
                    content=content,
                    domain=domain_of(url),
                )
            )
        return out


class UrlProvider:
    """Источник по ссылке, которую прислал пользователь.

    Проверяет, что адрес безопасен и что страница реально скачалась.
    """

    name = "url"

    async def find(self, query: str, limit: int = 1) -> Sequence[FetchedSource]:
        if not query.startswith(("http://", "https://")):
            return []
        # В потоке: синхронный fetch_page иначе держит event loop до 12 с.
        content = await asyncio.to_thread(fetch_page, query)
        if not content:
            return []
        return [
            FetchedSource(
                id="user_url",
                title=query,
                url=query,
                content=content,
                domain=domain_of(query),
            )
        ]


class SerperProvider:
    """Поиск через Serper.dev — русские новости и обычный веб.

    Ключ только из окружения (SERPER_API_KEY). Без ключа провайдер
    молча выключается: пайплайн продолжит работу на базе фактов.

    Serper отдаёт сниппеты, а не полные страницы, поэтому провайдер
    дополнительно скачивает найденные страницы: для сверки цитат нужен
    текст, а не заголовок.
    """

    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"
    TIMEOUT = 12

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _search_sync(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Синхронный запрос к Serper. Пустой список при любой ошибке."""
        if not self._api_key:
            return []
        payload = json.dumps(
            {"q": query, "gl": "ru", "hl": "ru", "num": max(limit, 6)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        hits: list[dict[str, Any]] = []
        for item in data.get("organic", [])[:limit]:
            if not isinstance(item, dict):
                continue
            hits.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("link", "")),
                    "snippet": str(item.get("snippet", "")),
                    "date": str(item.get("date", "")),
                }
            )
        return hits

    async def find(self, query: str, limit: int = 3) -> Sequence[FetchedSource]:
        if not self.enabled:
            return []
        try:
            hits = await asyncio.to_thread(self._search_sync, query, limit)
        except Exception:
            return []

        out: list[FetchedSource] = []
        for index, hit in enumerate(hits):
            url = hit["url"]
            if not url:
                continue
            # Скачиваем страницу: сниппет слишком короткий для цитаты.
            content = await asyncio.to_thread(fetch_page, url)
            if not content:
                content = hit["snippet"]
            if not content:
                continue
            # Сниппет — это ровно тот кусок, который поиск нашёл по запросу.
            # Ставим его первым: в скачанной странице начало — навигация
            # сайта, а нужное предложение может быть далеко внизу.
            if hit["snippet"]:
                content = hit["snippet"] + " " + content
            out.append(
                FetchedSource(
                    id=f"serper_{index}",
                    title=hit["title"] or url,
                    url=url,
                    content=content[:MAX_CONTENT_CHARS],
                    published_at=hit["date"] or None,
                    domain=domain_of(url),
                )
            )
        return out


class NullProvider:
    """Нет источников. Пайплайн обязан честно сказать «не с чем сверять»."""

    name = "none"

    async def find(self, query: str, limit: int = 3) -> Sequence[FetchedSource]:
        return []


def combine(providers: Sequence[SourceProvider]) -> list[SourceProvider]:
    """Несколько провайдеров по приоритету. Порядок сохраняется."""
    return list(providers)
