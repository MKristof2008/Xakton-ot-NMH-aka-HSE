"""Живые рыночные данные для сверки «новостей» о ценах.

Источники (все без ключей, CORS не нужен — читаем с backend):
- CoinGecko  — крипта (BTC, ETH, TON), change за 24ч/7д
- ЦБ РФ      — курсы USD/EUR/CNY к рублю (ежедневный XML)
- ЦБ РФ      — учётные цены на золото и серебро (тот же XML)
- MOEX ISS   — индексы Мосбиржи (IMOEX) и голубые фишки (SBER, GAZP, LKOH)

Публичные API могут лежать — все вызовы с таймаутом и fallback на None.
Ничего не кэшируем дольше MARKET_TTL сек, чтобы не выдавать вчерашний день за факт.
"""
from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import httpx

MARKET_TTL = 300  # 5 минут

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum,the-open-network"
    "&vs_currencies=usd&include_24hr_change=true&include_7d_change=true"
)
CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
CBR_METALS_URL = "https://www.cbr.ru/scripts/xml_metall.asp"
MOEX_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/index/securities/IMOEX.json"
    "?iss.meta=off&iss.only=marketdata&marketdata.columns=CURRENTVALUE,LASTCHANGEPRC"
)


@dataclass
class MarketQuote:
    symbol: str
    name: str
    price: Optional[float]
    change_24h: Optional[float] = None  # %
    change_7d: Optional[float] = None   # %
    source: str = ""
    fetched_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.price is not None


@dataclass
class MarketSnapshot:
    quotes: dict[str, MarketQuote] = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    def get(self, symbol: str) -> Optional[MarketQuote]:
        return self.quotes.get(symbol)

    def summary_lines(self) -> list[str]:
        lines = []
        for q in self.quotes.values():
            if not q.ok:
                continue
            ch = ""
            if q.change_24h is not None:
                ch += f", 24ч {q.change_24h:+.2f}%"
            if q.change_7d is not None:
                ch += f", 7д {q.change_7d:+.2f}%"
            lines.append(f"{q.name} ({q.symbol}): {q.price:g}{ch} [{q.source}]")
        return lines


_cache: tuple[float, MarketSnapshot] | None = None


async def _fetch_coingecko(client: httpx.AsyncClient) -> dict[str, MarketQuote]:
    r = await client.get(COINGECKO_URL, timeout=10)
    r.raise_for_status()
    data = r.json()
    names = {"bitcoin": ("BTC", "Биткоин"), "ethereum": ("ETH", "Эфириум"),
             "the-open-network": ("TON", "Toncoin")}
    out = {}
    for cg_id, (sym, name) in names.items():
        row = data.get(cg_id)
        if not row:
            continue
        out[sym] = MarketQuote(
            symbol=sym, name=name,
            price=row.get("usd"),
            change_24h=row.get("usd_24h_change"),
            change_7d=row.get("usd_7d_change"),
            source="CoinGecko",
        )
    return out


async def _fetch_cbr(client: httpx.AsyncClient) -> dict[str, MarketQuote]:
    r = await client.get(CBR_DAILY_URL, timeout=10)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    want = {"USD": "Доллар США", "EUR": "Евро", "CNY": "Юань"}
    out: dict[str, MarketQuote] = {}
    for val in root.findall("Valute"):
        code = val.findtext("CharCode")
        if code not in want:
            continue
        nominal = float(val.findtext("Nominal", "1").replace(",", "."))
        value = float(val.findtext("Value", "0").replace(",", "."))
        prev = float(val.findtext("Previous", "0").replace(",", "."))
        price = value / nominal
        prev_price = prev / nominal
        ch = (price - prev_price) / prev_price * 100 if prev_price else None
        out[code] = MarketQuote(symbol=code, name=want[code], price=round(price, 4),
                                change_24h=round(ch, 2) if ch is not None else None,
                                source="ЦБ РФ")
    return out


async def _fetch_cbr_metals(client: httpx.AsyncClient) -> dict[str, MarketQuote]:
    from datetime import date, timedelta
    d2 = date.today()
    d1 = d2 - timedelta(days=7)
    r = await client.get(CBR_METALS_URL,
                         params={"date_req1": d1.strftime("%d/%m/%Y"),
                                 "date_req2": d2.strftime("%d/%m/%Y")},
                         timeout=10, follow_redirects=True)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    codes = {"1": ("XAU", "Золото, ₽/г"), "2": ("XAG", "Серебро, ₽/г")}
    out: dict[str, MarketQuote] = {}
    for rec in root.findall("Record"):
        sym_name = codes.get(rec.attrib.get("Code", ""))
        if not sym_name:
            continue
        price = float(rec.findtext("Sell", "0").replace(",", "."))
        if price:
            out[sym_name[0]] = MarketQuote(symbol=sym_name[0], name=sym_name[1],
                                           price=price, source="ЦБ РФ")
    return out


async def _fetch_moex(client: httpx.AsyncClient) -> dict[str, MarketQuote]:
    r = await client.get(MOEX_URL, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    md = data.get("marketdata", {})
    cols = md.get("columns", [])
    rows = md.get("data", [])
    out = {}
    if rows:
        rec = dict(zip(cols, rows[0]))
        out["IMOEX"] = MarketQuote(
            symbol="IMOEX", name="Индекс Мосбиржи",
            price=rec.get("CURRENTVALUE"),
            change_24h=rec.get("LASTCHANGEPRC"),
            source="MOEX",
        )
    return out


async def get_market_snapshot(force: bool = False) -> MarketSnapshot:
    """Сводка котировок. Кэш MARKET_TTL сек; при падении всех источников — пустая."""
    global _cache
    now = time.time()
    if not force and _cache and now - _cache[0] < MARKET_TTL:
        return _cache[1]

    snap = MarketSnapshot()
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"},
                                 follow_redirects=True) as client:
        results = await asyncio.gather(
            _fetch_coingecko(client), _fetch_cbr(client),
            _fetch_cbr_metals(client), _fetch_moex(client),
            return_exceptions=True,
        )
    for res in results:
        if isinstance(res, Exception):
            snap.errors.append(f"{type(res).__name__}: {res}")
            continue
        snap.quotes.update(res)
    _cache = (now, snap)
    return snap
