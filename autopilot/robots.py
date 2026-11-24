"""Соблюдение robots.txt.

Стандартный `urllib.robotparser` в Python 3.13 трактует правило `Disallow: /?`
как `Disallow: /` и блокирует всю площадку. Здесь компактный парсер с
корректной семантикой: группы по User-agent, шаблоны `*` и `$`, побеждает
самое длинное совпавшее правило, при равенстве — `Allow`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from .logsetup import get_logger

log = get_logger("robots")


@dataclass
class _Rule:
    pattern: str
    allow: bool

    def matches(self, path: str) -> bool:
        regex = re.escape(self.pattern)
        regex = regex.replace(r"\*", ".*")
        if regex.endswith(r"\$"):
            regex = regex[:-2] + "$"
        return re.match(f"^{regex}", path) is not None

    @property
    def weight(self) -> int:
        return len(self.pattern.replace("*", "").replace("$", ""))


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[_Rule] = field(default_factory=list)

    def allows(self, path: str) -> bool:
        best: _Rule | None = None
        for rule in self.rules:
            if not rule.matches(path):
                continue
            if best is None or rule.weight > best.weight or (
                    rule.weight == best.weight and rule.allow):
                best = rule
        return True if best is None else best.allow


def parse_robots(text: str) -> list[_Group]:
    groups: list[_Group] = []
    current: _Group | None = None
    last_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if current is None or not last_was_agent:
                current = _Group()
                groups.append(current)
            current.agents.append(value.lower())
            last_was_agent = True
        elif key in ("allow", "disallow"):
            last_was_agent = False
            if current is None or not value:
                continue
            current.rules.append(_Rule(value, allow=(key == "allow")))
        else:
            last_was_agent = False
    return groups


def _select(groups: list[_Group], user_agent: str) -> _Group | None:
    ua = user_agent.lower()
    specific: _Group | None = None
    wildcard: _Group | None = None
    for group in groups:
        for agent in group.agents:
            if agent == "*":
                wildcard = wildcard or group
            elif agent and agent in ua:
                specific = group
    return specific or wildcard


class RobotsCache:
    """Один запрос robots.txt на домен; ошибки сети трактуются как «разрешено»."""

    def __init__(self, user_agent: str, timeout: float = 8.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, _Group | None] = {}

    def _load(self, origin: str) -> _Group | None:
        if origin in self._cache:
            return self._cache[origin]
        group: _Group | None = None
        try:
            resp = httpx.get(f"{origin}/robots.txt", timeout=self.timeout,
                             headers={"User-Agent": self.user_agent},
                             follow_redirects=True)
            if resp.status_code == 200:
                group = _select(parse_robots(resp.text), self.user_agent)
        except (httpx.HTTPError, OSError) as exc:
            log.debug("robots.txt недоступен для %s: %s", origin, exc)
        self._cache[origin] = group
        return group

    def allowed(self, url: str) -> bool:
        try:
            parts = urlsplit(url)
        except ValueError:
            return True
        if not parts.scheme or not parts.netloc:
            return True
        group = self._load(f"{parts.scheme}://{parts.netloc}")
        if group is None:
            return True
        path = parts.path or "/"
        if parts.query:
            path += f"?{parts.query}"
        return group.allows(path)
