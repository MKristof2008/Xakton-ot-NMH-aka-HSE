"""Мониторинг вызовов модели: JSONL-лог каждого запроса.

Обёртка над любым LLMClient: фиксирует время, размеры промптов и ответа,
успех/ошибку. Ключ и содержимое промптов в лог не пишем — только метрики
и первые символы системного промпта для идентификации шага пайплайна.

Лог нужен, чтобы видеть, как система оперирует моделью: сколько вызовов
уходит на одну проверку, где таймауты, как растёт контекст.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

from .llm_client import LLMClient, LLMResponse


class MonitoringLLMClient(LLMClient):
    """Пишет метрики каждого обмена с моделью в JSONL."""

    def __init__(self, inner: LLMClient, path: Union[str, Path]):
        self.inner = inner
        self.path = Path(path)

    async def complete(self, system: str, user: str) -> LLMResponse:
        started = time.monotonic()
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step": system.strip().splitlines()[0][:80] if system.strip() else "",
            "system_chars": len(system),
            "user_chars": len(user),
        }
        try:
            resp = await self.inner.complete(system, user)
        except Exception as e:
            record.update(ok=False, error=f"{type(e).__name__}: {e}"[:200],
                          latency_ms=int((time.monotonic() - started) * 1000))
            self._write(record)
            raise
        record.update(ok=True, response_chars=len(resp.text),
                      latency_ms=int((time.monotonic() - started) * 1000))
        self._write(record)
        return resp

    def _write(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # мониторинг не должен ломать проверку


def summarize_log(path: Union[str, Path]) -> dict:
    """Сводка по логу: число вызовов, доля ошибок, средняя/макс. задержка."""
    path = Path(path)
    calls = 0
    errors = 0
    latencies: list[int] = []
    if not path.exists():
        return {"calls": 0, "errors": 0, "avg_latency_ms": 0, "max_latency_ms": 0}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            calls += 1
            if not rec.get("ok"):
                errors += 1
            if isinstance(rec.get("latency_ms"), int):
                latencies.append(rec["latency_ms"])
    return {
        "calls": calls,
        "errors": errors,
        "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
    }
