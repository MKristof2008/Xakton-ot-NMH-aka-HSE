"""Живой прогон пайплайна «Лакмус» с реальным LLM-клиентом.

Примеры:
    cd <корень репозитория>
    py -3 -m ml.lakmus.live --text "Сегодня в Москве идёт снег."
    py -3 -m ml.lakmus.live --image photo.png
    py -3 -m ml.lakmus.live --url "https://example.com/news"

Переменные окружения: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ml.lakmus.context import CheckContext
from ml.lakmus.pipeline import check
from ml.lakmus.schemas import CheckInput, InputType


def build_llm_client():
    try:
        from doreposta.llm_client import LLMClient
    except ImportError as exc:
        raise SystemExit(
            "Не удалось импортировать LLMClient из doreposta.llm_client: " + str(exc)
        )
    return LLMClient()


async def run_text(text: str) -> None:
    llm = build_llm_client()
    ctx = CheckContext(
        check_id="live_text",
        input=CheckInput(type=InputType.text, extracted_text=text),
        text=text,
        llm=llm,
    )
    result = await check(ctx)
    print(result.model_dump_json(indent=2, ensure_ascii=False))


async def run_image(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Файл не найден: {path}")
    llm = build_llm_client()
    data = path.read_bytes()
    ctx = CheckContext(
        check_id="live_image",
        input=CheckInput(type=InputType.image),
        data=data,
        llm=llm,
    )
    result = await check(ctx)
    print(result.model_dump_json(indent=2, ensure_ascii=False))


async def run_url(url: str) -> None:
    llm = build_llm_client()
    ctx = CheckContext(
        check_id="live_url",
        input=CheckInput(type=InputType.url, extracted_text=url),
        text=url,
        llm=llm,
    )
    result = await check(ctx)
    print(result.model_dump_json(indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Живой прогон Лакмус с LLM")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Текст для проверки")
    group.add_argument("--image", help="Путь к изображению")
    group.add_argument("--url", help="URL, текст которого уже извлёк backend")
    args = parser.parse_args(argv)

    if args.text:
        asyncio.run(run_text(args.text))
    elif args.image:
        asyncio.run(run_image(Path(args.image)))
    elif args.url:
        asyncio.run(run_url(args.url))


if __name__ == "__main__":
    main()
