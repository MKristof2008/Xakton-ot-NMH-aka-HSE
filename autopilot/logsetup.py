"""Логирование в консоль и в файл с ротацией."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: Path | None = None,
                  to_file: bool = True) -> logging.Logger:
    """Идемпотентно настраивает логгер `autopilot`."""
    root = logging.getLogger("autopilot")
    if getattr(root, "_configured", False):
        return root

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    stream = sys.stdout
    try:  # кириллица в Windows-консоли
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    console = logging.StreamHandler(stream)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)

    if to_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "autopilot.log", maxBytes=5_000_000, backupCount=5,
            encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)

    root._configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"autopilot.{name}")
