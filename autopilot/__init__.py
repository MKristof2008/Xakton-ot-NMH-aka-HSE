"""BAYKAM Autopilot — автономный цикл «сбор -> проверка -> дообучение».

Пакет аддитивный: существующий код `ml/doreposta` и `backend/` не меняется,
пайплайн переиспользуется как библиотека.

Точка входа: `python -m autopilot <команда>` из каталога `repo/`.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .settings import ensure_ml_on_path

ensure_ml_on_path()
