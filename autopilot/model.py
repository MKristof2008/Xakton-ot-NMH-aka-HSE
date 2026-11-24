"""Реестр версий ML-модели: активная модель, горячая перезагрузка, промоушен.

Активная версия хранится в `data/models/active.json`; при изменении файла
указателя модель перезагружается без перезапуска процесса. Пока автопилот
не обучил свою первую версию, используется модель проекта
`ml/doreposta/fake_model.joblib` (только чтение).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .db import Database
from .logsetup import get_logger
from .settings import ACTIVE_MODEL_POINTER, ML_DIR, MODELS_DIR, ensure_ml_on_path

log = get_logger("model")
ensure_ml_on_path()

# Артефакт проекта: в разных ветках он лежит в doreposta/ или lakmus/.
_PROJECT_MODEL_CANDIDATES = (
    ML_DIR / "doreposta" / "fake_model.joblib",
    ML_DIR / "lakmus" / "fake_model.joblib",
    ML_DIR / "data" / "fake_model.joblib",
)
PROJECT_MODEL = next((p for p in _PROJECT_MODEL_CANDIDATES if p.exists()),
                     _PROJECT_MODEL_CANDIDATES[0])


class ModelRegistry:
    def __init__(self, db: Database, models_dir: Path = MODELS_DIR,
                 pointer: Path = ACTIVE_MODEL_POINTER):
        self.db = db
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pointer = Path(pointer)
        self._model = None
        self._version: str | None = None
        self._source: str | None = None
        self._pointer_mtime: float | None = None
        self._loaded_at: float = 0.0

    # -- активная версия -------------------------------------------------
    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.pointer.exists():
            return None
        try:
            return json.loads(self.pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @property
    def version(self) -> str:
        if self._model is not None and self._version:
            return self._version
        return "untrained"

    @property
    def model(self):
        """Загруженный объект модели (None, если модели ещё нет)."""
        return self._model if self.load() else None

    @property
    def source(self) -> str:
        return self._source or "none"

    def active_path(self) -> Path | None:
        return self._resolve_path()[0]

    def _resolve_path(self) -> tuple[Path | None, str, str]:
        """(путь, версия, источник) активной модели."""
        pointer = self._read_pointer()
        if pointer and pointer.get("path"):
            path = Path(pointer["path"])
            if not path.is_absolute():
                path = self.models_dir / path
            if path.exists():
                return path, str(pointer.get("version", path.stem)), "autopilot"
        if PROJECT_MODEL.exists():
            return PROJECT_MODEL, "project-baseline", "project"
        return None, "untrained", "none"

    def load(self, force: bool = False) -> bool:
        """Загружает модель, если указатель изменился (или force)."""
        mtime = self.pointer.stat().st_mtime if self.pointer.exists() else 0.0
        if not force and self._model is not None and mtime == self._pointer_mtime:
            return True
        path, version, source = self._resolve_path()
        if path is None:
            self._model, self._version, self._source = None, "untrained", "none"
            self._pointer_mtime = mtime
            return False
        try:
            import joblib

            self._model = joblib.load(path)
            self._version, self._source = version, source
            self._loaded_at = time.time()
            log.info("модель загружена: %s (%s, %.1f МБ)",
                     version, source, path.stat().st_size / 1e6)
        except Exception as exc:  # битый артефакт не должен ронять монитор
            log.error("не удалось загрузить модель %s: %s", path, exc)
            self._model, self._version, self._source = None, "untrained", "none"
        self._pointer_mtime = mtime
        return self._model is not None

    def is_ready(self) -> bool:
        return self.load() and self._model is not None

    # -- предсказание ----------------------------------------------------
    def predict(self, text: str) -> float | None:
        """P(фейк) в [0, 1]; None, если модели нет или текст короткий."""
        if not text or len(text.strip()) < 20:
            return None
        if not self.load():
            return None
        try:
            proba = self._model.predict_proba([text])[0]
            return float(proba[1])
        except Exception as exc:
            log.warning("ошибка предсказания: %s", exc)
            return None

    # -- версии ----------------------------------------------------------
    def register(self, path: Path, metrics: dict[str, float], n_train: int,
                 promote: bool) -> str:
        version = Path(path).stem.replace("fake_model_", "") or f"v{int(time.time())}"
        self.db.register_model(version, str(path), metrics, n_train, active=False)
        if promote:
            self.promote(version, path)
        return version

    def promote(self, version: str, path: Path) -> None:
        pointer = {"version": version, "path": str(Path(path).resolve()),
                   "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = self.pointer.with_suffix(".tmp")
        tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.pointer)
        self.db.execute("UPDATE model_versions SET active=0")
        self.db.execute("UPDATE model_versions SET active=1 WHERE version=?", (version,))
        self.load(force=True)
        log.info("активная модель переключена на %s", version)

    def publish_to_app(self) -> Path | None:
        """Копирует активную модель в `ml/doreposta` (для backend/frontend).

        По умолчанию выключено (`trainer.promote_to_app`), чтобы не менять
        артефакты основного приложения без явного согласия.
        """
        if self._model is None and not self.load():
            return None
        path, _, source = self._resolve_path()
        if path is None or source != "autopilot":
            return None
        backup = None
        if PROJECT_MODEL.exists():
            backup = self.models_dir / f"app_backup_{int(time.time())}.joblib"
            shutil.copy2(PROJECT_MODEL, backup)
        shutil.copy2(path, PROJECT_MODEL)
        meta_src = path.with_suffix(".json")
        if meta_src.exists():
            shutil.copy2(meta_src, PROJECT_MODEL.with_name("fake_model_meta.json"))
        log.info("модель опубликована в приложение (бэкап: %s)", backup)
        return PROJECT_MODEL
