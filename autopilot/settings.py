"""Конфигурация, пути и загрузка окружения.

Пакет живёт внутри репозитория (`repo/autopilot/`), но не меняет ни один
существующий модуль: `ml/doreposta` подключается как библиотека.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUTOPILOT_DIR = Path(__file__).resolve().parent
REPO_DIR = AUTOPILOT_DIR.parent
PROJECT_DIR = REPO_DIR.parent
ML_DIR = REPO_DIR / "ml"

DATA_DIR = AUTOPILOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "autopilot.db"
CONFIG_PATH = AUTOPILOT_DIR / "config.json"
ACTIVE_MODEL_POINTER = MODELS_DIR / "active.json"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def ensure_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_ml_on_path() -> None:
    """`ml/` в sys.path — чтобы `import doreposta` находил код проекта."""
    ml = str(ML_DIR)
    if ml not in sys.path:
        sys.path.insert(0, ml)


def load_env_file(path: Path) -> int:
    """Минимальный загрузчик .env (без python-dotenv). Не перетирает окружение."""
    if not path.exists():
        return 0
    loaded = 0
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1
    except OSError:
        return loaded
    return loaded


def load_llm_env() -> int:
    """Ключ модели: `ml/.env`, затем `.env` проекта.

    Возвращает число загруженных ключей — удобно для диагностики
    «подхватился ли .env».
    """
    loaded = load_env_file(ML_DIR / ".env")
    loaded += load_env_file(PROJECT_DIR / ".env")
    return loaded


def llm_config_status() -> dict[str, object]:
    """Что известно про ключ модели — для статуса и диагностики."""
    return {
        "env_file": str(ML_DIR / ".env"),
        "env_exists": (ML_DIR / ".env").exists(),
        "has_key": bool(os.environ.get("LLM_API_KEY")),
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "model": os.environ.get("LLM_MODEL", ""),
    }


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

@dataclass
class CollectorConfig:
    interval_s: int = 900
    max_concurrency: int = 4
    request_timeout_s: float = 20.0
    retries: int = 3
    backoff_s: float = 2.0
    respect_robots: bool = True
    fetch_article_text: bool = True
    article_fetch_limit_per_source: int = 6
    max_items_per_source: int = 40
    min_text_chars: int = 80
    near_dup_hamming: int = 3
    near_dup_window_days: int = 3
    failure_cooldown_s: int = 1800
    user_agent: str = DEFAULT_USER_AGENT


@dataclass
class VerifierConfig:
    enabled: bool = True
    llm_enabled: bool = True           # false = только детерминированные проверки
    llm_hourly_budget: int = 40
    llm_daily_budget: int = 600   # потолок на сутки: 0 = без ограничения
    min_calls_per_item: int = 3   # extract + evaluate + арбитр
    verify_per_cycle: int = 12
    ml_reliability: float = 0.45
    # пороги разметки (меняются в config.json)
    min_label_confidence: float = 0.70  # ниже — метка не идёт в обучение
    min_sources_for_real: int = 1       # независимых доменов для метки «real»
    claims_full_support_conf: float = 0.80       # все утверждения подтверждены
    claims_full_contradict_conf: float = 0.85    # все утверждения опровергнуты
    claims_partial_conf: float = 0.70            # часть подтверждена / опровергнута
    claims_partial_share: float = 0.5            # доля опровергнутых для метки fake
    claims_max_unresolved_share: float = 0.5     # больше — «подтверждено» не считается
    # защита очереди от бесконечных повторов
    max_verify_attempts: int = 3        # попыток проверки на материал
    uncertain_ttl_hours: float = 48.0   # после этого — archived_uncertain
    max_sources: int = 6
    rag_top_k: int = 2
    min_overlap: float = 0.15          # порог смыслового совпадения с материалом
    trusted_domains: list[str] = field(default_factory=lambda: [
        "tass.ru", "ria.ru", "interfax.ru", "rbc.ru", "lenta.ru",
        "kommersant.ru", "banki.ru", "meduza.io", "cbr.ru", "vedomosti.ru",
        "forbes.ru", "stopfake.org", "xn--80aapkcj0afneddp0l.xn--p1ai",
    ])


@dataclass
class TrainerConfig:
    enabled: bool = True
    backend: str = "auto"              # auto | baseline | transformer
    min_new_examples: int = 50
    min_train_examples: int = 30   # меньше — модель не учим
    min_label_confidence: float = 0.70
    # правило активации новой версии
    min_f1_gain: float = 0.005          # строгий рост F1
    f1_tolerance: float = 0.005         # «равный» F1 в пределах допуска
    min_logloss_gain: float = 0.002     # улучшение LogLoss при равном F1
    min_auc_gain: float = 0.002         # улучшение ROC-AUC при равном F1
    min_data_growth: float = 0.10       # рост выборки при равном F1
    max_precision_drop: float = 0.01
    # абсолютный порог, если сравнение со старой моделью недостоверно
    min_accept_f1: float = 0.80
    min_accept_precision: float = 0.75
    # отложенная выборка
    eval_share_ru: float = 0.20
    eval_share_verified: float = 0.15
    min_fresh_eval: int = 15
    min_comparison_n: int = 40   # меньше — метрики шумные, моделей не сравниваем
    holdout_max_per_class: int = 200
    ru_repeat: int = 20                # вес русских примеров против английских
    verified_repeat: int = 25          # вес подтверждённых автопилотом меток
    use_en_corpus: bool = True
    max_en_rows: int = 8000
    seed: int = 42
    promote_to_app: bool = False
    keep_versions: int = 8
    transformer_model: str = "cointegrated/rubert-tiny2"
    transformer_epochs: float = 2.0


@dataclass
class AppConfig:
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    sources: list[dict[str, Any]] = field(default_factory=list)
    db_path: str = str(DB_PATH)
    retention_days: int = 30
    log_level: str = "INFO"
    log_to_file: bool = True

    @property
    def source_specs(self):
        from .feeds import SourceSpec

        specs = []
        for entry in self.sources:
            try:
                specs.append(SourceSpec.from_dict(entry))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Некорректный источник {entry.get('id')!r}: {exc}") from exc
        return specs


def _fill(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: Path | None = None) -> AppConfig:
    path = Path(path) if path else CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    cfg = AppConfig(
        collector=_fill(CollectorConfig, raw.get("collector")),
        verifier=_fill(VerifierConfig, raw.get("verifier")),
        trainer=_fill(TrainerConfig, raw.get("trainer")),
        sources=list(raw.get("sources") or []),
        db_path=str(raw.get("db_path") or DB_PATH),
        retention_days=int(raw.get("retention_days", 30)),
        log_level=str(raw.get("log_level", "INFO")).upper(),
        log_to_file=bool(raw.get("log_to_file", True)),
    )
    return cfg
