"""Компонент 3 — Trainer: непрерывное дообучение с защитой от деградации.

Железо этого ПК: 8 ядер CPU, ~8 ГБ ОЗУ, CUDA нет. Поэтому рабочая модель —
TF-IDF (слово 1–2 + char_wb 3–5) + калиброванная логистическая регрессия:
обучение занимает секунды. При наличии GPU и пакетов torch+transformers
бэкенд переключается на дообучение русского BERT (`trainer.backend`).

Данные:
- базовый корпус проекта: `ml/ru_fake_examples.json`, `ml/feedback_examples.jsonl`;
- подтверждённые метки автопилота (fake/real с уверенностью выше порога);
- английский корпус HuggingFace, если он есть в локальном кэше.

Отложенная выборка (`models/holdout.jsonl`):
- создаётся стратифицированно по классам, хеши текстов не дают дублей;
- пополняется свежими подтверждёнными метками (с потолком на класс);
- никогда не попадает в обучение.

Метрики: F1, precision, recall, accuracy, LogLoss, ROC-AUC — на русских
примерах (английский корпус иначе завышает F1 и скрывает деградацию).

Правило активации (`_decide`):
1. Если активная модель — проектный бейзлайн, обученный на том же корпусе,
   что и holdout, сравнение с ним недостоверно (утечка). Тогда действует
   абсолютный порог: `min_accept_f1` / `min_accept_precision`. Иначе при
   F1 = 1.0 модель не обновилась бы никогда.
2. Иначе — строгий рост F1, ЛИБО равный F1 при улучшении LogLoss или ROC-AUC,
   ЛИБО равный F1 при заметном росте обучающей выборки. Precision не должен
   упасть больше `max_precision_drop`.
3. Откат: `python -m autopilot models --promote <version>`.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .db import Database
from .logsetup import get_logger
from .model import ModelRegistry
from .settings import ML_DIR, MODELS_DIR, TrainerConfig
from .textnorm import detect_lang, normalize_for_hash

log = get_logger("trainer")


@dataclass
class Example:
    text: str
    label: int          # 1 = fake, 0 = real
    origin: str         # ru | feedback | verified | en
    weight: float = 1.0
    item_id: int | None = None


@dataclass
class TrainResult:
    promoted: bool
    version: str | None
    metrics: dict[str, float]
    baseline: dict[str, float]
    n_train: int
    n_test: int
    n_new: int
    notes: str


def _hash(text: str) -> str:
    """Отпечаток текста для защиты от дублей: регистр и пунктуация не влияют."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()[:24]


def _stratified(rows: Sequence[Example], share: float, seed: int) -> list[Example]:
    rng = random.Random(seed)
    by_label: dict[int, list[Example]] = {}
    for row in rows:
        by_label.setdefault(row.label, []).append(row)
    held: list[Example] = []
    for group in by_label.values():
        rng.shuffle(group)
        cut = int(round(len(group) * share))
        if len(group) >= 10:
            cut = max(1, min(cut, len(group) - 1))
        held.extend(group[:cut])
    return held


class Trainer:
    def __init__(self, db: Database, cfg: TrainerConfig, registry: ModelRegistry,
                 models_dir: Path = MODELS_DIR):
        self.db = db
        self.cfg = cfg
        self.registry = registry
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_path = self.models_dir / "holdout.jsonl"
        self.backend = self._choose_backend()

    def _choose_backend(self) -> str:
        if self.cfg.backend == "baseline":
            return "baseline"
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except Exception:
            has_gpu = False
        if self.cfg.backend == "transformer" or (self.cfg.backend == "auto" and has_gpu):
            try:
                import transformers  # noqa: F401
                import torch

                if torch.cuda.is_available():
                    return "transformer"
            except Exception:
                log.warning("torch/transformers недоступны — TF-IDF бейзлайн")
        return "baseline"

    # -- данные ----------------------------------------------------------
    def base_examples(self) -> list[Example]:
        """Базовый корпус проекта: обратная связь, эталонный набор, база фактов.

        Собранный автопилотом корпус идёт сюда же, но с низким весом —
        это стилевые метки, а не подтверждённые факты.
        """
        from .training_data import collect, describe

        rows = collect(self.db)
        if not rows:
            log.warning("в проекте не найдено ни одного источника обучающих данных")
            return []
        summary = describe(rows)
        log.info("обучающий корпус: %d примеров (fake %d, real %d) из %s",
                 summary["n"], summary["fake"], summary["real"],
                 ", ".join(f"{k}={v}" for k, v in summary["by_origin"].items()))
        return [Example(row.text, row.label, row.origin, row.weight, row.item_id)
                for row in rows]

    def en_examples(self) -> list[Example]:
        if not self.cfg.use_en_corpus:
            return []
        try:
            from datasets import load_dataset

            ds = load_dataset("GonzaloA/fake_news", split="train")
        except Exception as exc:
            log.info("EN-корпус недоступен (%s) — учимся на русских данных",
                     str(exc)[:120])
            return []
        out: list[Example] = []
        for row in ds.select(range(min(self.cfg.max_en_rows, len(ds)))):
            text = f"{row.get('title') or ''}. {(row.get('text') or '')[:1200]}".strip()
            if len(text) >= 80:
                out.append(Example(text, 1 - int(row["label"]), "en", 1.0))
        log.info("EN-корпус: %d примеров", len(out))
        return out

    def verified_examples(self) -> list[Example]:
        rows = self.db.labeled_examples(self.cfg.min_label_confidence)
        return [
            Example(row["text"], 1 if row["label"] == "fake" else 0, "verified",
                    float(self.cfg.verified_repeat), int(row["id"]))
            for row in rows
        ]

    # -- holdout ---------------------------------------------------------
    def _load_holdout(self) -> list[Example]:
        if not self.holdout_path.exists():
            return []
        out = []
        try:
            for line in self.holdout_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    out.append(Example(row["text"], int(row["label"]),
                                       row.get("origin", "ru")))
        except (OSError, json.JSONDecodeError):
            return []
        return out

    def _save_holdout(self, holdout: Sequence[Example]) -> None:
        with self.holdout_path.open("w", encoding="utf-8") as fh:
            for ex in holdout:
                fh.write(json.dumps({"text": ex.text, "label": ex.label,
                                     "origin": ex.origin}, ensure_ascii=False) + "\n")

    def ensure_holdout(self, base: Sequence[Example],
                       verified: Sequence[Example]) -> list[Example]:
        """Отложенная выборка: стратифицирована, пополняется свежими метками.

        - базовые примеры берутся один раз с `eval_share_ru` на класс;
        - новые подтверждённые примеры добавляются с `eval_share_verified`,
          без дублей по тексту и с потолком `holdout_max_per_class` на класс;
        - удалённые/устаревшие строки не мешают: файл перезаписывается целиком.
        """
        holdout = self._load_holdout()
        known = {_hash(ex.text) for ex in holdout}

        if not holdout:
            for ex in _stratified(base, self.cfg.eval_share_ru, self.cfg.seed):
                holdout.append(Example(ex.text, ex.label, "ru"))
                known.add(_hash(ex.text))
            log.info("holdout: %d базовых примеров", len(holdout))

        counts = Counter(ex.label for ex in holdout)
        added = 0
        for ex in verified:
            h = _hash(ex.text)
            if h in known:
                continue
            if counts[ex.label] >= self.cfg.holdout_max_per_class:
                continue
            if int(h[:8], 16) % 100 >= int(self.cfg.eval_share_verified * 100):
                continue
            holdout.append(Example(ex.text, ex.label, "verified"))
            known.add(h)
            counts[ex.label] += 1
            added += 1
        if added:
            log.info("holdout: +%d свежих примеров (fake=%d, real=%d)",
                     added, counts[1], counts[0])

        self._save_holdout(holdout)
        return holdout

    def holdout_summary(self) -> dict[str, Any]:
        """Состав отложенной выборки: по классам, по происхождению, насыщение.

        `saturated` — F1 на holdout равен 1.0. Это не качество модели,
        а признак слишком простой выборки: различать версии по F1 здесь
        бессмысленно, поэтому включаются LogLoss / ROC-AUC / объём данных.
        """
        rows = self._load_holdout()
        by_label = Counter(ex.label for ex in rows)
        by_origin = Counter(ex.origin for ex in rows)
        last = self.db.last_training_run()
        f1 = last["f1"] if last is not None else None
        return {
            "n": len(rows),
            "fake": by_label.get(1, 0),
            "real": by_label.get(0, 0),
            "fresh": by_origin.get("verified", 0),
            "by_origin": dict(by_origin),
            "last_f1": f1,
            "saturated": f1 is not None and float(f1) >= 0.999,
            "min_comparison_n": self.cfg.min_comparison_n,
            "small": len(rows) < self.cfg.min_comparison_n,
        }

    # -- метрики ---------------------------------------------------------
    @staticmethod
    def metrics(y_true: Sequence[int], y_pred: Sequence[int],
                y_prob: Sequence[float] | None = None) -> dict[str, float]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out = {"f1": round(f1, 4), "precision": round(precision, 4),
               "recall": round(recall, 4),
               "accuracy": round((tp + tn) / max(1, len(y_true)), 4),
               "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y_true)}
        if y_prob is not None and len(set(y_true)) > 1:
            try:
                from sklearn.metrics import log_loss, roc_auc_score

                clipped = [min(max(float(p), 1e-6), 1 - 1e-6) for p in y_prob]
                out["logloss"] = round(float(log_loss(y_true, clipped)), 4)
                out["roc_auc"] = round(float(roc_auc_score(y_true, list(y_prob))), 4)
            except Exception as exc:
                log.debug("logloss/auc недоступны: %s", exc)
        return out

    def evaluate(self, model, holdout: Sequence[Example]) -> dict[str, float]:
        """Метрики на русских примерах: английские тексты F1 только завышают."""
        rows = [ex for ex in holdout if detect_lang(ex.text) == "ru"] or list(holdout)
        if not rows:
            return {}
        try:
            proba = model.predict_proba([ex.text for ex in rows])[:, 1]
        except Exception as exc:
            log.warning("модель не смогла предсказать holdout: %s", exc)
            return {}
        pred = (proba >= 0.5).astype(int).tolist()
        return self.metrics([ex.label for ex in rows], pred, proba.tolist())

    def _comparison_set(self, holdout: Sequence[Example]) -> list[Example]:
        """Свежие примеры предпочтительнее: на них нет утечки из обучения."""
        fresh = [ex for ex in holdout if ex.origin == "verified"]
        return fresh if len(fresh) >= self.cfg.min_fresh_eval else list(holdout)

    # -- решение о публикации -------------------------------------------
    def _decide(self, metrics: dict, baseline: dict, baseline_leaked: bool,
                n_train: int, prev_n_train: int) -> tuple[bool, list[str]]:
        if not metrics:
            return False, ["метрики не посчитаны — публикация невозможна"]

        f1 = float(metrics.get("f1", 0.0))
        precision = float(metrics.get("precision", 0.0))

        if baseline_leaked or not baseline:
            ok = (f1 >= self.cfg.min_accept_f1
                  and precision >= self.cfg.min_accept_precision)
            why = ("старейший бейзлайн обучен на этой же выборке — сравнение недостоверно"
                   if baseline_leaked else "старой модели нет")
            verdict = ("приём по абсолютному порогу" if ok else "ниже абсолютного порога")
            return ok, [f"{why}: {verdict} "
                        f"(F1 {f1:.3f} при пороге {self.cfg.min_accept_f1}, "
                        f"precision {precision:.3f} при пороге "
                        f"{self.cfg.min_accept_precision})"]

        old_f1 = float(baseline.get("f1", 0.0))
        old_precision = float(baseline.get("precision", 0.0))
        old_logloss = baseline.get("logloss")
        old_auc = baseline.get("roc_auc")
        new_logloss = metrics.get("logloss")
        new_auc = metrics.get("roc_auc")

        gain = f1 >= old_f1 + self.cfg.min_f1_gain
        equal = f1 >= old_f1 - self.cfg.f1_tolerance
        better_logloss = (old_logloss is not None and new_logloss is not None
                          and new_logloss <= old_logloss - self.cfg.min_logloss_gain)
        better_auc = (old_auc is not None and new_auc is not None
                      and new_auc >= old_auc + self.cfg.min_auc_gain)
        more_data = (prev_n_train > 0
                     and n_train >= prev_n_train * (1 + self.cfg.min_data_growth))
        precision_ok = precision >= old_precision - self.cfg.max_precision_drop

        reasons: list[str] = []
        if gain:
            reasons.append(f"F1 {old_f1:.3f}->{f1:.3f}")
        elif equal and (better_logloss or better_auc or more_data):
            if better_logloss:
                reasons.append(f"LogLoss {old_logloss:.3f}->{new_logloss:.3f}")
            if better_auc:
                reasons.append(f"ROC-AUC {old_auc:.3f}->{new_auc:.3f}")
            if more_data:
                reasons.append(f"данных {prev_n_train}->{n_train}")
        ok = (gain or (equal and (better_logloss or better_auc or more_data))) \
            and precision_ok

        if not ok:
            if not equal and not gain:
                reasons = [f"F1 упал: {old_f1:.3f}->{f1:.3f}"]
            elif not precision_ok:
                reasons = [f"precision упал: {old_precision:.3f}->{precision:.3f}"]
            else:
                reasons = [f"F1 не вырос ({old_f1:.3f}->{f1:.3f}), "
                           "LogLoss/ROC-AUC/объём данных без улучшения"]
        return ok, reasons

    def _previous_n_train(self) -> int:
        active = self.db.active_model()
        if active is not None and active["n_train"]:
            return int(active["n_train"])
        last = self.db.query_one(
            "SELECT n_train FROM training_runs WHERE promoted=1 "
            "ORDER BY id DESC LIMIT 1")
        return int(last["n_train"] or 0) if last else 0

    # -- модели ----------------------------------------------------------
    def build_baseline(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                     sublinear_tf=True, max_features=150_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=2, sublinear_tf=True, max_features=150_000)),
        ])
        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=2000, C=2.0), cv=3, method="sigmoid")
        return Pipeline([("tfidf", features), ("clf", clf)])

    def _train_transformer(self, texts: Sequence[str], labels: Sequence[int],
                           weights: Sequence[float]):
        """GPU-ветка: дообучение русского BERT (по умолчанию rubert-tiny2)."""
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  Trainer as HFTrainer, TrainingArguments)

        name = self.cfg.transformer_model

        class _DS(Dataset):
            def __init__(self, enc, labels_, weights_):
                self.enc, self.labels, self.weights = enc, labels_, weights_

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, i):
                item = {k: v[i] for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
                item["weight"] = torch.tensor(self.weights[i], dtype=torch.float)
                return item

        tokenizer = AutoTokenizer.from_pretrained(name)
        enc = tokenizer(list(texts), truncation=True, padding=True,
                        max_length=256, return_tensors="pt")
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

        class _Weighted(HFTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                sample_weights = inputs.pop("weight", None)
                outputs = model(**inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, inputs["labels"], reduction="none")
                loss = (loss * sample_weights).mean() if sample_weights is not None \
                    else loss.mean()
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=str(self.models_dir / "hf_runs"),
            num_train_epochs=self.cfg.transformer_epochs,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        )
        _Weighted(model=model, args=args, train_dataset=_DS(enc, labels, weights)).train()
        return _TransformerWrapper(model, tokenizer, np)

    # -- обучение --------------------------------------------------------
    def train(self, *, reason: str = "manual",
              include_en: bool = True) -> TrainResult | None:
        started = time.time()
        base = self.base_examples()
        verified = self.verified_examples()
        pool = base + verified
        # порог низкий осознанно: честный корпус проекта невелик, а holdout
        # и правило активации всё равно не пропустят слабую модель
        if len(pool) < self.cfg.min_train_examples or len({ex.label for ex in pool}) < 2:
            log.warning("мало данных для обучения: %d примеров (порог %d)",
                        len(pool), self.cfg.min_train_examples)
            return None

        holdout = self.ensure_holdout(base, verified)
        holdout_hashes = {_hash(ex.text) for ex in holdout}
        train_rows = [ex for ex in pool if _hash(ex.text) not in holdout_hashes]
        if include_en:
            train_rows.extend(self.en_examples())
        if len({ex.label for ex in train_rows}) < 2:
            log.warning("в обучающей выборке только один класс — обучение отменено")
            return None

        # длинные тексты дополнительно режем на фрагменты
        expanded = list(train_rows)
        for ex in train_rows:
            if len(ex.text) > 700:
                for chunk in _chunks(ex.text)[:3]:
                    expanded.append(Example(chunk, ex.label, ex.origin, ex.weight * 0.4))

        texts = [ex.text for ex in expanded]
        labels = [ex.label for ex in expanded]
        weights = [ex.weight for ex in expanded]

        comparison = self._comparison_set(holdout)
        if len(comparison) < self.cfg.min_comparison_n:
            log.warning("выборка для сравнения мала: %d примеров (порог %d) — "
                        "метрики на ней шумные, решение может быть случайным",
                        len(comparison), self.cfg.min_comparison_n)
        run_id = self.db.start_training_run(len(texts), len(verified), len(comparison))
        active_model = self.registry.model
        baseline = self.evaluate(active_model, comparison) if active_model is not None else {}
        baseline_f1 = float(baseline.get("f1", 0.0))

        # проектный артефакт обучался на данных проекта — если holdout
        # собран из тех же источников, его метрики завышены (утечка)
        baseline_leaked = bool(baseline) and self.registry.source == "project" \
            and any(ex.origin == "ru" for ex in comparison)

        try:
            if self.backend == "transformer":
                model = self._train_transformer(texts, labels, weights)
            else:
                model = self.build_baseline()
                model.fit(texts, labels, clf__sample_weight=weights)
            metrics = self.evaluate(model, comparison)
        except Exception as exc:
            log.exception("обучение упало: %s", exc)
            self.db.finish_training_run(run_id, metrics={}, baseline_f1=baseline_f1,
                                        promoted=False, model_version=None,
                                        notes=f"ошибка: {exc}")
            return None

        promoted, reasons = self._decide(
            metrics, baseline, baseline_leaked, len(texts), self._previous_n_train())

        version = time.strftime("v%Y%m%d_%H%M%S")
        notes = reasons[:]
        if promoted:
            path = self.models_dir / f"fake_model_{version}.joblib"
            try:
                import joblib

                tmp = path.with_suffix(".tmp")
                joblib.dump(model, tmp)
                tmp.replace(path)
                path.with_suffix(".json").write_text(json.dumps({
                    "version": version, "backend": self.backend,
                    "metrics": metrics, "baseline": baseline,
                    "baseline_leaked": baseline_leaked,
                    "n_train": len(texts), "n_new": len(verified),
                    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": ("TF-IDF(word 1-2, char_wb 3-5) + LogReg (калибр.)"
                              if self.backend == "baseline" else self.cfg.transformer_model),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                self.registry.register(path, metrics, len(texts), promote=True)
                self.db.mark_trained([ex.item_id for ex in verified if ex.item_id])
                if self.cfg.promote_to_app:
                    self.registry.publish_to_app()
            except Exception as exc:
                log.exception("не удалось сохранить модель: %s", exc)
                promoted, notes = False, [f"ошибка сохранения: {exc}"]
        else:
            notes = ["отклонено: " + "; ".join(reasons)]

        dist = Counter(ex.label for ex in comparison)
        notes.append(f"выборка: {len(comparison)} (fake {dist.get(1, 0)}, "
                     f"real {dist.get(0, 0)}), обучающих {len(texts)}")
        if float(metrics.get("f1", 0.0)) >= 0.999:
            notes.append("F1 насыщен (1.0): выборка простая, решают "
                         "LogLoss/ROC-AUC/объём данных")
        self.db.finish_training_run(run_id, metrics=metrics, baseline_f1=baseline_f1,
                                    promoted=promoted,
                                    model_version=version if promoted else None,
                                    notes="; ".join(notes))
        self._prune_versions()
        log.info("обучение (%s): %s, F1 %.4f (было %.4f), precision %.4f, recall %.4f, "
                 "LogLoss %s, ROC-AUC %s, примеров %d, %.1f с; %s",
                 reason, "принято" if promoted else "отклонено",
                 float(metrics.get("f1", 0.0)), baseline_f1,
                 float(metrics.get("precision", 0.0)),
                 float(metrics.get("recall", 0.0)),
                 metrics.get("logloss", "—"), metrics.get("roc_auc", "—"),
                 len(texts), time.time() - started, "; ".join(notes))
        return TrainResult(promoted=promoted, version=version if promoted else None,
                           metrics=metrics, baseline=baseline, n_train=len(texts),
                           n_test=len(comparison), n_new=len(verified),
                           notes="; ".join(notes))

    def _prune_versions(self) -> None:
        files = sorted(self.models_dir.glob("fake_model_*.joblib"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        active = self.registry.active_path()
        for path in files[self.cfg.keep_versions:]:
            if active and path.resolve() == active.resolve():
                continue
            try:
                path.unlink()
                path.with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                pass

    def should_retrain(self) -> bool:
        if not self.cfg.enabled:
            return False
        last = self.db.last_training_finished_at()
        new_labels = self.db.new_labels_since(last, self.cfg.min_label_confidence)
        if new_labels >= self.cfg.min_new_examples:
            log.info("накопилось %d новых подтверждённых меток — дообучение", new_labels)
            return True
        return False


class _TransformerWrapper:
    """Обёртка HF-модели с интерфейсом `predict_proba`."""

    def __init__(self, model, tokenizer, np):
        self.model, self.tokenizer, self.np = model, tokenizer, np

    def predict_proba(self, texts):
        import torch

        self.model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = self.tokenizer(list(texts[i:i + 16]), truncation=True,
                                       padding=True, max_length=256,
                                       return_tensors="pt").to(self.model.device)
                chunks.append(torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy())
        return self.np.vstack(chunks) if chunks else self.np.zeros((0, 2))


def _chunks(text: str, size: int = 450) -> list[str]:
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for part in parts:
        if len(buf) + len(part) <= size:
            buf = f"{buf} {part}".strip()
        else:
            if buf:
                out.append(buf)
            buf = part[:size]
    if buf:
        out.append(buf)
    return out
