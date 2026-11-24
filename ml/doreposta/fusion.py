"""Слияние сигналов по субъективной логике Йосанга (ТЗ «Лакмус», раздел 6.6).

Каждый сигнал превращается в мнение (b, d, u): вера «настоящее», вера
«подделка», неопределённость; b + d + u = 1. Мнения объединяются попарно
cumulative fusion. Итог: score = round(100·(b + 0.5·u)), uncertainty = u.

Главное свойство: система знает, когда она не знает. Нет сигналов —
не «примерно 50%», а честный Туман.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Пороги уровней (раздел 3 ТЗ). Порядок проверки важен.
FOG_UNCERTAINTY = 0.40
STRONG_SIGNAL = 0.6
CONFLICT_U_BONUS = 0.2
LOW_COVERAGE_MIN_U = 0.45

LEVEL_FOG = "fog"                       # Туман
LEVEL_CONFIRMED = "confirmed"           # Подтверждено
LEVEL_LIKELY_REAL = "likely_real"       # Скорее настоящее
LEVEL_DOUBTFUL = "doubtful"             # Сомнительно
LEVEL_LIKELY_FAKE = "likely_fake"       # Скорее подделка

LEVEL_LABELS = {
    LEVEL_FOG: "Туман",
    LEVEL_CONFIRMED: "Подтверждено",
    LEVEL_LIKELY_REAL: "Скорее настоящее",
    LEVEL_DOUBTFUL: "Сомнительно",
    LEVEL_LIKELY_FAKE: "Скорее подделка",
}


@dataclass(frozen=True)
class Opinion:
    b: float  # belief «настоящее»
    d: float  # belief «подделка»
    u: float  # неопределённость

    def __post_init__(self):
        total = self.b + self.d + self.u
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"b + d + u должно быть 1, получено {total}")
        for v in (self.b, self.d, self.u):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"компоненты мнения в [0, 1], получено {v}")

    @property
    def probability(self) -> float:
        return self.b + 0.5 * self.u


def opinion_from_signal(verdict: str, strength: float, reliability: float) -> Opinion:
    """Сигнал -> мнение (формулы из ТЗ 6.6)."""
    strength = min(max(strength, 0.0), 1.0)
    reliability = min(max(reliability, 0.0), 1.0)
    if verdict == "supports_real":
        b = strength * reliability
        return Opinion(b=b, d=0.0, u=1.0 - b)
    if verdict == "supports_fake":
        d = strength * reliability
        return Opinion(b=0.0, d=d, u=1.0 - d)
    # neutral / unavailable — сигнал ничего не добавляет
    return Opinion(b=0.0, d=0.0, u=1.0)


def fuse(left: Opinion, right: Opinion) -> Opinion:
    """Cumulative fusion двух мнений."""
    k = left.u + right.u - left.u * right.u
    if k == 0:  # оба абсолютно уверены и противоречат — конфликт максимален
        return Opinion(b=(left.b + right.b) / 2, d=(left.d + right.d) / 2, u=0.0)
    return Opinion(
        b=(left.b * right.u + right.b * left.u) / k,
        d=(left.d * right.u + right.d * left.u) / k,
        u=left.u * right.u / k,
    )


def fuse_all(opinions: Sequence[Opinion]) -> Opinion:
    """Сворачивает список мнений. Пустой вход — полная неопределённость."""
    if not opinions:
        return Opinion(b=0.0, d=0.0, u=1.0)
    result = opinions[0]
    for op in opinions[1:]:
        result = fuse(result, op)
    return result


@dataclass(frozen=True)
class TrustResult:
    score: int            # 0–100, 100 = скорее настоящее
    uncertainty: float    # 0–1
    coverage: float       # доля отработавших сигналов
    conflict: bool
    level: str
    level_label: str


def compute_trust(
    signals: Sequence,  # объекты с полями verdict, strength, reliability
    has_provenance: bool = False,
) -> TrustResult:
    """Итог доверия по списку сигналов с поправками из ТЗ 6.6.

    has_provenance — есть ли сильный сигнал происхождения (C2PA, первоисточник),
    без него уровень «Подтверждено» не присваивается.
    """
    total = len(signals)
    working = [s for s in signals if getattr(s, "verdict", "unavailable") != "unavailable"]
    coverage = (len(working) / total) if total else 0.0

    opinions = [opinion_from_signal(s.verdict, s.strength, s.reliability) for s in working]
    # Правило из ТЗ: сильная проверка по источникам приглушает слабый
    # стилистический флаг — риторика не перевешивает факты.
    strong_claim = any(
        o.b > STRONG_SIGNAL or o.d > STRONG_SIGNAL for o in opinions)
    if strong_claim:
        opinions = [
            Opinion(o.b * 0.3, o.d * 0.3, 1.0 - (o.b + o.d) * 0.3)
            if getattr(s, "id", "") == "rhetoric" else o
            for s, o in zip(working, opinions)
        ]
    merged = fuse_all(opinions)

    conflict = any(o.b > STRONG_SIGNAL for o in opinions) and \
               any(o.d > STRONG_SIGNAL for o in opinions)

    u = merged.u
    if conflict:
        u = min(1.0, u + CONFLICT_U_BONUS)
    if coverage < 0.5:
        u = max(u, LOW_COVERAGE_MIN_U)

    score = round(100 * merged.probability)

    # Порядок проверки важен (ТЗ, раздел 3).
    if u > FOG_UNCERTAINTY or conflict:
        level = LEVEL_FOG
    elif score >= 80 and has_provenance:
        level = LEVEL_CONFIRMED
    elif score >= 60:
        level = LEVEL_LIKELY_REAL
    elif score >= 35:
        level = LEVEL_DOUBTFUL
    else:
        level = LEVEL_LIKELY_FAKE

    return TrustResult(
        score=score,
        uncertainty=round(u, 4),
        coverage=round(coverage, 4),
        conflict=conflict,
        level=level,
        level_label=LEVEL_LABELS[level],
    )
