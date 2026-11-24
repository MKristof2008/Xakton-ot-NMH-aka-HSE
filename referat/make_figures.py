"""Рисунки для реферата «Поиск и критическая оценка информации в Интернете».

Все схемы строятся из фактических данных проекта: источники, сигналы и
пороги берутся из кода, а не рисуются «на глаз».

Запуск:  py -3 referat/make_figures.py
Результат: referat/fig/fig1..fig7.png (300 dpi)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
# Код продукта лежит в repo-nmh/ (рабочий репозиторий).
PROJECT = ROOT
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(parents=True, exist_ok=True)
# ml/ — корень пакета lakmus, чтобы пороги брались из кода, а не переписывались.
sys.path.insert(0, str(PROJECT / "ml"))

# Палитра под тему приложения: инженерный синий, холодная бумага, красный акцент.
INK = "#0a1a33"
MUTED = "#56637a"
LINE = "#a9b6cb"
PAPER = "#eef2f8"
BLUE = "#12409b"
BLUE_SOFT = "#dbe6fb"
RED = "#c02a1e"
RED_SOFT = "#fbe7e3"
WARN = "#e8a13a"
WARN_SOFT = "#fdf1dc"
OK = "#0f6b45"
OK_SOFT = "#e2f1ea"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.edgecolor": LINE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _box(ax, x, y, w, h, text, face=BLUE_SOFT, edge=INK, bold=False, size=12):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, style="-|>", color=INK, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        color=color, linewidth=1.3, zorder=1,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}"))


def _canvas(w=10, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", path.name)


# ---------------------------------------------------------------- рисунок 1
def fig1_search_cycle():
    """Цикл поиска и критической оценки информации."""
    fig, ax = _canvas(10.4, 4.4)

    top = [("Информационная\nпотребность", 0.02), ("Формулировка\nзапроса", 0.27),
           ("Выбор поисковой\nсистемы", 0.52), ("Просмотр\nвыдачи", 0.77)]
    bottom = [("Использование\nили отказ", 0.02), ("Перекрёстная\nпроверка", 0.27),
              ("Критическая\nоценка", 0.52), ("Отбор\nисточников", 0.77)]

    for text, x in top:
        _box(ax, x, 0.66, 0.21, 0.20, text)
    for text, x in bottom:
        face = WARN_SOFT if "Критическая" in text or "Перекрёстная" in text else BLUE_SOFT
        _box(ax, x, 0.18, 0.21, 0.20, text, face=face)

    for i in range(3):
        _arrow(ax, top[i][1] + 0.21, 0.76, top[i + 1][1], 0.76)
    _arrow(ax, 0.875, 0.66, 0.875, 0.38)
    for i in range(3, 0, -1):
        _arrow(ax, bottom[i][1], 0.28, bottom[i - 1][1] + 0.21, 0.28)
    # возврат к уточнению запроса
    _arrow(ax, 0.375, 0.38, 0.375, 0.66, dashed=True, color=MUTED, rad=-0.45)
    ax.text(0.30, 0.52, "уточнение\nзапроса", ha="right", va="center",
            fontsize=11, style="italic", color=MUTED)

    ax.text(0.5, 0.05, "Выделенные блоки — этапы критической оценки информации",
            ha="center", va="center", fontsize=11, style="italic", color=MUTED)
    _save(fig, "fig1_cycle.png")


# ---------------------------------------------------------------- рисунок 2
def fig2_sift():
    """Четыре шага метода SIFT."""
    fig, ax = _canvas(10.4, 3.4)
    steps = [
        ("S\nStop", "Остановись:\nне реагируй сразу,\nоцени цель"),
        ("I\nInvestigate", "Изучи источник:\nкто автор, чей\nинтерес, репутация"),
        ("F\nFind", "Найди лучшее\nпокрытие: другие\nнезависимые источники"),
        ("T\nTrace", "Проследи до\nпервоисточника\nи исходного контекста"),
    ]
    xs = [0.02, 0.27, 0.52, 0.77]
    for (title, body), x in zip(steps, xs):
        _box(ax, x, 0.60, 0.21, 0.32, title, face=OK_SOFT, bold=True, size=15)
        _box(ax, x, 0.10, 0.21, 0.40, body, face=BLUE_SOFT, size=11)
        _arrow(ax, x + 0.105, 0.60, x + 0.105, 0.50)
    for i in range(3):
        _arrow(ax, xs[i] + 0.21, 0.76, xs[i + 1], 0.76)
    _save(fig, "fig2_sift.png")


# ---------------------------------------------------------------- рисунок 3
def fig3_pipeline():
    """Конвейер обработки сообщения: как он собран в коде."""
    steps = [
        ("Ввод", "текст, ссылка,\nскриншот или видео", BLUE_SOFT),
        ("Тема", "класс утверждения\nбез модели: словари", BLUE_SOFT),
        ("Утверждения", "до трёх фактов;\nмнения отсекаются", BLUE_SOFT),
        ("Источники", "база фактов →\nинтернет → ссылка", BLUE_SOFT),
        ("Оценка", "статус строго по\nтексту источника", BLUE_SOFT),
        ("Проверка кодом", "цитата дословно\nв источнике", WARN_SOFT),
        ("Сигналы", "вердикт, сила,\nнадёжность", BLUE_SOFT),
        ("Слияние", "субъективная логика:\nscore, uncertainty", BLUE_SOFT),
        ("Ответ", "вердикт + разбор\nпо тарифу", OK_SOFT),
    ]
    fig, ax = _canvas(9.6, 7.4)
    h = 0.088
    gap = 0.016
    y = 0.98 - h
    for i, (title, body, face) in enumerate(steps):
        _box(ax, 0.03, y, 0.36, h, title, face=face, bold=True, size=12)
        ax.text(0.42, y + h / 2, body, ha="left", va="center", fontsize=11, color=INK)
        if i < len(steps) - 1:
            _arrow(ax, 0.21, y, 0.21, y - gap + 0.002)
        y -= h + gap
    _save(fig, "fig3_pipeline.png")


# ---------------------------------------------------------------- рисунок 4
def fig4_signals():
    """Сигналы проверки: какие есть и что дают."""
    groups = [
        ("Текст и ссылки", BLUE_SOFT, [
            ("claims", "есть ли в сообщении проверяемые утверждения"),
            ("kb_facts", "сверка чисел и дат с курируемой базой фактов"),
            ("sources", "найдены ли первоисточники и как они согласуются"),
            ("quotes", "дословные цитаты, проверенные кодом"),
            ("rhetoric", "стиль давления: флаг для читателя, а не признак лжи"),
            ("machine_text", "признаки машинно сгенерированного текста"),
        ]),
        ("Изображения", WARN_SOFT, [
            ("provenance", "метаданные файла: камера, дата, редактор (EXIF)"),
            ("ai_artifacts", "следы генератора на самом изображении"),
            ("manipulation", "следы монтажа и повторного сжатия"),
            ("screenshot", "тип скриншота и читаемость текста на нём"),
            ("uniqueness", "дубликаты по перцептивному хешу"),
        ]),
        ("Видео", OK_SOFT, [
            ("video", "метаданные контейнера, подписи генератора, статичность кадров"),
        ]),
    ]
    fig, ax = _canvas(10.4, 7.6)
    header, row, gap = 0.075, 0.058, 0.022
    total = sum(header + row * len(items) for _, _, items in groups) + gap * (len(groups) - 1)
    y = 1.0 - (1.0 - total) / 2
    for title, face, items in groups:
        height = header + row * len(items)
        ax.add_patch(FancyBboxPatch((0.02, y - height), 0.96, height,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="white", edgecolor=LINE, linewidth=1.1))
        ax.add_patch(FancyBboxPatch((0.02, y - header), 0.96, header,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor=face, edgecolor=LINE, linewidth=1.1))
        ax.text(0.045, y - header / 2, title, ha="left", va="center",
                fontsize=12.5, fontweight="bold", color=INK)
        ty = y - header - row / 2
        for name, text in items:
            ax.text(0.05, ty, name, ha="left", va="center", fontsize=10.5,
                    color=BLUE, fontweight="bold", family="DejaVu Sans")
            ax.text(0.245, ty, "— " + text, ha="left", va="center", fontsize=11,
                    color=INK)
            ty -= row
        y -= height + gap
    _save(fig, "fig4_signals.png")


# ---------------------------------------------------------------- рисунок 5
def fig5_levels():
    """Шкала уровней доверия — пороги из fusion.py."""
    from lakmus.fusion import FUSION_CONFIG

    bands = FUSION_CONFIG["level_bands"]
    confirmed = FUSION_CONFIG["confirmed_score"]
    fog = FUSION_CONFIG["fog_threshold"]

    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")

    spans = [
        (0, bands["doubtful"], "Скорее\nподделка", RED_SOFT, RED),
        (bands["doubtful"], bands["likely_real"], "Сомнительно", WARN_SOFT, "#a86a12"),
        (bands["likely_real"], confirmed, "Скорее\nнастоящее", OK_SOFT, OK),
        (confirmed, 100, "Подтверждено\n(+ первоисточник)", "#cfe8db", "#0b5236"),
    ]
    for x0, x1, label, face, edge in spans:
        ax.add_patch(FancyBboxPatch((x0, 0.36), x1 - x0, 0.34,
                                    boxstyle="square,pad=0",
                                    facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text((x0 + x1) / 2, 0.53, label, ha="center", va="center",
                fontsize=11, color=INK)

    for value in (0, bands["doubtful"], bands["likely_real"], confirmed, 100):
        ax.plot([value, value], [0.34, 0.36], color=INK, linewidth=1.0)
        ax.text(value, 0.29, f"{value:g}", ha="center", va="top", fontsize=11)
    ax.text(50, 0.13, "score, баллы", ha="center", va="center",
            fontsize=11, style="italic", color=MUTED)

    ax.add_patch(FancyBboxPatch((18, 0.76), 64, 0.18,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="#e6eaf2", edgecolor=MUTED,
                                linewidth=1.1, linestyle="--"))
    ax.text(50, 0.85,
            f"«Туман»: uncertainty > {fog:.2f} или конфликт сильных сигналов —\n"
            "независимо от score",
            ha="center", va="center", fontsize=11, color=INK)
    _save(fig, "fig5_levels.png")


# ---------------------------------------------------------------- рисунок 6
def fig6_tiers():
    """Три тарифа: вердикт одинаковый, растёт только объём разбора."""
    rows = [
        ("База (0 ₽)", "вердикт, тема,\nграницы проверки", "220", "600"),
        ("Плюс (300 ₽/мес)", "+ рубрика, цитаты,\n«что делать»", "600", "1200"),
        ("Макс (5 000 ₽/мес)", "+ готовый текст\nдля пересылки", "1400", "2000"),
    ]
    fig, ax = _canvas(10.4, 3.6)
    ax.text(0.02, 0.90, "Тариф", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(0.30, 0.90, "Что открывается", fontsize=11, color=MUTED, fontweight="bold")
    ax.text(0.66, 0.90, "Лимит ответа,\nтокенов", fontsize=11, color=MUTED,
            fontweight="bold", ha="center")
    ax.text(0.85, 0.90, "Текст источника,\nсимволов", fontsize=11, color=MUTED,
            fontweight="bold", ha="center")
    y = 0.66
    faces = [BLUE_SOFT, "#c9d9f7", "#a9c3f2"]
    for (name, what, tokens, chars), face in zip(rows, faces):
        ax.add_patch(FancyBboxPatch((0.02, y), 0.96, 0.22,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    facecolor=face, edgecolor=LINE, linewidth=1.0))
        ax.text(0.035, y + 0.11, name, ha="left", va="center", fontsize=12,
                fontweight="bold", color=INK)
        ax.text(0.30, y + 0.11, what, ha="left", va="center", fontsize=11, color=INK)
        ax.text(0.66, y + 0.11, tokens, ha="center", va="center", fontsize=12, color=INK)
        ax.text(0.85, y + 0.11, chars, ha="center", va="center", fontsize=12, color=INK)
        y -= 0.27
    ax.text(0.5, 0.06, "Вердикт и оценка доверия на всех тарифах одинаковы — "
            "различается только объём объяснения",
            ha="center", va="center", fontsize=11, style="italic", color=MUTED)
    _save(fig, "fig6_tiers.png")


# ---------------------------------------------------------------- рисунок 7
def fig7_metrics():
    """Результаты на эталонном наборе: по классам кейсов."""
    report_path = PROJECT / "ml" / "data" / "eval_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    # Отчёт прогона: summary.by_class — {класс: {total, ok, accuracy}}.
    by_class = data["summary"]["by_class"]
    total_ok = data["summary"]["ok"]
    total = data["summary"]["total"]

    titles = {
        "kb": "Факты из базы", "kb_refute": "Опровержение базой",
        "negation": "Отрицания", "news_true": "Реальные новости",
        "news_fake": "Выдумки с покрытой темой", "future": "Прогнозы",
        "out_of_kb": "Вне базы", "any": "Прочее",
    }
    order = ["kb", "kb_refute", "negation", "news_true", "news_fake", "future", "out_of_kb"]
    labels, values, notes = [], [], []
    for key in order:
        stat = by_class.get(key)
        if not stat:
            continue
        labels.append(titles.get(key, key))
        values.append(100 * stat["ok"] / stat["total"])
        notes.append(f"{stat['ok']}/{stat['total']}")

    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    bars = ax.barh(range(len(labels))[::-1], values, color=[BLUE if v == 100 else WARN
                                                            for v in values],
                   height=0.6, edgecolor=INK, linewidth=0.8)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Доля верных ответов, %", fontsize=11)
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for bar, value, note in zip(bars, values, notes):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{value:.0f}%  ({note})", va="center", fontsize=10, color=INK)
    ax.set_title(f"Эталонный набор: {total_ok}/{total} ({100 * total_ok / total:.0f}%), "
                 "без ключей модели и поиска",
                 fontsize=12, color=INK, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, "fig7_metrics.png")


if __name__ == "__main__":
    fig1_search_cycle()
    fig2_sift()
    fig3_pipeline()
    fig4_signals()
    fig5_levels()
    fig6_tiers()
    fig7_metrics()
