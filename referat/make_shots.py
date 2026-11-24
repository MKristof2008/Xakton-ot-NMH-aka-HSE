"""Кадры интерфейса для документации и реферата.

Снимает реальный интерфейс с работающего сервера: текст, ссылка, скриншот,
видео и разбор. Ничего не дорисовывается — это скриншоты живого приложения.

Запуск (сервер должен быть поднят на 127.0.0.1:8010):
    py -3 referat/make_shots.py
Результат: docs/demo/*.png в рабочем репозитории и referat/fig/fig8_verdict.png
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT
OUT = PROJECT / "docs" / "demo"
FIG = Path(__file__).resolve().parent / "fig"
TMP = Path(__file__).resolve().parent / "_media"
API = "http://127.0.0.1:8010"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)


def post(path: str, data: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        API + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-Token", token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def make_media() -> tuple[Path, Path]:
    """Скриншот-картинка и ролик со следом генератора в метаданных."""
    png = TMP / "screenshot.png"
    im = Image.new("RGB", (960, 540), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 960, 90], fill=(18, 64, 155))
    d.text((28, 36), "СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года.",
           fill="white")
    d.text((28, 130), "Перешлите всем! Источники сообщают, что банкоматы", fill="black")
    d.text((28, 160), "перестанут выдавать наличные уже с воскресенья.", fill="black")
    d.rectangle([28, 210, 932, 500], outline=(169, 182, 203), width=2)
    im.save(png)

    mp4 = TMP / "clip.mp4"
    raw = b"Generated with Sora"
    body = b"\x00" + bytes([len(raw)]) + b"\x00\x00" + raw
    hdr = bytearray(8)
    hdr[0:4] = (len(body) + 8).to_bytes(4, "big")
    hdr[4:8] = b"\xa9too"
    ftyp_body = b"isom" + b"\x00\x00\x02\x00" + b"isomiso2"
    ftyp = bytearray(8)
    ftyp[0:4] = (len(ftyp_body) + 8).to_bytes(4, "big")
    ftyp[4:8] = b"ftyp"
    mp4.write_bytes(bytes(ftyp) + ftyp_body + bytes(hdr) + body)
    return png, mp4


def main() -> None:
    png, mp4 = make_media()
    tag = str(int(time.time() * 1000))
    user = post("/api/auth/register", {
        "email": f"docs{tag}@example.com",
        "password": "docs-pass-12345",
        "name": "Документация",
    })
    post("/api/upgrade", {"plan": "team"}, user["token"])
    token = user["token"]

    url = API + "/?token=" + token
    TEXT = ("СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года. "
            "Перешлите всем!")
    # Ссылка — официальный пресс-релиз: страница стабильно доступна и
    # даёт содержательный разбор, а не «страница не скачалась».
    LINK = "https://www.cbr.ru/press/pr/?file=25102024_133000key.htm"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=1.5)

        def shot(name: str, full: bool = False):
            page.screenshot(path=str(OUT / name), full_page=full)
            print("shot", name)

        def settle() -> None:
            """Дождаться конца анимации счётчика и поставить карточку в кадр."""
            page.wait_for_timeout(2200)
            card = page.locator(".verdict").first
            if card.count():
                card.scroll_into_view_if_needed()
                page.wait_for_timeout(600)

        # 1. Главный экран
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(2000)
        shot("01-home.png")

        # 2. Текст: ввод и вердикт
        page.locator("textarea").first.fill(TEXT)
        page.wait_for_timeout(400)
        shot("02-text-input.png")
        page.get_by_role("button", name="Проверить").last.click()
        page.wait_for_timeout(9000)
        settle()
        shot("03-text-verdict.png")
        # Карточка вердикта — отдельным кадром. Закреплённую шапку прячем:
        # иначе она попадает в снимок элемента и перекрывает заголовок.
        page.evaluate("""() => {
            for (const el of document.querySelectorAll('header, nav, .top, .topbar')) {
                el.style.display = 'none';
            }
        }""")
        # Шапка сайта закреплена и при снимке карточки наезжает на неё.
        page.add_style_tag(content=".header { display: none !important; }")
        page.wait_for_timeout(400)
        card = page.locator(".verdict").first
        if card.count():
            card.screenshot(path=str(FIG / "_card.png"))
            # Паспорт и кнопки обратной связи в реферате не нужны: там
            # локальный адрес и служебные элементы, разбор кончается выше.
            im = Image.open(FIG / "_card.png")
            w, h = im.size
            im.crop((0, 0, w, int(h * 0.865))).save(FIG / "fig8_verdict.png")
            print("saved fig8_verdict.png")
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.locator("textarea").first.fill(TEXT)
        page.get_by_role("button", name="Проверить").last.click()
        page.wait_for_timeout(9000)
        page.wait_for_timeout(2000)
        shot("10-reasoning.png", full=True)

        # 3. Ссылка: ввод и вердикт
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.locator("input[type='url']").first.fill(LINK)
        page.wait_for_timeout(400)
        shot("04-link-input.png")

        def link_verdict(timeout_ms: int = 45000) -> bool:
            """Дождаться карточки вердикта; сеть бывает медленной — ждём."""
            step = 1000
            waited = 0
            while waited < timeout_ms:
                page.wait_for_timeout(step)
                waited += step
                if page.locator(".verdict__word").count():
                    return True
                if page.locator(".error, [role='alert']").count():
                    return False
            return False

        page.get_by_role("button", name="Проверить").last.click()
        if not link_verdict():
            # Страница могла не скачаться с первого раза: пробуем ещё.
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Проверить").last.click()
            link_verdict()
        settle()
        shot("05-link-verdict.png")

        # 4. Скриншот
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.get_by_role("tab", name="Скриншот").click()
        page.wait_for_timeout(300)
        page.set_input_files("#check-image", str(png))
        page.wait_for_timeout(400)
        shot("06-photo-input.png")
        page.get_by_role("button", name="Проверить").last.click()
        page.wait_for_timeout(12000)
        settle()
        shot("07-photo-verdict.png")

        # 5. Видео
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.get_by_role("tab", name="Видео").click()
        page.wait_for_timeout(300)
        page.set_input_files("#check-video", str(mp4))
        page.wait_for_timeout(400)
        shot("08-video-input.png")
        page.get_by_role("button", name="Проверить").last.click()
        page.wait_for_timeout(12000)
        settle()
        shot("09-video-verdict.png")

        browser.close()

    # Крупный кадр карточки вердикта уже сохранён снимком элемента.
    # Экран автопилота — без входа: там только статистика контура.
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100},
                                device_scale_factor=1.5)
        page.goto(API + "/#autopilot", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(FIG / "fig9_autopilot.png"), full_page=False)
        browser.close()
    print("saved fig9_autopilot.png")


if __name__ == "__main__":
    main()
