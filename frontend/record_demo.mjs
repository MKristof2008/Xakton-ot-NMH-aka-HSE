import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Демо для презентации: четыре входа подряд — текст, ссылка, фото, видео.
// Записывается видео экрана и кадр после каждого вердикта.
const API = 'http://127.0.0.1:8000';
// Кадры кладём в docs/demo рядом с репозиторием, а не в абсолютный путь
// конкретной машины: скрипт должен работать после клонирования где угодно.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, '..', 'docs', 'demo');
const TMP = path.join(HERE, '..', '.demo-tmp');
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(TMP, { recursive: true });

const email = `demo${Date.now()}@example.com`;
const reg = await fetch(`${API}/api/auth/register`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password: 'demo-pass-12345', name: 'Демо' }),
}).then(r => r.json());
await fetch(`${API}/api/upgrade`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-Token': reg.token },
  body: JSON.stringify({ plan: 'team' }),
});
console.log('аккаунт:', email, '| тариф: Макс');

// Материалы
const pngPath = `${TMP}/demo-photo.png`;
const mp4Path = `${TMP}/demo-video.mp4`;
fs.writeFileSync(pngPath, Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6300010000050001',
  'hex',
));
function mp4(tool) {
  const raw = Buffer.from(tool, 'utf8');
  const body = Buffer.concat([Buffer.from([0, raw.length, 0, 0]), raw]);
  const hdr = Buffer.alloc(8);
  hdr.writeUInt32BE(body.length + 8, 0);
  hdr.write('\xa9too', 4, 'latin1');
  const ftypBody = Buffer.concat([
    Buffer.from('isom', 'latin1'), Buffer.from([0, 0, 2, 0]),
    Buffer.from('isomiso2', 'latin1'),
  ]);
  const ftypHdr = Buffer.alloc(8);
  ftypHdr.writeUInt32BE(ftypBody.length + 8, 0);
  ftypHdr.write('ftyp', 4, 'latin1');
  return Buffer.concat([ftypHdr, ftypBody, hdr, body]);
}
fs.writeFileSync(mp4Path, mp4('Generated with Sora'));

const browser = await chromium.launch({
  headless: false, slowMo: 240, args: ['--start-maximized'],
});
const ctx = await browser.newContext({
  viewport: null,
  recordVideo: { dir: TMP, size: { width: 1600, height: 900 } },
});
await ctx.addInitScript(t => {
  try { localStorage.setItem('provereno_token', t); } catch {}
}, reg.token);
const page = await ctx.newPage();

async function shot(name) {
  await page.screenshot({ path: path.join(OUT, name) });
  console.log('  кадр:', name);
}

async function waitVerdict(maxSec = 35) {
  for (let i = 0; i < maxSec; i++) {
    await page.waitForTimeout(1000);
    const t = await page.locator('body').innerText();
    if (/ПОЧЕМУ ТАКОЙ ВЕРДИКТ|Почему такой вердикт|ИЗ ЧЕГО СЛОЖИЛСЯ ВЕРДИКТ/i.test(t)) return t;
  }
  return await page.locator('body').innerText();
}

function firstLines(text, n = 5) {
  return text.split('\n').map(s => s.trim()).filter(Boolean).slice(0, n).join(' · ');
}

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
await page.waitForTimeout(1200);
await shot('01-home.png');

// ---------- 1. ТЕКСТ ----------
console.log('\n1. ТЕКСТ');
await page.getByRole('tab', { name: 'Текст' }).click();
await page.waitForTimeout(500);
await page.locator('#check-text').fill(
  'СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года. Перешлите всем!',
);
await page.waitForTimeout(700);
await shot('02-text-input.png');
await page.locator('.btn--primary').click();
let body = await waitVerdict();
await page.waitForTimeout(1600);
await shot('03-text-verdict.png');
console.log('  вердикт:', firstLines(body, 4));

// ---------- 2. ССЫЛКА ----------
console.log('\n2. ССЫЛКА');
await page.locator('#check-link').fill('https://www.cbr.ru/press/pr/');
await page.locator('#check-text').fill(
  'ЦБ поднял ключевую ставку до 21% годовых на заседании 25 октября 2024 года.',
);
await page.waitForTimeout(800);
await shot('04-link-input.png');
await page.locator('.btn--primary').click();
body = await waitVerdict();
await page.waitForTimeout(1600);
await shot('05-link-verdict.png');
console.log('  вердикт:', firstLines(body, 4));

// ---------- 3. ФОТО ----------
console.log('\n3. ФОТО');
await page.locator('#check-link').fill('');
await page.getByRole('tab', { name: 'Скриншот' }).click();
await page.waitForTimeout(600);
await page.locator('#check-image').setInputFiles(pngPath);
await page.waitForTimeout(900);
await shot('06-photo-input.png');
await page.locator('.btn--primary').click();
body = await waitVerdict();
await page.waitForTimeout(1600);
await shot('07-photo-verdict.png');
console.log('  вердикт:', firstLines(body, 4));

// ---------- 4. ВИДЕО ----------
console.log('\n4. ВИДЕО');
await page.getByRole('tab', { name: 'Видео' }).click();
await page.waitForTimeout(600);
await page.locator('#check-video').setInputFiles(mp4Path);
await page.waitForTimeout(900);
await shot('08-video-input.png');
await page.locator('.btn--primary').click();
body = await waitVerdict();
await page.waitForTimeout(1600);
await shot('09-video-verdict.png');
console.log('  вердикт:', firstLines(body, 5));

// Финальный кадр — рубрика.
const reasoning = page.getByText('Почему такой вердикт');
if (await reasoning.count()) {
  await reasoning.first().scrollIntoViewIfNeeded();
  await page.waitForTimeout(1500);
  await shot('10-reasoning.png');
}

await page.waitForTimeout(1500);
const videoPath = await page.video().path();
await ctx.close();
await browser.close();

const final = path.join(OUT, 'demo-walkthrough.webm');
fs.copyFileSync(videoPath, final);
console.log('\nВИДЕО:', final, Math.round(fs.statSync(final).size / 1024), 'КБ');
