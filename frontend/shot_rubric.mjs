import { chromium } from '@playwright/test';
import os from 'node:os';

// Демо рубрики в браузере: открываем проверку на тарифе «Макс» и снимаем экран.
const API = 'http://127.0.0.1:8000';
// Временная папка системы: без привязки к имени пользователя.
const SHOTS = os.tmpdir();

const reg = await fetch(`${API}/api/auth/register`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: `shot${Date.now()}@example.com`,
    password: 'demo-pass-12345',
    name: 'Демо',
  }),
}).then(r => r.json());

await fetch(`${API}/api/upgrade`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-Token': reg.token },
  body: JSON.stringify({ plan: 'team' }),
});

const browser = await chromium.launch({ headless: false, slowMo: 200, args: ['--start-maximized'] });
const ctx = await browser.newContext({ viewport: null });
await ctx.addInitScript(t => {
  try { localStorage.setItem('provereno_token', t); } catch {}
}, reg.token);
const page = await ctx.newPage();

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
await page.locator('#check-text').fill(
  'СРОЧНО! ЦБ запретил наличные с 1 марта 2026 года. Перешлите всем!',
);
await page.locator('.btn--primary').click();
await page.waitForTimeout(11000);
await page.screenshot({ path: `${SHOTS}/rubric-verdict.png` });

// Прокручиваем к рубрике.
const block = page.getByText('Почему такой вердикт');
if (await block.count()) {
  await block.first().scrollIntoViewIfNeeded();
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${SHOTS}/rubric-criteria.png` });
  console.log('Рубрика найдена на странице');
} else {
  console.log('Рубрика на странице НЕ найдена');
}
const body = await page.locator('body').innerText();
console.log('Есть «Тема»:', body.includes('Тема'));
console.log('Есть «Основание»:', body.includes('Основание'));
console.log('Есть «Правила темы»:', body.includes('Правила темы'));

await new Promise(() => {});
