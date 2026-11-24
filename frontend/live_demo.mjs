import { chromium } from '@playwright/test';
import fs from 'node:fs';
import os from 'node:os';

const API = 'http://127.0.0.1:8000';
// Временная папка системы: скрипт не должен зависеть от имени пользователя.
const SHOTS = os.tmpdir();

// 1. Аккаунт создаём через API — так демо не зависит от формы входа.
const email = `demo${Date.now()}@example.com`;
const reg = await fetch(`${API}/api/auth/register`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password: 'demo-pass-12345', name: 'Демо' }),
});
const user = await reg.json();
console.log('Аккаунт:', user.email || user, '| токен:', (user.token || '').slice(0, 10) + '…');

// 2. Тариф «Макс» для полного разбора в демо.
const upgraded = await fetch(`${API}/api/upgrade`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'X-Token': user.token },
  body: JSON.stringify({ plan: 'team' }),
}).then(r => r.json());
console.log('Тариф:', upgraded);

const pngPath = `${SHOTS}/revizor-demo.png`;
fs.writeFileSync(pngPath, Buffer.from(
  '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6300010000050001',
  'hex',
));

const browser = await chromium.launch({ headless: false, slowMo: 220, args: ['--start-maximized'] });
const context = await browser.newContext({ viewport: null });
await context.addInitScript(token => {
  try { localStorage.setItem('provereno_token', token); } catch {}
}, user.token);
const page = await context.newPage();
page.on('pageerror', e => console.log('PAGE ERROR:', e.message));

async function show(label, file) {
  const text = await page.locator('body').innerText();
  console.log(`\n===== ${label} =====\n${text.slice(-1500)}`);
  await page.screenshot({ path: `${SHOTS}/${file}` });
}

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
await show('Главная', 'r1-home.png');

// Текст: тариф «Макс» — полный разбор, «что делать» и готовая пересылка.
await page.locator('#check-text').fill(
  'СРОЧНО! ЦБ запретил снятие наличных с 1 марта 2026 года. Перешлите всем!',
);
await page.locator('.btn--primary').click();
await page.waitForTimeout(12000);
await show('ТЕКСТ — тариф Макс', 'r2-text-max.png');

// Скриншот.
await page.getByRole('tab', { name: 'Скриншот' }).click();
await page.locator('#check-image').setInputFiles(pngPath);
await page.locator('.btn--primary').click();
await page.waitForTimeout(9000);
await show('ФОТО — тариф Макс', 'r3-photo-max.png');

console.log('\nБраузер оставлен открытым.');
await new Promise(() => {});
