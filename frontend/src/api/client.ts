import { apiBaseUrl } from '../config';
import type {
  ApiKey,
  AutopilotRecentResponse,
  AutopilotStats,
  ApiKeyUsage,
  CheckResponse,
  FeedResponse,
  HistoryItem,
  Payment,
  Me,
  PricingResponse,
  Stats,
  TopResponse,
} from '../types';

const TOKEN_KEY = 'provereno_token';

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, message: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

function readToken(): string | null {
  // ?token= нужен только для демо-скриншотов и проверки тарифа в браузере.
  try {
    const fromUrl = new URLSearchParams(window.location.search).get('token');
    if (fromUrl) {
      localStorage.setItem(TOKEN_KEY, fromUrl);
      return fromUrl;
    }
  } catch {
    /* ignore */
  }
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* приватный режим — просто работаем без сохранения */
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = readToken();
  if (token) headers.set('X-Token', token);
  if (init?.body && !headers.has('Content-Type') && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, { ...init, headers });
  } catch {
    throw new ApiError('NETWORK', 'Сервер недоступен. Проверьте, что он запущен.', 0);
  }

  if (!response.ok) {
    let code = `HTTP_${response.status}`;
    let message = 'Запрос не выполнен';
    try {
      const body = (await response.json()) as { error?: { code?: string; message?: string } };
      if (body?.error?.code) code = body.error.code;
      if (body?.error?.message) message = body.error.message;
    } catch {
      /* тело не JSON */
    }
    throw new ApiError(code, message, response.status);
  }
  return (await response.json()) as T;
}

/** Регистрация: без аккаунта проверок нет, токен живёт в localStorage. */
export async function register(
  email: string,
  password: string,
  name = '',
): Promise<Me> {
  const user = await request<Me & { token: string }>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, name }),
  });
  writeToken(user.token);
  return user;
}

export async function login(email: string, password: string): Promise<Me> {
  const user = await request<Me & { token: string }>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  writeToken(user.token);
  return user;
}

export async function logout(): Promise<void> {
  try {
    await request('/auth/logout', { method: 'POST' });
  } catch {
    /* сервер недоступен — токен всё равно убираем локально */
  }
  clearToken();
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* приватный режим */
  }
}

/** Есть ли сохранённая сессия. Без неё интерфейс показывает вход. */
/** Гостевой токен для внутренних вызовов, если человек ещё не вошёл. */
export async function ensureUser(): Promise<void> {
  if (readToken()) return;
  const user = await request<{ id: string; token: string }>('/users', {
    method: 'POST',
  });
  writeToken(user.token);
}

export function hasSession(): boolean {
  return Boolean(readToken());
}

export async function getMe(): Promise<Me> {
  return request<Me>('/me');
}

/**
 * Проверка сообщения. Можно передать текст, ссылку, скриншот или видео —
 * все входы идут одним контрактом. Ссылка нужна для сценария «поделиться
 * из мессенджера», файл — для скриншота или ролика.
 */
export async function runCheck(
  text: string,
  extra: { url?: string; image?: File; video?: File } = {},
): Promise<CheckResponse> {
  if (extra.image || extra.video) {
    const body = new FormData();
    if (text) body.append('text', text);
    if (extra.url) body.append('url', extra.url);
    if (extra.image) body.append('image', extra.image);
    if (extra.video) body.append('video', extra.video);
    return request<CheckResponse>('/check', { method: 'POST', body });
  }
  return request<CheckResponse>('/check', {
    method: 'POST',
    body: JSON.stringify({ text, url: extra.url || null }),
  });
}

export async function getFeed(hours = 168): Promise<FeedResponse> {
  return request<FeedResponse>(`/feed?hours=${hours}`);
}

export async function getAutopilotStats(): Promise<AutopilotStats> {
  return request<AutopilotStats>('/autopilot/stats');
}

export async function getAutopilotRecent(
  limit = 30,
  label?: 'fake' | 'real' | 'uncertain',
): Promise<AutopilotRecentResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (label) query.set('label', label);
  return request<AutopilotRecentResponse>(`/autopilot/recent?${query.toString()}`);
}

export async function sendFeedback(payload: {
  check_id: string;
  verdict_ok: boolean;
  expected?: string;
  note?: string;
}): Promise<{ ok: boolean }> {
  return request('/feedback', { method: 'POST', body: JSON.stringify(payload) });
}

export async function getApiKeys(): Promise<{ items: ApiKey[]; usage: ApiKeyUsage }> {
  return request('/keys');
}

export async function createApiKey(label = ''): Promise<ApiKey> {
  return request('/keys', { method: 'POST', body: JSON.stringify({ label }) });
}

export async function revokeApiKey(id: string): Promise<{ ok: boolean }> {
  return request(`/keys/${id}`, { method: 'DELETE' });
}

export async function getTop(hours = 48): Promise<TopResponse> {
  return request<TopResponse>(`/top?hours=${hours}`);
}

/** История проверок текущего пользователя. */
export async function getHistory(limit = 50): Promise<{ items: HistoryItem[] }> {
  return request<{ items: HistoryItem[] }>(`/history?limit=${limit}`);
}

export async function getStats(): Promise<Stats> {
  return request<Stats>('/stats');
}

export async function getPricing(): Promise<PricingResponse> {
  return request<PricingResponse>('/pricing');
}

export async function upgrade(plan = 'pro'): Promise<void> {
  await request('/upgrade', { method: 'POST', body: JSON.stringify({ plan }) });
}

export async function createPayment(
  plan: string,
  method: 'card' | 'sbp' | 'invoice' = 'card',
): Promise<Payment> {
  return request<Payment>('/payments', {
    method: 'POST',
    body: JSON.stringify({ plan, method }),
  });
}

export async function confirmPayment(paymentId: string): Promise<void> {
  await request('/payments/confirm', {
    method: 'POST',
    body: JSON.stringify({ payment_id: paymentId }),
  });
}
