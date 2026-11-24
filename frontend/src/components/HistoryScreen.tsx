import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { getHistory } from '../api/client';
import type { HistoryItem } from '../types';
import { toneForLevel } from '../lib/verdict';
import { plural } from '../lib/plural';

/**
 * История проверок. Главред назвал её базовой вещью: человек проверил
 * новость вчера и хочет вернуться к вердикту, а не искать сообщение заново.
 */
export function HistoryScreen() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    getHistory()
      .then((res) => {
        if (alive) setItems(res.items);
      })
      .catch(() => {
        if (alive) setError('Не удалось загрузить историю.');
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="view">
      <section className="hero">
        <span className="eyebrow">Ваши проверки</span>
        <h1 className="hero__title">
          История <em>проверок</em>
        </h1>
        <p className="hero__lead">
          Всё, что вы проверяли. Откройте запись — вердикт и паспорт утверждения
          останутся по той же ссылке.
        </p>
      </section>

      {error && (
        <p className="alert alert--bad" role="alert">
          <span aria-hidden="true">!</span>
          {error}
        </p>
      )}

      {items === null && !error && <HistorySkeleton />}

      {items && items.length === 0 && (
        <div className="empty">
          <span className="empty__mark" aria-hidden="true">
            §
          </span>
          <h3>Пока пусто</h3>
          <p>Проверьте первое сообщение — оно появится здесь.</p>
        </div>
      )}

      {items && items.length > 0 && (
        <ol className="top-list">
          {items.map((item, i) => (
            <li
              key={item.id}
              className="top-item top-item--link"
              style={{ '--i': i } as CSSProperties}
            >
              <a
                className="top-item__anchor"
                href={item.passport}
                target="_blank"
                rel="noreferrer noopener"
              >
                {item.title ? (
                  <>
                    <p className="top-item__text">{item.title}</p>
                    <p className="history-item__note">Проверена статья по ссылке</p>
                  </>
                ) : (
                  <p className="top-item__text">{item.text}</p>
                )}
                <div className="top-item__meta">
                  <span className={`tag tag--${toneForLevel(item.level)}`}>
                    {item.level_label}
                  </span>
                  <span>{formatWhen(item.created_at)}</span>
                </div>
              </a>
            </li>
          ))}
        </ol>
      )}

      {items && items.length > 0 && (
        <p className="hint">
          Показано {items.length}{' '}
          {plural(items.length, ['проверка', 'проверки', 'проверок'])}.
        </p>
      )}
    </div>
  );
}

/** Дата проверки коротко: «сегодня, 14:20», «вчера, 09:15», иначе дата. */
function formatWhen(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const time = date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
  if (sameDay) return `сегодня, ${time}`;
  if (date.toDateString() === yesterday.toDateString()) return `вчера, ${time}`;
  return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' });
}

function HistorySkeleton() {
  return (
    <div className="skeleton" role="status" aria-live="polite">
      <span className="skel skel--sub" />
      <span className="skel skel--sub-short" />
      <span className="skel skel--sub" />
      <span className="visually-hidden">Загружаем историю</span>
    </div>
  );
}
