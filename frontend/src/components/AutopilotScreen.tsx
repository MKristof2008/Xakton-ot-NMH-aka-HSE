import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { getAutopilotRecent, getAutopilotStats } from '../api/client';
import type { AutopilotItem, AutopilotRecentResponse, AutopilotStats } from '../types';
import { plural } from '../lib/plural';

/**
 * Автопилот: фоновый контур, который сам собирает сообщения из сети,
 * проверяет их и дообучает модель на подтверждённом.
 *
 * Страница читает только сводку и последние размеченные материалы.
 * Если автопилот не запускался, честно показывает, что данных пока нет.
 */
type Filter = 'all' | 'fake' | 'real' | 'uncertain';

const FILTERS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'Все' },
  { id: 'fake', label: 'Фейк' },
  { id: 'real', label: 'Правда' },
  { id: 'uncertain', label: 'Неясно' },
];

const LABEL_TONE: Record<AutopilotItem['label'], string> = {
  fake: 'bad',
  real: 'ok',
  uncertain: 'fog',
};

const LABEL_TEXT: Record<AutopilotItem['label'], string> = {
  fake: 'Фейк',
  real: 'Правда',
  uncertain: 'Неясно',
};

function pct(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return `${Math.round(value * 100)}%`;
}

function formatMoment(ts: number | null | undefined): string {
  if (!ts) return '—';
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function AutopilotScreen() {
  const [stats, setStats] = useState<AutopilotStats | null>(null);
  const [recent, setRecent] = useState<AutopilotRecentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  useEffect(() => {
    let alive = true;
    getAutopilotStats()
      .then((res) => {
        if (alive) setStats(res);
      })
      .catch(() => {
        if (alive) setError('Не удалось загрузить статистику автопилота.');
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    setRecent(null);
    getAutopilotRecent(30, filter === 'all' ? undefined : filter)
      .then((res) => {
        if (alive) setRecent(res);
      })
      .catch(() => {
        if (alive) setError('Не удалось загрузить размеченные материалы.');
      });
    return () => {
      alive = false;
    };
  }, [filter]);

  const model = stats?.model;
  const items = recent?.items ?? [];

  return (
    <div className="view">
      <section className="hero">
        <span className="eyebrow">Фоновый контур</span>
        <h1 className="hero__title">
          Авто<em>пилот</em>
        </h1>
        <p className="hero__lead">
          Собирает сообщения из лент и каналов, проверяет их по базе знаний
          и дообучает модель на том, что подтвердилось. Метка ставится только
          при внешнем основании — опровержение фактчекеров, проверка утверждений
          или цитата из источника.
        </p>
      </section>

      {error && (
        <p className="alert alert--bad" role="alert">
          <span aria-hidden="true">!</span>
          {error}
        </p>
      )}

      {!stats && !error && <AutopilotSkeleton />}

      {stats && !stats.available && (
        <div className="empty">
          <span className="empty__mark" aria-hidden="true">
            §
          </span>
          <h3>Автопилот ещё не запускался</h3>
          <p>
            Запустите <code>.\autopilot\RUN_AUTOPILOT.ps1 -Once</code> — собранные
            и проверенные материалы появятся здесь.
          </p>
        </div>
      )}

      {stats && stats.available && (
        <>
          <div className="ap-tiles">
            <div className="ap-tile">
              <span className="ap-tile__n">{stats.items_total}</span>
              <span className="ap-tile__k">собрано материалов</span>
            </div>
            <div className="ap-tile">
              <span className="ap-tile__n">{stats.fake}</span>
              <span className="ap-tile__k">признано фейком</span>
            </div>
            <div className="ap-tile">
              <span className="ap-tile__n">{stats.real}</span>
              <span className="ap-tile__k">подтверждено</span>
            </div>
            <div className="ap-tile">
              <span className="ap-tile__n">{stats.pending}</span>
              <span className="ap-tile__k">в очереди на проверку</span>
            </div>
          </div>

          <div className="ap-panels">
            <section className="ap-panel">
              <h2 className="ap-panel__title">Модель</h2>
              {model ? (
                <dl className="ap-kv">
                  <dt>Версия</dt>
                  <dd>{model.version ?? '—'}</dd>
                  <dt>F1</dt>
                  <dd>{model.f1?.toFixed(3) ?? '—'}</dd>
                  <dt>Precision</dt>
                  <dd>{model.precision_score?.toFixed(3) ?? '—'}</dd>
                  <dt>Recall</dt>
                  <dd>{model.recall?.toFixed(3) ?? '—'}</dd>
                  <dt>Примеров</dt>
                  <dd>{model.n_train ?? '—'}</dd>
                </dl>
              ) : (
                <p className="ap-note">Модель ещё не переобучалась автопилотом.</p>
              )}
            </section>

            <section className="ap-panel">
              <h2 className="ap-panel__title">Расход модели</h2>
              <dl className="ap-kv">
                <dt>За час</dt>
                <dd>{stats.llm.last_hour} вызовов</dd>
                <dt>За сутки</dt>
                <dd>{stats.llm.today} вызовов</dd>
                <dt>Неясно</dt>
                <dd>{stats.uncertain}</dd>
                <dt>В архиве</dt>
                <dd>{stats.archived}</dd>
              </dl>
            </section>

            <section className="ap-panel">
              <h2 className="ap-panel__title">Источники</h2>
              {stats.sources.length > 0 ? (
                <ul className="ap-sources">
                  {stats.sources.slice(0, 6).map((src) => (
                    <li key={src.source_id}>
                      <span className="ap-sources__name">{src.source_name || src.source_id}</span>
                      <span className="ap-sources__role">
                        {src.role === 'factcheck' ? 'фактчек' : 'новости'}
                      </span>
                      <span className="ap-sources__n">{src.n}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="ap-note">Пока ничего не собрано.</p>
              )}
            </section>
          </div>

          <section className="ap-recent">
            <div className="feed-summary">
              {FILTERS.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className={`feed-filter${filter === f.id ? ' is-active' : ''}`}
                  onClick={() => setFilter(f.id)}
                >
                  {f.label}
                </button>
              ))}
              <span className="feed-summary__note">
                {recent ? `${recent.total} размечено` : 'загрузка…'}
              </span>
            </div>

            {!recent && <AutopilotSkeleton />}

            {recent && items.length === 0 && (
              <div className="empty">
                <span className="empty__mark" aria-hidden="true">
                  §
                </span>
                <h3>Размеченных материалов пока нет</h3>
                <p>
                  Автопилот собрал {stats.items_total}{' '}
                  {plural(stats.items_total, ['материал', 'материала', 'материалов'])},
                  но метку пока не поставил: нужен внешний источник, который
                  подтверждает или опровергает. Подключите ключ модели в{' '}
                  <code>ml/.env</code>, чтобы включить проверку утверждений.
                </p>
              </div>
            )}

            {items.length > 0 && (
              <ol className="feed-list">
                {items.map((item, i) => (
                  <li
                    key={item.id}
                    className="feed-item"
                    style={{ '--i': i } as CSSProperties}
                  >
                    <div className="feed-item__head">
                      <span className={`tag tag--${LABEL_TONE[item.label]}`}>
                        {LABEL_TEXT[item.label]}
                      </span>
                      <span className="feed-item__date">
                        {item.source_name || item.source_id} · {pct(item.label_confidence)}
                      </span>
                    </div>
                    <p className="feed-item__text">
                      {item.title || item.excerpt}
                    </p>
                    {item.label_reason && (
                      <p className="ap-reason">{item.label_reason}</p>
                    )}
                    <div className="feed-item__foot">
                      <span>{formatMoment(item.verified_at)}</span>
                      {item.url && (
                        <a
                          className="feed-item__link"
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Источник
                          <span aria-hidden="true">↗</span>
                        </a>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </>
      )}
    </div>
  );
}

function AutopilotSkeleton() {
  return (
    <div className="feed-list" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <div className="feed-item feed-item--skeleton" key={i}>
          <span className="sk sk--line" />
          <span className="sk sk--line sk--short" />
        </div>
      ))}
    </div>
  );
}
