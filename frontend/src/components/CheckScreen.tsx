import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, getMe, runCheck } from '../api/client';
import type { CheckResponse, Me } from '../types';
import { maxTextLength } from '../config';
import { useReducedMotion } from '../hooks/useReducedMotion';
import { EXAMPLES } from '../lib/examples';
import { VerdictCard } from './VerdictCard';
import { VerdictSkeleton } from './VerdictSkeleton';

interface Props {
  me: Me | null;
  onMe: (me: Me) => void;
  /** Попросить войти: вызывается в момент проверки, если аккаунта нет. */
  onNeedAuth?: (reason: string) => void;
}

/** Минимальная выдержка скелетона: вердикт должен ощущаться как решение системы. */
const MIN_SKELETON_MS = 620;

/** Ссылка из мессенджера: «поделиться» вставляет адрес в буфер. */
const LINK_RE = /^https?:\/\/\S+$/i;

/** Что можно приложить к проверке: текст, скриншот или видео. */
type Mode = 'text' | 'image' | 'video';

const FILE_TABS: { id: Mode; label: string }[] = [
  { id: 'text', label: 'Текст' },
  { id: 'image', label: 'Скриншот' },
  { id: 'video', label: 'Видео' },
];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** sessionStorage может быть недоступен (приватный режим) — не роняем экран. */
function readPreset(): string | null {
  try {
    return sessionStorage.getItem('preset_text');
  } catch {
    return null;
  }
}

function clearPreset(): void {
  try {
    sessionStorage.removeItem('preset_text');
  } catch {
    /* нечего чистить */
  }
}

export function CheckScreen({ me, onMe, onNeedAuth }: Props) {
  // ?q=... кладётся в sessionStorage до рендера — забираем его ровно один раз.
  const [preset] = useState(readPreset);
  const [text, setText] = useState(() => preset ?? '');
  // Отдельное поле для ссылки: человек жмёт «поделиться» и вставляет адрес,
  // не задумываясь, что это не текст сообщения.
  const [link, setLink] = useState('');
  const [mode, setMode] = useState<Mode>('text');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<CheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const reduced = useReducedMotion();
  const resultRef = useRef<HTMLDivElement>(null);
  // Индикатор вкладок: отдельный элемент, который едет transform'ом, а не перерисовкой.
  const tabsRef = useRef<HTMLDivElement>(null);
  const [indicator, setIndicator] = useState({ x: 0, y: 0, w: 0, h: 0, ready: false });
  // Текст, который ждёт входа: после авторизации проверка запустится сама.
  const pendingText = useRef<string | null>(null);

  // Пересчитываем положение индикатора при смене вкладки, размера окна и загрузке шрифтов.
  useEffect(() => {
    const wrap = tabsRef.current;
    if (!wrap) return;
    const measure = () => {
      const active = wrap.querySelector<HTMLElement>('.input-tab.is-active');
      if (!active) return;
      setIndicator((prev) => ({
        x: active.offsetLeft,
        y: active.offsetTop,
        w: active.offsetWidth,
        h: active.offsetHeight,
        ready: prev.ready,
      }));
    };
    measure();
    // первый замер без анимации, дальше — плавный переезд
    const raf = requestAnimationFrame(() => setIndicator((p) => ({ ...p, ready: true })));
    window.addEventListener('resize', measure);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', measure);
    };
  }, [mode]);

  const submit = useCallback(
    async (raw?: string) => {
      const value = (raw ?? text).trim();
      const image = mode === 'image' ? file : null;
      const clip = mode === 'video' ? file : null;
      // Ссылку можно проверить и без текста — бэкенд вытащит его со страницы.
      const onlyUrl = !value && !image && !clip && LINK_RE.test(link.trim());

      if (mode === 'image' && !image) {
        setError('Выберите скриншот: PNG, JPEG, WebP или GIF до 8 МБ.');
        return;
      }
      if (mode === 'video' && !clip) {
        setError('Выберите видео: MP4 или MOV до 25 МБ.');
        return;
      }
      if (!onlyUrl && !image && !clip && value.length < 8) {
        setError(
          link.trim()
            ? 'Ссылка должна начинаться с http:// или https://'
            : 'Нужно хотя бы 8 символов или ссылка — иначе проверять нечего.',
        );
        return;
      }

      // Аккаунта нет — не гоняем запрос в пустоту, а просим войти.
      // Текст остаётся в поле: после входа человек нажмёт кнопку снова
      // и проверка пройдёт по уже введённому сообщению.
      if (!me && onNeedAuth) {
        pendingText.current = value;
        onNeedAuth('Чтобы проверить сообщение, войдите или создайте аккаунт. Текст уже сохранён.');
        return;
      }

      setBusy(true);
      setError(null);
      setResult(null);

      try {
        const [data] = await Promise.all([
          runCheck(value, {
            url: link.trim() || undefined,
            image: image ?? undefined,
            video: clip ?? undefined,
          }),
          reduced ? Promise.resolve() : delay(MIN_SKELETON_MS),
        ]);
        setResult(data);
        if (typeof data.quota_left === 'number') {
          // Бэкенд отдаёт quota_left только на free — обновляем счётчик, не трогая тариф.
          onMe({ ...(me ?? { id: '', plan: 'free' }), quota_left: data.quota_left });
        }
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err.code === 'QUOTA_EXCEEDED' ? err.message : `Ошибка: ${err.message}`);
        } else {
          setError('Не удалось проверить. Попробуйте ещё раз.');
        }
      } finally {
        setBusy(false);
      }
    },
    [text, link, mode, file, me, onMe, reduced],
  );

  // ?q=... сразу запускает проверку — так удобно делиться вердиктом.
  useEffect(() => {
    if (!preset) return;
    clearPreset();
    void submit(preset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Гость нажал «Проверить», вошёл в аккаунт — доделываем проверку сами,
  // чтобы не заставлять нажимать кнопку второй раз.
  const wasAuthed = Boolean(me);
  useEffect(() => {
    if (wasAuthed || !me || !pendingText.current) return;
    const text = pendingText.current;
    pendingText.current = null;
    void submit(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [me]);

  // Вердикт появился — мягко подводим его в кадр на телефоне.
  useEffect(() => {
    if (!result || !resultRef.current) return;
    const box = resultRef.current.getBoundingClientRect();
    if (box.top > window.innerHeight - 80) {
      resultRef.current.scrollIntoView({
        behavior: reduced ? 'auto' : 'smooth',
        block: 'nearest',
      });
    }
  }, [result, reduced]);

  /** После подписки перепроверяем тот же материал, чтобы разбор раскрылся без перезагрузки. */
  const handleUpgraded = useCallback(async () => {
    try {
      onMe(await getMe());
    } catch {
      /* тариф обновится при следующем запросе */
    }
    const trimmed = text.trim();
    const image = mode === 'image' ? file : null;
    const clip = mode === 'video' ? file : null;
    const onlyUrl = !trimmed && !image && !clip && LINK_RE.test(link.trim());
    if (!onlyUrl && !image && !clip && trimmed.length < 8) return;

    setBusy(true);
    setResult(null);
    try {
      setResult(
        await runCheck(trimmed, {
          url: link.trim() || undefined,
          image: image ?? undefined,
          video: clip ?? undefined,
        }),
      );
    } catch {
      setError('Подписка включена, но разбор не загрузился. Проверьте материал ещё раз.');
    } finally {
      setBusy(false);
    }
  }, [text, link, mode, file, onMe]);

  const chars = text.length;
  const nearLimit = chars > maxTextLength * 0.9;

  return (
    <div className="view">
      <section className="hero">
        <span className="eyebrow">Проверка новостей</span>
        <h1 className="hero__title">
          Фейк или правда — <em>за одну проверку</em>
        </h1>
        <p className="hero__lead">
          Вставьте сообщение из чата, поста или пересылки. Мы сверим его с курируемой базой
          фактов и покажем, откуда взялась новость.
        </p>
      </section>

      <section className="panel" aria-labelledby="check-form-title">
        <h2 className="visually-hidden" id="check-form-title">
          Проверить сообщение
        </h2>
        {/* Ссылка из мессенджера — отдельное поле: главред спрашивал
            именно про этот сценарий дважды. Текст можно не вставлять. */}
        <div className="field">
          <label className="field__label" htmlFor="check-link">
            <span>Ссылка на новость</span>
            <span className="field__hint">необязательно</span>
          </label>
          <input
            id="check-link"
            className="input"
            type="url"
            inputMode="url"
            autoComplete="off"
            spellCheck={false}
            value={link}
            placeholder="https://t.me/… или адрес сайта"
            onChange={(e) => {
              setLink(e.target.value);
              setError(null);
            }}
          />
        </div>

        <div className="input-tabs" role="tablist" aria-label="Что приложить" ref={tabsRef}>
          {/* Индикатор активной вкладки: едет transform'ом, раскладка не прыгает. */}
          <span
            className={`input-tab-ind${indicator.ready ? '' : ' is-jump'}`}
            aria-hidden="true"
            style={
              {
                '--x': `${indicator.x}px`,
                '--y': `${indicator.y}px`,
                '--w': `${indicator.w}px`,
                '--h': `${indicator.h}px`,
              } as React.CSSProperties
            }
          />
          {FILE_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={mode === tab.id}
              className={`input-tab${mode === tab.id ? ' is-active' : ''}`}
              onClick={() => {
                setMode(tab.id);
                setFile(null);
                setError(null);
                setResult(null);
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="field">
          {mode === 'text' && (
            <>
              <label className="field__label" htmlFor="check-text">
                <span>Текст сообщения</span>
                <span className="field__hint">до {maxTextLength} символов</span>
              </label>
              <textarea
                id="check-text"
                className="textarea"
                rows={5}
                maxLength={maxTextLength}
                value={text}
                aria-describedby="check-counter"
                placeholder="Например: СРОЧНО! ЦБ запретил наличные с 1 марта!"
                onChange={(e) => {
                  setText(e.target.value);
                  setError(null);
                }}
              />
            </>
          )}

          {mode === 'image' && (
            <>
              <label className="field__label" htmlFor="check-image">
                <span>Скриншот</span>
                <span className="field__hint">PNG, JPEG, WebP, GIF · до 8 МБ</span>
              </label>
              <div className="file-drop">
                <input
                  id="check-image"
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                    setError(null);
                  }}
                />
                <span className="file-drop__btn">Выбрать файл</span>
                <span className={`file-drop__name${file ? ' file-drop__name--chosen' : ''}`}>
                  {file ? `${file.name} · ${Math.round(file.size / 1024)} КБ` : 'Файл не выбран'}
                </span>
              </div>
              <p className="field__note">
                Проверяем метаданные файла, следы генерации и монтажа, контекст публикации.
              </p>
            </>
          )}

          {mode === 'video' && (
            <>
              <label className="field__label" htmlFor="check-video">
                <span>Видео</span>
                <span className="field__hint">MP4, MOV · до 25 МБ</span>
              </label>
              <div className="file-drop">
                <input
                  id="check-video"
                  type="file"
                  accept="video/mp4,video/quicktime"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                    setError(null);
                  }}
                />
                <span className="file-drop__btn">Выбрать файл</span>
                <span className={`file-drop__name${file ? ' file-drop__name--chosen' : ''}`}>
                  {file ? `${file.name} · ${Math.round(file.size / 1024)} КБ` : 'Файл не выбран'}
                </span>
              </div>
              <p className="field__note">
                Читаем метаданные ролика, ищем следы генерации и монтажа, проверяем,
                не является ли ролик статичной картинкой. Звук и речь не анализируются.
              </p>
            </>
          )}

          <div className="form__foot">
            <span
              className={`counter${nearLimit ? ' counter--near' : ''}`}
              id="check-counter"
            >
              {mode === 'text' ? `${chars} / ${maxTextLength}` : ''}
            </span>
            <button
              className="btn btn--primary"
              type="button"
              onClick={() => void submit()}
              disabled={busy}
              aria-busy={busy}
            >
              {busy ? (
                <>
                  <span className="btn__spinner" aria-hidden="true" />
                  Проверяем…
                </>
              ) : (
                <>
                  Проверить
                  <span className="btn__arrow" aria-hidden="true">
                    →
                  </span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Примеры — это готовые тексты: в режимах файла они не нужны. */}
        {mode === 'text' && (
          <div className="chips" role="group" aria-label="Примеры сообщений">
            <span className="chips__label">Примеры</span>
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="chip"
                title={example}
                onClick={() => {
                  setText(example);
                  setError(null);
                  setResult(null);
                }}
              >
                {example.slice(0, 42)}…
              </button>
            ))}
          </div>
        )}
      </section>

      {error && (
        <p className="alert alert--bad" role="alert">
          <span aria-hidden="true">!</span>
          {error}
        </p>
      )}

      <div ref={resultRef} aria-live="polite" aria-atomic="false">
        {busy && <VerdictSkeleton />}
        {!busy && result && (
          <VerdictCard
            key={result.id}
            result={result}
            onUpgraded={() => void handleUpgraded()}
          />
        )}
      </div>

      {!busy && !result && !error && (
        <p className="hint">
          Бесплатно — вердикт и уровень доверия. По подписке — разбор по каждому сигналу
          и ссылки на первоисточники.
        </p>
      )}
    </div>
  );
}
