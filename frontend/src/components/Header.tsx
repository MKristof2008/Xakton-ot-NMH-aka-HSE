import { useState } from 'react';
import { logout } from '../api/client';
import type { Me } from '../types';
import type { Tab } from '../lib/tabs';
import { BrandMark } from './icons';

const TABS: { id: Tab; label: string }[] = [
  { id: 'check', label: 'Проверить' },
  { id: 'feed', label: 'Лента' },
  { id: 'autopilot', label: 'Автопилот' },
  { id: 'history', label: 'История' },
  { id: 'top', label: 'Топы' },
  { id: 'plans', label: 'Тарифы' },
  { id: 'access', label: 'Доступ' },
];

interface Props {
  tab: Tab;
  me: Me | null;
  onNavigate: (tab: Tab) => void;
  onLogout: () => void;
  onNeedAuth: (reason: string) => void;
}

/** Подпись тарифа — совпадает с названиями в /api/pricing. */
function planLabel(plan?: string): string {
  if (plan === 'pro') return 'Плюс';
  if (plan === 'team') return 'Макс';
  return 'База';
}

export function Header({ tab, me, onNavigate, onLogout, onNeedAuth }: Props) {
  const paid = me?.plan === 'pro' || me?.plan === 'team';
  const [menuOpen, setMenuOpen] = useState(false);

  async function handleLogout() {
    await logout();
    onLogout();
  }

  return (
    <header className="header">
      <div className="header__inner">
        <button className="brand" type="button" onClick={() => onNavigate('check')}>
          <span className="brand__mark" aria-hidden="true">
            <BrandMark />
          </span>
          <span className="brand__name">Ревизор</span>
        </button>

        <nav className="nav" aria-label="Разделы">
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav__link${tab === item.id ? ' is-active' : ''}`}
              aria-current={tab === item.id ? 'page' : undefined}
              onClick={() => onNavigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {!me && (
          <button
            type="button"
            className="btn btn--ghost header__login"
            onClick={() => onNeedAuth('Войдите, чтобы проверять сообщения и видеть свою историю.')}
          >
            Войти
          </button>
        )}

        {me && (
        <div className="account">
          <button
            type="button"
            className="account__btn"
            aria-expanded={menuOpen}
            aria-haspopup="menu"
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span className={`plan-chip plan-chip--${me?.plan ?? 'free'}`}>
              {/* Названия те же, что в тарифах: База / Плюс / Макс. */}
              {planLabel(me?.plan)}
            </span>
            <span className="account__avatar" aria-hidden="true">
              {(me?.name || me?.email || '?').slice(0, 1).toUpperCase()}
            </span>
          </button>

          {menuOpen && (
            <>
              <div
                className="account__scrim"
                onClick={() => setMenuOpen(false)}
                aria-hidden="true"
              />
              <div className="account__menu" role="menu">
                <div className="account__who">
                  <strong>{me?.name || 'Без имени'}</strong>
                  <span>{me?.email}</span>
                </div>
                {me?.expires_at && paid && (
                  <p className="account__row">
                    Подписка до {me.expires_at.slice(0, 10)}
                  </p>
                )}
                {typeof me?.quota_left === 'number' && !paid && (
                  <p className="account__row">
                    Осталось сегодня: {me.quota_left} из {me.quota_per_day}
                  </p>
                )}
                <button
                  type="button"
                  className="account__logout"
                  role="menuitem"
                  onClick={handleLogout}
                >
                  Выйти
                </button>
              </div>
            </>
          )}
        </div>
        )}
      </div>
    </header>
  );
}
