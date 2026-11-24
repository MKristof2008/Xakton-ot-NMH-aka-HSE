import type { Level, Signal, Verdict } from '../types';

/** Смысловой тон: зелёный — правда, красный — фейк, серый — непонятно. */
export type Tone = 'ok' | 'bad' | 'fog';

export function toneForLevel(level: Level | string): Tone {
  if (level === 'confirmed' || level === 'likely_real') return 'ok';
  if (level === 'likely_fake' || level === 'doubtful') return 'bad';
  return 'fog';
}

export function toneForSignal(verdict: Signal['verdict']): Tone {
  if (verdict === 'supports_real') return 'ok';
  if (verdict === 'supports_fake') return 'bad';
  return 'fog';
}

export function signalLabel(verdict: Signal['verdict']): string {
  return {
    supports_real: 'подтверждает',
    supports_fake: 'опровергает',
    neutral: 'нейтрально',
    unavailable: 'нет данных',
  }[verdict];
}

/** Пояснение вердикта одной фразой — под крупным словом.

 * Фраза общая и всегда правдивая: причину называет разбор ниже
 * («Почему такой вердикт»). Раньше здесь стояло «База фактов утверждение
 * опровергает» — это было неверно, когда вердикт ставился по признакам
 * жанра вирусной выдумки, а не по опровержению источника.
 */
export function verdictSubline(verdict: Verdict): string {
  if (verdict === 'правда') return 'Нашлось подтверждение в проверенных источниках.';
  if (verdict === 'фейк') return 'Нашлось весомое основание считать сообщение подделкой.';
  return 'Однозначного ответа нет: данных не хватает. Честнее сказать «непонятно», чем угадывать.';
}

export function stepLabel(kind: string): string {
  return (
    {
      hold: 'не пересылать',
      verify: 'проверить',
      share: 'можно пересылать',
      ask_source: 'спросить источник',
    }[kind] ?? 'шаг'
  );
}

export function stepTone(kind: string): Tone {
  if (kind === 'hold') return 'bad';
  if (kind === 'share') return 'ok';
  return 'fog';
}

/** Доля неуверенности 0..1 → проценты, с защитой от кривых значений. */
export function uncertaintyPercent(value: number): number {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
  return Math.round(clamped * 100);
}
